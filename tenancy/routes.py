from datetime import datetime
from typing import Dict, List, Optional
import secrets

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from auth import get_current_user
from database import get_db
from models import AsyncTask, StorageObject, User
from storage.service import generic_storage
from tenancy.config import list_tenant_keys, upsert_tenant_key, delete_tenant_key


router = APIRouter(prefix="/tenants", tags=["tenants"])
api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=True)


class TenantKeyRequest(BaseModel):
    api_key: str
    tenant_id: str


class TenantStatus(BaseModel):
    tenant_id: str
    object_count: int
    total_bytes: int
    last_object_created_at: Optional[datetime]


class TenantStatusResponse(BaseModel):
    tenants: List[TenantStatus]
    total_objects: int
    total_bytes: int


class TenantCreateRequest(BaseModel):
    tenant_id: str
    api_key: Optional[str] = None
    generate_api_key: bool = False


class TenantCreateResponse(BaseModel):
    tenant_id: str
    api_key: str
    generated_api_key: bool
    directories: List[str]


class TenantDeleteResponse(BaseModel):
    tenant_id: str
    deleted_objects: int
    deleted_bytes: int
    removed_api_keys: List[str]
    removed_directories: List[str]


@router.get("/keys")
def list_keys(current_user: User = Depends(get_current_user)) -> Dict[str, str]:
    if current_user.trust_level != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return list_tenant_keys()


@router.post("/keys")
def upsert_key(req: TenantKeyRequest, current_user: User = Depends(get_current_user)) -> Dict[str, str]:
    if current_user.trust_level != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return upsert_tenant_key(req.api_key, req.tenant_id)


@router.delete("/keys/{key}")
def delete_key(key: str, current_user: User = Depends(get_current_user)) -> Dict[str, str]:
    if current_user.trust_level != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return delete_tenant_key(key)


@router.get("/status", response_model=TenantStatusResponse)
def tenant_status(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> TenantStatusResponse:
    if current_user.trust_level != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    rows = (
        db.query(
            StorageObject.tenant_id,
            func.count(StorageObject.id),
            func.coalesce(func.sum(StorageObject.file_size_bytes), 0),
            func.max(StorageObject.created_at),
        )
        .group_by(StorageObject.tenant_id)
        .all()
    )

    configured = list_tenant_keys()
    seen: Dict[str, TenantStatus] = {}

    total_objects = 0
    total_bytes = 0
    tenants: List[TenantStatus] = []

    for tenant_id, object_count, total_bytes_sum, last_created in rows:
        tenant_id = tenant_id or "arkturian"
        status = TenantStatus(
            tenant_id=tenant_id,
            object_count=int(object_count or 0),
            total_bytes=int(total_bytes_sum or 0),
            last_object_created_at=last_created,
        )
        tenants.append(status)
        seen[tenant_id] = status
        total_objects += status.object_count
        total_bytes += status.total_bytes

    # Ensure tenants with configured keys but no objects are listed
    for key, tenant_id in configured.items():
        if tenant_id not in seen:
            tenants.append(
                TenantStatus(
                    tenant_id=tenant_id,
                    object_count=0,
                    total_bytes=0,
                    last_object_created_at=None,
                )
            )

    # Sort tenants alphabetically for stable output
    tenants.sort(key=lambda t: t.tenant_id)

    return TenantStatusResponse(
        tenants=tenants,
        total_objects=total_objects,
        total_bytes=total_bytes,
    )


@router.post("", response_model=TenantCreateResponse)
def create_tenant(
    req: TenantCreateRequest,
    current_user: User = Depends(get_current_user),
) -> TenantCreateResponse:
    if current_user.trust_level != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    tenant_id = (req.tenant_id or "").strip()
    if not tenant_id:
        raise HTTPException(status_code=400, detail="tenant_id is required")
    tenant_id = tenant_id.lower()

    configured = list_tenant_keys()
    if any(value == tenant_id for value in configured.values()):
        raise HTTPException(status_code=409, detail=f"Tenant '{tenant_id}' already exists")

    provided_key = (req.api_key or "").strip()
    if provided_key and provided_key in configured:
        raise HTTPException(status_code=409, detail="API key already in use")

    generated_api_key = False
    api_key = provided_key
    if not api_key:
        if not req.generate_api_key:
            req.generate_api_key = True
        api_key = secrets.token_urlsafe(24)
        generated_api_key = True

    directories = generic_storage.ensure_tenant_directories(tenant_id)
    upsert_tenant_key(api_key, tenant_id)

    return TenantCreateResponse(
        tenant_id=tenant_id,
        api_key=api_key,
        generated_api_key=generated_api_key,
        directories=directories,
    )


@router.delete("/{tenant_id}", response_model=TenantDeleteResponse)
def delete_tenant(
    tenant_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> TenantDeleteResponse:
    if current_user.trust_level != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    tenant_id = tenant_id.strip().lower()
    if tenant_id in {"arkturian", "oneal"}:
        raise HTTPException(status_code=400, detail="Default tenants cannot be deleted")

    objects = db.query(StorageObject).filter(StorageObject.tenant_id == tenant_id).all()
    object_ids = [obj.id for obj in objects]
    deleted_objects = len(objects)
    deleted_bytes = int(sum(obj.file_size_bytes or 0 for obj in objects))

    for obj in objects:
        try:
            generic_storage.delete(obj.object_key, tenant_id or obj.tenant_id or "arkturian")
        except Exception:
            pass

    if object_ids:
        db.query(AsyncTask).filter(AsyncTask.object_id.in_(object_ids)).delete(synchronize_session=False)
        db.query(StorageObject).filter(StorageObject.id.in_(object_ids)).delete(synchronize_session=False)
    db.commit()

    removed_dirs = generic_storage.delete_tenant_directories(tenant_id)

    removed_keys: List[str] = []
    for key, tenant in list(list_tenant_keys().items()):
        if tenant == tenant_id:
            delete_tenant_key(key)
            removed_keys.append(key)

    return TenantDeleteResponse(
        tenant_id=tenant_id,
        deleted_objects=deleted_objects,
        deleted_bytes=deleted_bytes,
        removed_api_keys=removed_keys,
        removed_directories=removed_dirs,
    )


