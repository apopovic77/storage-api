from datetime import datetime
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from auth import get_current_user
from database import get_db
from models import StorageObject, User
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


