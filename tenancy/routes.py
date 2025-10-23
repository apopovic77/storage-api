from fastapi import APIRouter, Depends, HTTPException
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from typing import Dict

from auth import get_current_user
from models import User
from tenant_config import list_tenant_keys, upsert_tenant_key, delete_tenant_key


router = APIRouter(prefix="/tenants", tags=["tenants"])
api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=True)


class TenantKeyRequest(BaseModel):
    api_key: str
    tenant_id: str


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


