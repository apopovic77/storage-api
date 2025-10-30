"""
Tenant configuration: map API keys to tenant identifiers.

Backed by a JSON file for runtime updates via admin API.
"""

from typing import Optional, Dict
from pathlib import Path
from fastapi import Header
import json
import os


TENANT_CONFIG_PATH = Path(os.getenv("TENANT_CONFIG_PATH", "./tenant_config.json"))
_DEFAULT_MAP: Dict[str, str] = {
    "Inetpass1": "arkturian",
    "oneal_demo_token": "oneal",  # O'Neal tenant access
}


def _load_map() -> Dict[str, str]:
    try:
        if TENANT_CONFIG_PATH.exists():
            with open(TENANT_CONFIG_PATH, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                # Normalize keys/values to strings
                return {str(k): str(v) for k, v in data.items()}
    except Exception:
        pass
    return dict(_DEFAULT_MAP)


def _save_map(map_: Dict[str, str]) -> None:
    try:
        TENANT_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    with open(TENANT_CONFIG_PATH, "w") as f:
        json.dump(map_, f, indent=2, sort_keys=True)


def tenant_id_for_api_key(api_key: Optional[str]) -> Optional[str]:
    if not api_key:
        return None
    return _load_map().get(api_key)


def list_tenant_keys() -> Dict[str, str]:
    return _load_map()


def api_key_for_tenant(tenant_id: str) -> Optional[str]:
    """Reverse lookup helper to find an API key for a given tenant."""
    if not tenant_id:
        return None
    for key, value in _load_map().items():
        if value == tenant_id:
            return key
    return None


def upsert_tenant_key(api_key: str, tenant_id: str) -> Dict[str, str]:
    api_key = str(api_key)
    tenant_id = str(tenant_id)
    m = _load_map()
    m[api_key] = tenant_id
    _save_map(m)
    return m


def delete_tenant_key(api_key: str) -> Dict[str, str]:
    api_key = str(api_key)
    m = _load_map()
    if api_key in m:
        del m[api_key]
        _save_map(m)
    return m


def get_tenant_id(api_key: str = Header(None, alias="X-API-KEY")) -> str:
    """
    FastAPI Dependency: Extract tenant_id from API key.

    Returns the tenant_id associated with the API key.
    Defaults to 'arkturian' if no mapping exists.
    """
    tenant_id = tenant_id_for_api_key(api_key)
    if not tenant_id:
        # Fallback to default tenant for backward compatibility
        return "arkturian"
    return tenant_id

def get_tenant_id_optional(api_key: str = Header(None, alias="X-API-KEY")) -> Optional[str]:
    """
    FastAPI Dependency: Extract tenant_id from API key (optional).

    Returns the tenant_id associated with the API key or None if no API key provided.
    Used for endpoints that support both authenticated and public access.
    """
    return tenant_id_for_api_key(api_key)
