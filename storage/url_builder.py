"""
Dynamic URL builder for storage objects.

URLs are built dynamically based on the current request context,
not stored in the database. This allows for:
- Easy domain/server changes
- Development vs production flexibility
- No database migrations when URLs change
"""

from typing import Optional
from fastapi import Request


def build_storage_urls(
    object_id: int,
    tenant_id: str,
    checksum: Optional[str] = None,
    metadata_json: Optional[dict] = None,
    base_url: Optional[str] = None,
    storage_mode: str = "copy",
    stored_file_url: Optional[str] = None,
) -> dict:
    """
    Build URLs dynamically for a storage object.

    Args:
        object_id: Storage object ID
        tenant_id: Tenant ID
        checksum: File checksum for cache busting
        metadata_json: Metadata containing thumbnail_filename, webview_filename, etc.
        base_url: Base URL of the API (e.g., "https://api-storage.arkturian.com")
        storage_mode: Storage mode (copy, reference, external)
        stored_file_url: For external mode, use the stored proxy URL

    Returns:
        Dict with file_url, thumbnail_url, webview_url
    """
    if not base_url:
        # Fallback to default
        base_url = "https://api-storage.arkturian.com"

    base_url = base_url.rstrip("/")

    # For external mode, use the stored proxy URL if available
    if storage_mode == "external" and stored_file_url:
        file_url = stored_file_url
    else:
        # Main file URL uses the /storage/media/{id} endpoint
        file_url = f"{base_url}/storage/media/{object_id}"
        if checksum:
            file_url = f"{file_url}?v={checksum}"

    # Thumbnail URL (also uses media endpoint with variant parameter)
    thumbnail_url = None
    if metadata_json and metadata_json.get("thumbnail_filename"):
        # Use the media endpoint with variant=thumbnail
        thumbnail_url = f"{base_url}/storage/media/{object_id}?variant=thumbnail"
        if checksum:
            thumbnail_url = f"{thumbnail_url}&v={checksum}"

    # Webview URL (medium quality variant)
    webview_url = None
    if metadata_json and metadata_json.get("webview_filename"):
        webview_url = f"{base_url}/storage/media/{object_id}?variant=medium"
        if checksum:
            webview_url = f"{webview_url}&v={checksum}"

    return {
        "file_url": file_url,
        "thumbnail_url": thumbnail_url,
        "webview_url": webview_url,
    }


def get_base_url_from_request(request: Request) -> str:
    """
    Extract base URL from FastAPI request.

    Args:
        request: FastAPI Request object

    Returns:
        Base URL (e.g., "https://api-storage.arkturian.com")
    """
    # Get scheme (http/https)
    scheme = request.url.scheme

    # Get host (includes port if non-standard)
    host = request.headers.get("host") or request.client.host

    # Check for X-Forwarded-* headers (for reverse proxy scenarios)
    forwarded_proto = request.headers.get("x-forwarded-proto")
    forwarded_host = request.headers.get("x-forwarded-host")

    if forwarded_proto:
        scheme = forwarded_proto
    if forwarded_host:
        host = forwarded_host

    # Clean scheme: remove any trailing slashes, backslashes, or colons
    if scheme:
        scheme = scheme.strip().rstrip(":/\\")

    # Clean up host: remove leading/trailing slashes or protocol prefixes
    if host:
        host = host.strip()
        # Remove leading slashes and backslashes
        host = host.lstrip("/\\")
        # Remove protocol if accidentally included
        if host.startswith("http://"):
            host = host[7:]
        elif host.startswith("https://"):
            host = host[8:]
        # Remove trailing slashes and backslashes
        host = host.rstrip("/\\")

    return f"{scheme}://{host}"
