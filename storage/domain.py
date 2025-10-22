from __future__ import annotations

from typing import Optional
from sqlalchemy.orm import Session

from artrack.models import StorageObject
from storage.service import generic_storage


async def save_file_and_record(
    db: Session,
    *,
    owner_user_id: int,
    data: bytes,
    original_filename: str,
    context: Optional[str] = None,
    is_public: bool = False,
    collection_id: Optional[str] = None,
    link_id: Optional[str] = None,
    title: Optional[str] = None,
    description: Optional[str] = None,
    storage_mode: str = "copy",
    reference_path: Optional[str] = None,
    external_uri: Optional[str] = None,
    ai_context_metadata: Optional[dict] = None,
    tenant_id: Optional[str] = None,
) -> StorageObject:
    """Save file via GenericStorageService and create a StorageObject row.

    Args:
        storage_mode: "copy" (default) saves file to storage, "reference" for filesystem path, "external" for web URI
        reference_path: Required when storage_mode="reference" - path to existing file (e.g., "/mnt/oneal/2026/Helmets/Airframe.jpg")
        external_uri: Required when storage_mode="external" - external web URI (file remains on external server)
        ai_context_metadata: Optional context for AI analysis (file_path, brand, year, category, etc.)

    Returns the persisted StorageObject.
    """
    # External mode - file stays on external server, only metadata saved
    if storage_mode == "external":
        if not external_uri:
            raise ValueError("external_uri is required when storage_mode='external'")

        # Generate a pseudo object_key for the database
        import hashlib
        from datetime import datetime
        hash_suffix = hashlib.sha256(external_uri.encode()).hexdigest()[:8]
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        object_key = f"ext_{timestamp}_{hash_suffix}"

        # Use proxy URL as file_url
        file_url = f"/storage/proxy/PLACEHOLDER"  # Will be updated after object creation

        # Try to determine MIME type from URI or data
        mime_type = "application/octet-stream"
        if data and len(data) > 0:
            import magic
            try:
                mime_type = magic.from_buffer(data, mime=True)
            except:
                pass

        saved = {
            "object_key": object_key,
            "original_filename": original_filename,
            "file_url": file_url,
            "thumbnail_url": None,
            "webview_url": None,
            "mime_type": mime_type,
            "file_size_bytes": len(data) if data else 0,
            "checksum": hashlib.sha256(data if data else b"").hexdigest(),
        }

    # In reference mode, skip saving the original file but still generate thumbnails
    elif storage_mode == "reference":
        if not reference_path:
            raise ValueError("reference_path is required when storage_mode='reference'")

        # Generate thumbnails/variants from the referenced file, but don't copy original
        saved = await generic_storage.save_reference(
            data=data,
            original_filename=original_filename,
            owner_user_id=owner_user_id,
            context=context,
            reference_path=reference_path,
            tenant_id=tenant_id or "arkturian",
        )
    else:
        # Normal copy mode - save everything
        saved = await generic_storage.save(
            data=data,
            original_filename=original_filename,
            owner_user_id=owner_user_id,
            context=context,
            tenant_id=tenant_id or "arkturian",
        )

    storage_obj = StorageObject(
        owner_user_id=owner_user_id,
        tenant_id=tenant_id or "arkturian",
        object_key=saved["object_key"],
        original_filename=saved["original_filename"],
        file_url=saved["file_url"],
        thumbnail_url=saved.get("thumbnail_url"),
        webview_url=saved.get("webview_url"),
        mime_type=saved["mime_type"],
        file_size_bytes=saved["file_size_bytes"],
        checksum=saved["checksum"],
        is_public=is_public,
        context=context,
        collection_id=collection_id,
        link_id=link_id,
        title=title,
        description=description,
        width=saved.get("width"),
        height=saved.get("height"),
        duration_seconds=saved.get("duration_seconds"),
        bit_rate=saved.get("bit_rate"),
        latitude=saved.get("latitude"),
        longitude=saved.get("longitude"),
        metadata_json={},
        storage_mode=storage_mode,
        reference_path=reference_path,
        external_uri=external_uri,
        ai_context_metadata=ai_context_metadata or {},
    )
    db.add(storage_obj)
    db.commit()
    db.refresh(storage_obj)

    # Update file_url with actual object ID for external mode
    if storage_mode == "external":
        # Use public share.arkturian.com proxy for external URIs
        storage_obj.file_url = f"https://share.arkturian.com/proxy.php?id={storage_obj.id}"
        db.commit()
        db.refresh(storage_obj)

    return storage_obj


async def update_file_and_record(
    db: Session,
    *,
    storage_obj: StorageObject,
    data: bytes,
    context: Optional[str] = None,
) -> StorageObject:
    """Overwrite file via GenericStorageService and update StorageObject row."""
    updated = await generic_storage.update_file(
        storage_obj.object_key,
        data,
        tenant_id=storage_obj.tenant_id or "arkturian"
    )

    storage_obj.file_url = updated["file_url"]
    storage_obj.thumbnail_url = updated.get("thumbnail_url")
    storage_obj.webview_url = updated.get("webview_url")
    storage_obj.mime_type = updated["mime_type"]
    storage_obj.file_size_bytes = updated["file_size_bytes"]
    storage_obj.checksum = updated["checksum"]
    storage_obj.width = updated.get("width")
    storage_obj.height = updated.get("height")
    storage_obj.duration_seconds = updated.get("duration_seconds")
    storage_obj.bit_rate = updated.get("bit_rate")
    if context is not None:
        storage_obj.context = context
    db.commit()
    db.refresh(storage_obj)
    return storage_obj
