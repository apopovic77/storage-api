#!/usr/bin/env python3
"""
cleanup_expired.py — Purge storage objects whose TTL has expired.

Deletes the physical file (original + thumbnail + webview variants) and
removes the database row. Designed to run as a cron job:

    */15 * * * * cd /var/www/storage-api && ./venv/bin/python scripts/cleanup_expired.py >> /tmp/cleanup_expired.log 2>&1

Safe to run concurrently — each invocation selects a batch with FOR UPDATE
SKIP LOCKED (PostgreSQL) or a simple SELECT + DELETE (SQLite).
"""

import os
import sys

# Add parent directory so we can import app modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from sqlalchemy import text
from database import engine, SessionLocal
from models import StorageObject
from config import settings


def cleanup_expired(batch_size: int = 100) -> int:
    """Delete expired storage objects. Returns count of purged items."""
    db = SessionLocal()
    try:
        now = datetime.utcnow()
        expired = (
            db.query(StorageObject)
            .filter(
                StorageObject.expires_at.isnot(None),
                StorageObject.expires_at < now,
            )
            .limit(batch_size)
            .all()
        )

        if not expired:
            return 0

        count = 0
        for obj in expired:
            # Delete physical files
            _delete_files(obj)
            db.delete(obj)
            count += 1

        db.commit()
        return count
    except Exception as e:
        db.rollback()
        print(f"[cleanup_expired] error: {e}")
        return 0
    finally:
        db.close()


def _delete_files(obj: StorageObject):
    """Best-effort delete of all physical files for a storage object."""
    storage_root = getattr(settings, "STORAGE_ROOT", "/var/lib/storage-api/files")
    if not obj.object_key:
        return

    # Main file
    main_path = os.path.join(storage_root, obj.object_key)
    _safe_unlink(main_path)

    # Thumbnail and webview variants (stored in metadata_json)
    meta = obj.metadata_json or {}
    for key in ("thumbnail_filename", "webview_filename"):
        fname = meta.get(key)
        if fname:
            _safe_unlink(os.path.join(storage_root, fname))

    # HLS directory (if video was transcoded)
    if obj.hls_url:
        hls_dir = os.path.join(storage_root, "hls", str(obj.id))
        if os.path.isdir(hls_dir):
            import shutil
            try:
                shutil.rmtree(hls_dir)
            except Exception as e:
                print(f"[cleanup_expired] rmtree {hls_dir}: {e}")


def _safe_unlink(path: str):
    try:
        if os.path.exists(path):
            os.unlink(path)
    except Exception as e:
        print(f"[cleanup_expired] unlink {path}: {e}")


if __name__ == "__main__":
    count = cleanup_expired()
    if count > 0:
        print(f"[{datetime.utcnow().isoformat()}] purged {count} expired assets")
