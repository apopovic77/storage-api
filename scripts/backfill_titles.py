#!/usr/bin/env python
"""Backfill StorageObject.title from embedded media-container titles (ffprobe).

For existing video (and optionally audio) objects whose `title` is empty, probe
the file's container/stream tags and set the embedded title when present. Mirrors
the upload-time logic in storage.service._extract_metadata, so a re-probe yields
the same title a fresh upload would.

DB + file paths are resolved via the app config (database.SessionLocal,
generic_storage), so this runs correctly on any storage-api host. Run it from a
deployed checkout so the right .env/DB is loaded, e.g.:

    cd /var/www/api-storage.arkturian.com
    ./venv/bin/python scripts/backfill_titles.py --dry-run
    ./venv/bin/python scripts/backfill_titles.py            # apply
    ./venv/bin/python scripts/backfill_titles.py --audio    # also audio
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Allow running as `scripts/backfill_titles.py` — add the repo root to sys.path
# so the app modules (database, models, storage) import.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import or_

from database import SessionLocal
from models import StorageObject
from storage.service import generic_storage


def extract_container_title(file_path: Path):
    """Read an embedded title from ffprobe output (format tags first, then
    stream tags; tag keys are case-insensitive across containers)."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", "-show_streams", str(file_path)],
            capture_output=True, text=True, timeout=60,
        ).stdout
        pdata = json.loads(out or "{}")
    except Exception as e:  # noqa: BLE001
        print(f"    ffprobe failed: {e}")
        return None
    tags = (pdata.get("format") or {}).get("tags") or {}
    t = tags.get("title") or tags.get("TITLE")
    if not t:
        for s in pdata.get("streams", []) or []:
            st = s.get("tags") or {}
            t = st.get("title") or st.get("TITLE")
            if t:
                break
    return t.strip() if isinstance(t, str) and t.strip() else None


def resolve_path(obj):
    """Local filesystem path for an object, honouring storage_mode."""
    if obj.storage_mode == "external":
        return None  # no local file
    if obj.storage_mode == "reference" and obj.reference_path:
        return Path(obj.reference_path)
    try:
        return generic_storage.absolute_path_for_key(obj.object_key, obj.tenant_id or "arkturian")
    except Exception as e:  # noqa: BLE001
        print(f"    path-resolve error: {e}")
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="probe + report, write nothing")
    ap.add_argument("--limit", type=int, default=0, help="max objects to process (0 = all)")
    ap.add_argument("--audio", action="store_true", help="also backfill audio (default: video only)")
    args = ap.parse_args()

    mime_filter = (
        or_(StorageObject.mime_type.like("video/%"), StorageObject.mime_type.like("audio/%"))
        if args.audio else StorageObject.mime_type.like("video/%")
    )

    db = SessionLocal()
    try:
        q = (
            db.query(StorageObject)
            .filter(mime_filter)
            .filter(or_(StorageObject.title.is_(None), StorageObject.title == ""))
            .order_by(StorageObject.id)
        )
        if args.limit:
            q = q.limit(args.limit)
        objs = q.all()

        print(f"Candidates ({'video+audio' if args.audio else 'video'}, empty title): {len(objs)}")
        updated = missing = no_title = 0
        for o in objs:
            path = resolve_path(o)
            if not path or not Path(path).exists():
                print(f"  #{o.id} file missing: {path}")
                missing += 1
                continue
            title = extract_container_title(Path(path))
            if not title:
                no_title += 1
                continue
            print(f"  #{o.id} -> title={title!r}  ({o.original_filename})")
            if not args.dry_run:
                o.title = title
                db.commit()
            updated += 1

        verb = "(dry-run) would update" if args.dry_run else "updated"
        print(f"\nDone. {verb}: {updated} | no embedded title: {no_title} | file missing: {missing}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
