import argparse
import os
import sys
import shutil
from datetime import datetime, timedelta
from sqlalchemy import create_engine, or_
from sqlalchemy.orm import sessionmaker
from pathlib import Path

# Add the project root to the Python path to allow importing project modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from artrack.models import Base, StorageObject
from artrack.config import settings

DATABASE_URL = settings.DATABASE_URL
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# This is a simplified path resolver. For robustness, it should be identical to
# the one in the main application's storage_service.py.
UPLOAD_ROOT = Path(settings.STORAGE_UPLOAD_DIR)
MEDIA_DIR = UPLOAD_ROOT / "media"
THUMBNAILS_DIR = UPLOAD_ROOT / "thumbnails"
WEBVIEW_DIR = UPLOAD_ROOT / "webview"


def delete_storage_object_files(storage_obj: StorageObject, dry_run=False):
    """Deletes all files associated with a StorageObject."""
    print(f"Processing object ID {storage_obj.id} (key: {storage_obj.object_key})...")

    # 1. Delete main media file
    media_file = MEDIA_DIR / storage_obj.object_key
    if media_file.exists():
        print(f"  - Deleting media file: {media_file}")
        if not dry_run:
            media_file.unlink()
    else:
        print(f"  - Media file not found (already deleted?): {media_file}")

    # 2. Delete thumbnail
    if storage_obj.thumbnail_url:
        thumb_file = THUMBNAILS_DIR / Path(storage_obj.thumbnail_url).name
        if thumb_file.exists():
            print(f"  - Deleting thumbnail: {thumb_file}")
            if not dry_run:
                thumb_file.unlink()

    # 3. Delete webview version
    if storage_obj.webview_url:
        webview_file = WEBVIEW_DIR / Path(storage_obj.webview_url).name
        if webview_file.exists():
            print(f"  - Deleting webview file: {webview_file}")
            if not dry_run:
                webview_file.unlink()

    # 4. Delete HLS directory
    hls_dir = MEDIA_DIR / media_file.stem
    if hls_dir.is_dir():
        print(f"  - Deleting HLS directory: {hls_dir}")
        if not dry_run:
            shutil.rmtree(hls_dir)


def cleanup_by_days(db, days, dry_run):
    """Deletes all storage objects older than a certain number of days."""
    if days <= 0:
        print("Error: --days must be a positive number.")
        return

    cutoff_date = datetime.utcnow() - timedelta(days=days)
    print(f"Querying for objects created before {cutoff_date.strftime('%Y-%m-%d %H:%M:%S')}")

    objects_to_delete = db.query(StorageObject).filter(StorageObject.created_at < cutoff_date).all()

    if not objects_to_delete:
        print("No objects found older than the specified date.")
        return

    print(f"Found {len(objects_to_delete)} object(s) to delete.")
    for obj in objects_to_delete:
        delete_storage_object_files(obj, dry_run)
        if not dry_run:
            db.delete(obj)

    if not dry_run:
        db.commit()
    print("Cleanup by date completed.")


def cleanup_all(db, dry_run):
    """Deletes ALL storage objects and their corresponding files."""
    print("Querying for all storage objects...")
    all_objects = db.query(StorageObject).all()

    if not all_objects:
        print("No objects found in the database.")
    else:
        print(f"Found {len(all_objects)} object(s) to delete.")
        for obj in all_objects:
            delete_storage_object_files(obj, dry_run)

        print("Deleting all entries from the storage_objects table...")
        if not dry_run:
            db.query(StorageObject).delete()
            db.commit()

    # Also clean up directories in case of orphaned files
    print("Wiping storage directories...")
    for directory in [MEDIA_DIR, THUMBNAILS_DIR, WEBVIEW_DIR]:
        if directory.exists():
            print(f"  - Wiping {directory}")
            if not dry_run:
                shutil.rmtree(directory)
                directory.mkdir()

    print("Full cleanup completed.")


def main():
    parser = argparse.ArgumentParser(description="Clean up storage files and database entries.")
    parser.add_argument(
        "--days",
        type=int,
        help="Delete all files older than this many days."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Delete ALL files and objects from storage."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting anything."
    )

    args = parser.parse_args()

    if not args.days and not args.all:
        parser.print_help()
        print("\nError: You must specify either --days or --all.")
        sys.exit(1)

    if args.days and args.all:
        print("\nError: --days and --all are mutually exclusive.")
        sys.exit(1)

    if args.dry_run:
        print("--- DRY RUN MODE ---")
        print("No files or database records will be changed.")
        print("-" * 20)

    db = SessionLocal()
    try:
        if args.all:
            if not args.dry_run:
                confirm = input("Are you sure you want to delete ALL storage objects and files? This is irreversible. (yes/no):")
                if confirm.lower() != 'yes':
                    print("Aborted.")
                    sys.exit(0)
            cleanup_all(db, args.dry_run)
        elif args.days:
            cleanup_by_days(db, args.days, args.dry_run)
    finally:
        db.close()

    print("-" * 20)
    if args.dry_run:
        print("--- DRY RUN FINISHED ---")
    else:
        print("--- SCRIPT FINISHED ---")


if __name__ == "__main__":
    main()
