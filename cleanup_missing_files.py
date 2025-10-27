#!/usr/bin/env python3
"""
Cleanup Missing Storage Objects

Identifies and marks/deletes storage objects whose physical files are missing.
Preserves external references (storage_mode='external').

Usage:
    python cleanup_missing_files.py --dry-run  # Preview changes
    python cleanup_missing_files.py --delete   # Actually delete from DB
    python cleanup_missing_files.py --mark-missing  # Mark as missing (add flag)
"""

import sys
from pathlib import Path
from typing import List
import argparse

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from sqlalchemy.orm import Session
from database import get_db, database
from models import StorageObject
from storage.service import GenericStorageService

# Initialize storage service
storage_service = GenericStorageService()


def check_file_exists(obj: StorageObject) -> bool:
    """Check if physical file exists for storage object."""
    # External references don't need physical files
    if obj.storage_mode == "external":
        return True
    
    # Reference mode - file might be elsewhere
    if obj.storage_mode == "reference":
        if obj.reference_path:
            return Path(obj.reference_path).exists()
        return False
    
    # Copy mode - check if file exists in uploads
    if obj.object_key:
        path = storage_service.absolute_path_for_key(obj.object_key, obj.tenant_id or "arkturian")
        return path.exists()
    
    return False


def analyze_storage(db: Session):
    """Analyze storage objects and find missing files."""
    print("=" * 80)
    print("📊 Storage Analysis")
    print("=" * 80)
    
    all_objects = db.query(StorageObject).all()
    total = len(all_objects)
    
    external_refs = []
    missing_files = []
    existing_files = []
    
    for obj in all_objects:
        if obj.storage_mode == "external":
            external_refs.append(obj)
        elif check_file_exists(obj):
            existing_files.append(obj)
        else:
            missing_files.append(obj)
    
    print(f"\n📁 Total Objects: {total}")
    print(f"   ✅ External Refs: {len(external_refs)} (O'Neal products, etc.)")
    print(f"   ✅ Existing Files: {len(existing_files)}")
    print(f"   ❌ Missing Files:  {len(missing_files)}")
    
    if missing_files:
        print(f"\n⚠️  {len(missing_files)} objects reference files that don't exist!")
        print("\nSample missing objects:")
        for obj in missing_files[:5]:
            print(f"   ID {obj.id}: {obj.original_filename} (created: {obj.created_at})")
    
    return {
        "total": total,
        "external_refs": external_refs,
        "existing_files": existing_files,
        "missing_files": missing_files
    }


def mark_missing(db: Session, missing_files: List[StorageObject], dry_run: bool = True):
    """Mark objects as having missing files."""
    print(f"\n{'[DRY RUN] ' if dry_run else ''}📝 Marking {len(missing_files)} objects as missing...")
    
    for obj in missing_files:
        if not obj.metadata_json:
            obj.metadata_json = {}
        obj.metadata_json["file_missing"] = True
        obj.metadata_json["missing_since"] = "2025-10-27"
        
        if not dry_run:
            db.add(obj)
    
    if not dry_run:
        db.commit()
        print(f"✅ Marked {len(missing_files)} objects as missing")
    else:
        print(f"   Would mark {len(missing_files)} objects")


def delete_missing(db: Session, missing_files: List[StorageObject], dry_run: bool = True):
    """Delete objects with missing files."""
    print(f"\n{'[DRY RUN] ' if dry_run else ''}🗑️  Deleting {len(missing_files)} objects...")
    
    if dry_run:
        print(f"   Would delete {len(missing_files)} objects from database")
        return
    
    for obj in missing_files:
        db.delete(obj)
    
    db.commit()
    print(f"✅ Deleted {len(missing_files)} objects")


def main():
    parser = argparse.ArgumentParser(description="Cleanup missing storage objects")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true", help="Preview changes without modifying database")
    group.add_argument("--mark-missing", action="store_true", help="Mark missing objects with metadata flag")
    group.add_argument("--delete", action="store_true", help="Delete missing objects from database")
    
    args = parser.parse_args()
    
    # Connect to database
    db = next(get_db())
    
    try:
        # Analyze
        results = analyze_storage(db)
        missing = results["missing_files"]
        
        if not missing:
            print("\n✅ No missing files found!")
            return 0
        
        # Execute action
        if args.dry_run:
            print("\n" + "=" * 80)
            print("DRY RUN - No changes will be made")
            print("=" * 80)
            print("\nOptions:")
            print("  --mark-missing  Add 'file_missing' flag to metadata")
            print("  --delete        Remove from database entirely")
        elif args.mark_missing:
            mark_missing(db, missing, dry_run=False)
        elif args.delete:
            print("\n⚠️  WARNING: This will permanently delete objects from database!")
            response = input("Type 'yes' to confirm: ")
            if response.lower() == "yes":
                delete_missing(db, missing, dry_run=False)
            else:
                print("❌ Cancelled")
                return 1
        
        print("\n" + "=" * 80)
        print("✅ Cleanup complete")
        print("=" * 80)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())

