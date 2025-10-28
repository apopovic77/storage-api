#!/usr/bin/env python3
"""
Storage Cache Cleanup Script

Manages cache size by deleting old files when cache exceeds limit.
Uses LRU (Least Recently Used) strategy based on file access time.

Usage:
    python cleanup_cache.py --max-size 1G
    python cleanup_cache.py --max-size 500M --dry-run
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Tuple
import argparse


def parse_size(size_str: str) -> int:
    """Parse size string like '1G', '500M', '100K' to bytes."""
    size_str = size_str.upper().strip()

    multipliers = {
        'K': 1024,
        'M': 1024 * 1024,
        'G': 1024 * 1024 * 1024,
        'T': 1024 * 1024 * 1024 * 1024,
    }

    if size_str[-1] in multipliers:
        number = float(size_str[:-1])
        return int(number * multipliers[size_str[-1]])

    return int(size_str)


def get_cache_files(cache_dir: Path) -> List[Tuple[Path, int, float]]:
    """
    Get all cache files with their size and access time.

    Returns list of (path, size_bytes, access_time) sorted by access time (oldest first).
    """
    files = []

    for root, dirs, filenames in os.walk(cache_dir):
        for filename in filenames:
            filepath = Path(root) / filename
            try:
                stat = filepath.stat()
                files.append((filepath, stat.st_size, stat.st_atime))
            except (OSError, PermissionError):
                continue

    # Sort by access time (oldest first)
    files.sort(key=lambda x: x[2])

    return files


def format_size(bytes: int) -> str:
    """Format bytes to human readable size."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes < 1024.0:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.2f} PB"


def cleanup_cache(cache_dir: Path, max_size: int, dry_run: bool = False) -> dict:
    """
    Clean up cache directory to stay under max_size.

    Args:
        cache_dir: Path to cache directory
        max_size: Maximum cache size in bytes
        dry_run: If True, don't actually delete files

    Returns:
        dict with statistics
    """
    print(f"📂 Scanning cache directory: {cache_dir}")
    print(f"📏 Max cache size: {format_size(max_size)}")
    print()

    # Get all files
    files = get_cache_files(cache_dir)

    if not files:
        print("✓ Cache is empty")
        return {"total_size": 0, "files_deleted": 0, "bytes_freed": 0}

    # Calculate total size
    total_size = sum(size for _, size, _ in files)
    total_files = len(files)

    print(f"📊 Current cache stats:")
    print(f"  Files: {total_files:,}")
    print(f"  Size:  {format_size(total_size)}")
    print()

    # Check if cleanup needed
    if total_size <= max_size:
        print(f"✓ Cache size ({format_size(total_size)}) is under limit ({format_size(max_size)})")
        print(f"  No cleanup needed!")
        return {"total_size": total_size, "files_deleted": 0, "bytes_freed": 0}

    # Calculate how much to delete
    bytes_to_free = total_size - max_size
    print(f"⚠️  Cache exceeds limit by {format_size(bytes_to_free)}")
    print(f"🗑️  Need to delete: {format_size(bytes_to_free)}")
    print()

    # Delete oldest files until under limit
    deleted_files = 0
    bytes_freed = 0

    if dry_run:
        print("🔍 DRY RUN - Files that would be deleted:")
        print()
    else:
        print("🗑️  Deleting old cache files...")
        print()

    for filepath, size, access_time in files:
        if bytes_freed >= bytes_to_free:
            break

        # Show file info
        age_days = (time.time() - access_time) / 86400
        relative_path = filepath.relative_to(cache_dir)

        print(f"  {'[DRY RUN] ' if dry_run else ''}Deleting: {relative_path}")
        print(f"    Size: {format_size(size)}, Last access: {age_days:.1f} days ago")

        if not dry_run:
            try:
                filepath.unlink()
                deleted_files += 1
                bytes_freed += size
            except (OSError, PermissionError) as e:
                print(f"    ⚠️  Failed to delete: {e}")
                continue
        else:
            deleted_files += 1
            bytes_freed += size

    print()
    print("=" * 60)
    print("📊 Cleanup Summary:")
    print(f"  Files deleted: {deleted_files:,}")
    print(f"  Space freed:   {format_size(bytes_freed)}")
    print(f"  New size:      {format_size(total_size - bytes_freed)}")
    print(f"  Remaining:     {total_files - deleted_files:,} files")
    print()

    if dry_run:
        print("🔍 This was a DRY RUN - no files were actually deleted")
    else:
        print("✓ Cleanup complete!")

    return {
        "total_size": total_size,
        "files_deleted": deleted_files,
        "bytes_freed": bytes_freed,
        "new_size": total_size - bytes_freed,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Clean up storage cache by removing old files when cache exceeds size limit"
    )
    parser.add_argument(
        "--max-size",
        type=str,
        default="1G",
        help="Maximum cache size (e.g. 1G, 500M, 100K). Default: 1G"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/var/www/uploads/storage/webview",
        help="Cache directory path. Default: /var/www/uploads/storage/webview"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting"
    )

    args = parser.parse_args()

    # Parse size
    try:
        max_size = parse_size(args.max_size)
    except ValueError:
        print(f"❌ Invalid size format: {args.max_size}")
        print("   Use format like: 1G, 500M, 100K")
        sys.exit(1)

    # Check cache dir exists
    cache_dir = Path(args.cache_dir)
    if not cache_dir.exists():
        print(f"❌ Cache directory not found: {cache_dir}")
        sys.exit(1)

    # Run cleanup
    try:
        cleanup_cache(cache_dir, max_size, args.dry_run)
    except KeyboardInterrupt:
        print("\n❌ Cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
