#!/usr/bin/env python3
"""
Batch retranscode all videos in a collection

Usage:
    python batch_retranscode_collection.py COLLECTION_ID

Example:
    python batch_retranscode_collection.py TAROT
"""

import sys
import asyncio
import shutil
from pathlib import Path
from database import SessionLocal
from models import StorageObject
from transcoding import TranscoderFactory, TranscodingConfig, TranscodingMode


async def retranscode_video(video_id: int, source_file: Path, hls_dir: Path) -> bool:
    """Retranscode a single video and update HLS files"""
    temp_output = Path(f"/tmp/retranscode_{video_id}")

    try:
        print(f"  📹 Retranscoding video {video_id}...")

        # Clear temp
        if temp_output.exists():
            shutil.rmtree(temp_output)
        temp_output.mkdir()

        # Transcode
        config = TranscodingConfig(mode=TranscodingMode.LOCAL)
        transcoder = TranscoderFactory.create(config)
        result = await transcoder.transcode(source_file, temp_output)

        if not result.success:
            print(f"    ❌ Failed: {result.error}")
            return False

        if len(result.variants) == 0:
            print(f"    ⚠️  No variants generated")
            return False

        print(f"    ✅ Generated {len(result.variants)} variants")

        # Remove old HLS files (but keep thumbnails)
        for f in hls_dir.glob("*.m3u8"):
            f.unlink()
        for f in hls_dir.glob("*.ts"):
            f.unlink()

        # Copy new HLS files
        copied = 0
        for f in temp_output.glob("*.m3u8"):
            shutil.copy2(f, hls_dir / f.name)
            copied += 1
        for f in temp_output.glob("*.ts"):
            shutil.copy2(f, hls_dir / f.name)
            copied += 1

        print(f"    📦 Copied {copied} HLS files")

        # Cleanup
        shutil.rmtree(temp_output)
        return True

    except Exception as e:
        print(f"    ❌ Error: {e}")
        if temp_output.exists():
            shutil.rmtree(temp_output)
        return False


async def batch_retranscode_collection(collection_id: str):
    """Batch retranscode all videos in a collection"""
    db = SessionLocal()

    try:
        # Get all videos in collection
        videos = db.query(StorageObject).filter(
            StorageObject.collection_id == collection_id,
            StorageObject.mime_type.like("video/%"),
            StorageObject.transcoding_status == "completed"
        ).all()

        if not videos:
            print(f"❌ No completed videos found in collection '{collection_id}'")
            return

        print(f"🎬 Found {len(videos)} videos in collection '{collection_id}'")
        print(f"📋 Starting batch retranscode...\n")

        success_count = 0
        failed_count = 0

        for i, video in enumerate(videos, 1):
            print(f"[{i}/{len(videos)}] Video ID {video.id} - {video.original_filename}")

            # Get source file and HLS directory
            source_file = Path(f"/mnt/backup-disk/uploads/storage/media/{video.tenant_id}/{video.object_key}")
            basename = video.object_key.rsplit('.', 1)[0]
            hls_dir = source_file.parent / basename

            if not source_file.exists():
                print(f"    ⚠️  Source file not found: {source_file}")
                failed_count += 1
                continue

            if not hls_dir.exists():
                hls_dir.mkdir(parents=True, exist_ok=True)

            # Retranscode
            success = await retranscode_video(video.id, source_file, hls_dir)

            if success:
                success_count += 1
            else:
                failed_count += 1

            print()

        print(f"\n{'='*60}")
        print(f"✅ Successfully retranscoded: {success_count}/{len(videos)}")
        print(f"❌ Failed: {failed_count}/{len(videos)}")
        print(f"{'='*60}")

    finally:
        db.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python batch_retranscode_collection.py COLLECTION_ID")
        print("Example: python batch_retranscode_collection.py TAROT")
        sys.exit(1)

    collection_id = sys.argv[1]
    asyncio.run(batch_retranscode_collection(collection_id))
