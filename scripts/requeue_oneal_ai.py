#!/usr/bin/env python3
"""
Re-queue O'Neal products for AI analysis.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from database import SessionLocal
from models import StorageObject
import asyncio
from storage.service import enqueue_ai_safety_and_transcoding


async def requeue_oneal_products():
    """Re-queue all O'Neal products for AI analysis."""
    db = SessionLocal()

    try:
        # Find all O'Neal storage objects
        oneal_objects = db.query(StorageObject).filter(
            StorageObject.tenant_id == "oneal",
            StorageObject.storage_mode == "external"
        ).all()

        print(f"Found {len(oneal_objects)} O'Neal products to re-queue")

        for idx, obj in enumerate(oneal_objects, 1):
            print(f"[{idx}/{len(oneal_objects)}] Re-queueing {obj.id}: {obj.original_filename}")

            try:
                await enqueue_ai_safety_and_transcoding(obj, db=db, skip_ai_safety=False)
                print(f"  ✅ Queued for AI analysis")
            except Exception as e:
                print(f"  ❌ Error: {e}")

        print(f"\n✅ Re-queued {len(oneal_objects)} products for AI analysis")

    finally:
        db.close()


if __name__ == "__main__":
    asyncio.run(requeue_oneal_products())
