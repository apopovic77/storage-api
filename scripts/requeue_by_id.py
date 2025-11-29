import sys
from pathlib import Path

# Add the project root directory to sys.path
sys.path.insert(0, "/var/code/storage-api")

from artrack.database import SessionLocal
from artrack.models import StorageObject
import asyncio
from storage.service import enqueue_ai_safety_and_transcoding


async def requeue_by_id(tenant_id: str, storage_id: int):
    """Re-queue a specific storage object for AI analysis by tenant and ID."""
    db = SessionLocal()

    try:
        # Find the storage object
        obj = db.query(StorageObject).filter(
            StorageObject.tenant_id == tenant_id,
            StorageObject.id == storage_id
        ).first()

        if not obj:
            print(f"❌ Storage object with ID {storage_id} not found for tenant '{tenant_id}'")
            return

        print(f"Re-queueing {obj.id} ({obj.original_filename}) for tenant '{tenant_id}'")

        try:
            await enqueue_ai_safety_and_transcoding(obj, db=db, skip_ai_safety=False)
            print(f"  ✅ Queued for AI analysis and transcoding")
        except Exception as e:
            print(f"  ❌ Error: {e}")

    finally:
        db.close()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 requeue_by_id.py <tenant_id> <storage_id>")
        sys.exit(1)

    try:
        tenant_id = sys.argv[1]
        storage_id = int(sys.argv[2])
        asyncio.run(requeue_by_id(tenant_id, storage_id))
    except ValueError:
        print("Error: storage_id must be an integer")
        sys.exit(1)


async def requeue_by_id(tenant_id: str, storage_id: int):
    """Re-queue a specific storage object for AI analysis by tenant and ID."""
    db = SessionLocal()

    try:
        # Find the storage object
        obj = db.query(StorageObject).filter(
            StorageObject.tenant_id == tenant_id,
            StorageObject.id == storage_id
        ).first()

        if not obj:
            print(f"❌ Storage object with ID {storage_id} not found for tenant '{tenant_id}'")
            return

        print(f"Re-queueing {obj.id} ({obj.original_filename}) for tenant '{tenant_id}'")

        try:
            await enqueue_ai_safety_and_transcoding(obj, db=db, skip_ai_safety=False)
            print(f"  ✅ Queued for AI analysis and transcoding")
        except Exception as e:
            print(f"  ❌ Error: {e}")

    finally:
        db.close()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 requeue_by_id.py <tenant_id> <storage_id>")
        sys.exit(1)

    try:
        tenant_id = sys.argv[1]
        storage_id = int(sys.argv[2])
        asyncio.run(requeue_by_id(tenant_id, storage_id))
    except ValueError:
        print("Error: storage_id must be an integer")
        sys.exit(1)
