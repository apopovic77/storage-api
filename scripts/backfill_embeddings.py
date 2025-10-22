import asyncio
import os
from typing import Optional

from sqlalchemy.orm import Session

from database import get_db
from models import StorageObject


async def backfill_for_user(db: Session, owner_user_id: Optional[int] = None, tenant_domain: Optional[str] = None, limit: Optional[int] = None):
    from knowledge_graph.pipeline import kg_pipeline

    q = db.query(StorageObject)
    if owner_user_id is not None:
        q = q.filter(StorageObject.owner_user_id == owner_user_id)

    # Optional filter by collection or other scopes via env
    collection_like = os.getenv("BACKFILL_COLLECTION_LIKE")
    if collection_like:
        from sqlalchemy import func
        q = q.filter(func.coalesce(StorageObject.collection_id, "").ilike(f"%{collection_like}%"))

    if limit:
        q = q.limit(limit)

    items = q.all()
    print(f"Backfilling embeddings for {len(items)} objects...")

    count = 0
    for obj in items:
        try:
            result = await kg_pipeline.process_storage_object(obj, db)
            if result:
                count += 1
        except Exception as e:
            print(f"Failed processing object {obj.id}: {e}")
    print(f"Done. Successfully embedded {count} objects.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Backfill KG embeddings for storage objects")
    parser.add_argument("--owner", type=int, default=None, help="Owner user id to backfill")
    parser.add_argument("--limit", type=int, default=None, help="Max objects to process")
    args = parser.parse_args()

    db = next(get_db())
    try:
        asyncio.run(backfill_for_user(db, owner_user_id=args.owner, limit=args.limit))
    finally:
        db.close()


if __name__ == "__main__":
    main()


