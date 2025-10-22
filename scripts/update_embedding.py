#!/usr/bin/env python3
"""
Update embedding text for a storage object and re-embed into Knowledge Graph.

Usage:
    python3 update_embedding.py <object_id>                    # View current embedding text
    python3 update_embedding.py <object_id> --text "New text"  # Update embedding text
"""

import sys
import argparse
import asyncio
from pathlib import Path

sys.path.insert(0, "/var/www/storage-api")

from database import SessionLocal
from models import StorageObject
from knowledge_graph.embedding_service import embedding_service
from knowledge_graph.vector_store import vector_store
from knowledge_graph.models import EmbeddingVector


def get_current_embedding_text(obj: StorageObject) -> str:
    """Extract current embedding text from object."""
    metadata = obj.ai_context_metadata or {}
    embedding_info = metadata.get("embedding_info", {})
    return embedding_info.get("embeddingText", "")


def update_embedding_text(obj: StorageObject, new_text: str) -> None:
    """Update the embedding text in ai_context_metadata."""
    if obj.ai_context_metadata is None:
        obj.ai_context_metadata = {}
    
    if "embedding_info" not in obj.ai_context_metadata:
        obj.ai_context_metadata["embedding_info"] = {}
    
    obj.ai_context_metadata["embedding_info"]["embeddingText"] = new_text
    # Mark as modified for SQLAlchemy JSONB tracking
    from sqlalchemy.orm.attributes import flag_modified
    flag_modified(obj, "ai_context_metadata")


async def regenerate_embedding(obj: StorageObject) -> None:
    """Regenerate and update the embedding in Chroma Vector DB."""
    embedding_text = get_current_embedding_text(obj)
    
    if not embedding_text:
        print("❌ No embedding text found - cannot generate embedding")
        return
    
    print(f"🤖 Generating new embedding vector from text...")
    print(f"   Text: {embedding_text[:100]}..." if len(embedding_text) > 100 else f"   Text: {embedding_text}")
    
    # Generate embedding vector
    vector = await embedding_service.generate_embedding(embedding_text)
    
    print(f"✅ Generated {len(vector)}-dimensional vector")
    
    # Create EmbeddingVector model
    embedding = EmbeddingVector(
        storage_object_id=obj.id,
        vector=vector,
        embedding_text=embedding_text,
        metadata={
            "object_id": obj.id,
            "tenant_id": getattr(obj, "tenant_id", "arkturian"),
            "title": obj.title,
            "ai_category": obj.ai_category,
            "ai_tags": obj.ai_tags or []
        }
    )
    
    # Upsert into Chroma
    print(f"💾 Updating Knowledge Graph (Chroma Vector DB)...")
    vector_store.upsert_embedding(embedding)
    
    print(f"✅ Embedding updated in Knowledge Graph!")


async def main():
    parser = argparse.ArgumentParser(description="Update embedding text and re-embed")
    parser.add_argument("object_id", type=int, help="Storage object ID")
    parser.add_argument("--text", type=str, help="New embedding text (if updating)")
    parser.add_argument("--regenerate", action="store_true", help="Regenerate embedding from current text")
    
    args = parser.parse_args()
    
    db = SessionLocal()
    
    try:
        # Get object
        obj = db.query(StorageObject).filter(StorageObject.id == args.object_id).first()
        
        if not obj:
            print(f"❌ Object {args.object_id} not found")
            sys.exit(1)
        
        print(f"📦 Object: {obj.id} - {obj.title}")
        print(f"   Filename: {obj.original_filename}")
        print(f"   Tenant: {getattr(obj, 'tenant_id', 'arkturian')}")
        print()
        
        # Get current embedding text
        current_text = get_current_embedding_text(obj)
        
        if args.text:
            # Update embedding text
            print("=== CURRENT EMBEDDING TEXT ===")
            print(current_text if current_text else "(empty)")
            print()
            print("=== NEW EMBEDDING TEXT ===")
            print(args.text)
            print()
            
            # Update in database
            update_embedding_text(obj, args.text)
            db.commit()
            print("✅ Updated embedding text in database")
            print()
            
            # Regenerate embedding
            await regenerate_embedding(obj)
            
        elif args.regenerate:
            # Just regenerate from current text
            print("=== CURRENT EMBEDDING TEXT ===")
            print(current_text if current_text else "(empty)")
            print()
            
            await regenerate_embedding(obj)
            
        else:
            # Just show current embedding text
            print("=== CURRENT EMBEDDING TEXT ===")
            print(current_text if current_text else "(empty)")
            print()
            print("To update:")
            print(f'  python3 update_embedding.py {args.object_id} --text "Your new embedding text here"')
            print()
            print("To regenerate embedding from current text:")
            print(f'  python3 update_embedding.py {args.object_id} --regenerate')
    
    finally:
        db.close()


if __name__ == "__main__":
    asyncio.run(main())
