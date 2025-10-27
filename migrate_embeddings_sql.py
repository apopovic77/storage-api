#!/usr/bin/env python3
"""
SQL-Based Migration Script: Per-Tenant Collection Separation

Migrates embeddings by directly querying the old ChromaDB SQLite database
and inserting into new tenant-specific collections.

This bypasses ChromaDB version incompatibility issues.
"""

import sys
import os
import argparse
import sqlite3
import json
from typing import Dict, List
from pathlib import Path

# Fix for older SQLite versions
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
from chromadb.config import Settings


def analyze_old_db(db_path: str):
    """Analyze the old ChromaDB SQLite database."""
    print(f"\n📊 Analyzing old ChromaDB at: {db_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get total count
    cursor.execute("SELECT COUNT(*) FROM embeddings")
    total = cursor.fetchone()[0]
    print(f"   Total embeddings: {total}")

    if total == 0:
        print("   ⚠️  No embeddings to migrate")
        conn.close()
        return None

    # Get tenant distribution
    cursor.execute("""
        SELECT m.string_value as tenant_id, COUNT(*) as count
        FROM embeddings e
        JOIN embedding_metadata m ON e.id = m.id
        WHERE m.key = 'tenant_id'
        GROUP BY m.string_value
    """)

    tenant_dist = {}
    for row in cursor.fetchall():
        tenant_dist[row[0]] = row[1]

    print(f"\n   Tenant Distribution:")
    for tenant, count in tenant_dist.items():
        print(f"     {tenant}: {count} embeddings")

    # Get brand distribution
    cursor.execute("""
        SELECT m.string_value as brand, COUNT(*) as count
        FROM embeddings e
        JOIN embedding_metadata m ON e.id = m.id
        WHERE m.key = 'brand'
        GROUP BY m.string_value
        ORDER BY count DESC
    """)

    print(f"\n   Brand Distribution:")
    for row in cursor.fetchall():
        print(f"     {row[0]}: {row[1]} embeddings")

    conn.close()
    return {'total': total, 'tenant_dist': tenant_dist}


def load_embeddings_from_sql(db_path: str, tenant_id: str):
    """Load embeddings for a specific tenant from SQLite."""
    print(f"\n📥 Loading embeddings for tenant '{tenant_id}' from SQL...")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all embeddings for this tenant
    cursor.execute("""
        SELECT DISTINCT e.id, e.embedding_id, e.embedding
        FROM embeddings e
        JOIN embedding_metadata m ON e.id = m.id
        WHERE m.key = 'tenant_id' AND m.string_value = ?
    """, (tenant_id,))

    embeddings_data = []
    for row in cursor.fetchall():
        emb_internal_id, emb_id, emb_vector_blob = row

        # Deserialize embedding vector (stored as blob)
        import struct
        import array
        if emb_vector_blob:
            # ChromaDB stores vectors as float32 array
            vector = list(struct.unpack(f'{len(emb_vector_blob)//4}f', emb_vector_blob))
        else:
            print(f"     ⚠️  Warning: Embedding {emb_id} has no vector data")
            continue

        # Get all metadata for this embedding
        cursor.execute("""
            SELECT key, string_value, int_value, float_value, bool_value
            FROM embedding_metadata
            WHERE id = ?
        """, (emb_internal_id,))

        metadata = {}
        for meta_row in cursor.fetchall():
            key, str_val, int_val, float_val, bool_val = meta_row

            # Use the appropriate value based on what's not None
            if str_val is not None:
                metadata[key] = str_val
            elif int_val is not None:
                metadata[key] = int_val
            elif float_val is not None:
                metadata[key] = float_val
            elif bool_val is not None:
                metadata[key] = bool(bool_val)

        # Get document text
        cursor.execute("""
            SELECT document
            FROM embeddings
            WHERE id = ?
        """, (emb_internal_id,))
        doc_row = cursor.fetchone()
        document = doc_row[0] if doc_row and doc_row[0] else ""

        embeddings_data.append({
            'id': emb_id,
            'vector': vector,
            'metadata': metadata,
            'document': document
        })

    conn.close()
    print(f"   ✅ Loaded {len(embeddings_data)} embeddings")
    return embeddings_data


def migrate_to_tenant_collection(embeddings_data: List[Dict], dest_client, new_tenant_id: str, dry_run: bool = False):
    """Migrate embeddings to tenant-specific collection."""
    if not embeddings_data:
        print(f"   ⚠️  No embeddings to migrate")
        return 0

    print(f"\n🚀 Migrating {len(embeddings_data)} embeddings to tenant '{new_tenant_id}'")

    if dry_run:
        print(f"   [DRY RUN] Would create collection: tenant_{new_tenant_id}_knowledge")
        print(f"   [DRY RUN] Would migrate {len(embeddings_data)} embeddings")
        return len(embeddings_data)

    # Create tenant collection
    collection_name = f"tenant_{new_tenant_id}_knowledge"
    collection = dest_client.get_or_create_collection(
        name=collection_name,
        metadata={
            "description": f"Semantic knowledge graph embeddings for tenant: {new_tenant_id}",
            "tenant_id": new_tenant_id,
            "migrated_from": "arkturian_knowledge"
        }
    )

    # Prepare batch data
    ids = []
    embeddings = []
    metadatas = []
    documents = []

    for item in embeddings_data:
        ids.append(item['id'])
        embeddings.append(item['vector'])

        # Update metadata with correct tenant_id
        metadata = item['metadata'].copy()
        metadata['tenant_id'] = new_tenant_id
        metadatas.append(metadata)

        documents.append(item['document'])

    # Batch upsert in chunks (ChromaDB has batch size limits)
    BATCH_SIZE = 100
    for i in range(0, len(ids), BATCH_SIZE):
        batch_ids = ids[i:i+BATCH_SIZE]
        batch_embeddings = embeddings[i:i+BATCH_SIZE]
        batch_metadatas = metadatas[i:i+BATCH_SIZE]
        batch_documents = documents[i:i+BATCH_SIZE]

        collection.upsert(
            ids=batch_ids,
            embeddings=batch_embeddings,
            metadatas=batch_metadatas,
            documents=batch_documents
        )
        print(f"   ✅ Migrated batch {i//BATCH_SIZE + 1} ({len(batch_ids)} embeddings)")

    # Verify
    final_count = collection.count()
    print(f"   ✅ Total embeddings in {collection_name}: {final_count}")

    return final_count


def main():
    parser = argparse.ArgumentParser(description="SQL-based ChromaDB migration to per-tenant collections")
    parser.add_argument(
        '--source-db',
        default='/var/lib/chromadb/chroma.sqlite3',
        help='Source ChromaDB SQLite database path'
    )
    parser.add_argument(
        '--dest',
        default='/var/www/api-storage.arkturian.com/chroma_db',
        help='Destination ChromaDB directory'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be migrated without actually migrating'
    )
    parser.add_argument(
        '--old-tenant',
        default='arkturian',
        help='Old tenant_id to migrate from'
    )
    parser.add_argument(
        '--new-tenant',
        default='oneal',
        help='New tenant_id to migrate to'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("🔄 SQL-Based ChromaDB Tenant Migration")
    print("=" * 80)
    print(f"Source DB:   {args.source_db}")
    print(f"Destination: {args.dest}")
    print(f"Mode:        {'DRY RUN' if args.dry_run else 'LIVE MIGRATION'}")
    print(f"Mapping:     {args.old_tenant} → {args.new_tenant}")
    print("=" * 80)

    # Analyze old database
    stats = analyze_old_db(args.source_db)
    if not stats:
        return 1

    # Load embeddings from SQL
    embeddings_data = load_embeddings_from_sql(args.source_db, args.old_tenant)
    if not embeddings_data:
        print(f"\n❌ No embeddings found for tenant '{args.old_tenant}'")
        return 1

    # Connect to destination
    print(f"\n📦 Connecting to destination ChromaDB at: {args.dest}")
    dest_client = chromadb.PersistentClient(
        path=args.dest,
        settings=Settings(
            anonymized_telemetry=False,
            allow_reset=False
        )
    )

    # Migrate
    migrated_count = migrate_to_tenant_collection(
        embeddings_data,
        dest_client,
        args.new_tenant,
        dry_run=args.dry_run
    )

    # Summary
    print("\n" + "=" * 80)
    print("📊 Migration Summary")
    print("=" * 80)
    print(f"Total embeddings in source: {stats['total']}")
    print(f"Embeddings {'to be migrated' if args.dry_run else 'migrated'}: {migrated_count}")

    if not args.dry_run:
        print("\n✅ Migration completed successfully!")
        print(f"\n⚠️  Old database at {args.source_db} is still intact")
        print(f"   You can safely delete /var/lib/chromadb after verifying the migration.")
    else:
        print("\n💡 This was a dry run. Re-run without --dry-run to perform actual migration.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
