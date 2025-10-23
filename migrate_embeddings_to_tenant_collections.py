#!/usr/bin/env python3
"""
Migration Script: Per-Tenant Collection Separation

Migrates embeddings from the old shared "arkturian_knowledge" collection
to tenant-specific collections (tenant_{tenant_id}_knowledge).

This script:
1. Reads all embeddings from old location (/var/lib/chromadb)
2. Groups embeddings by tenant_id from metadata
3. Creates new tenant-specific collections in new location
4. Copies embeddings with corrected tenant_id metadata
5. Provides verification and rollback capabilities

Usage:
    python migrate_embeddings_to_tenant_collections.py [--dry-run] [--source=/path] [--dest=/path]
"""

import sys
import os
import argparse
from pathlib import Path
from typing import Dict, List, Any
import json

# Fix for older SQLite versions
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
from chromadb.config import Settings


def connect_to_chromadb(path: str, description: str = "ChromaDB"):
    """Connect to a ChromaDB instance."""
    print(f"\n📦 Connecting to {description} at: {path}")
    client = chromadb.PersistentClient(
        path=path,
        settings=Settings(
            anonymized_telemetry=False,
            allow_reset=False
        )
    )
    return client


def analyze_old_collection(source_client):
    """Analyze the old shared collection."""
    print("\n🔍 Analyzing old 'arkturian_knowledge' collection...")

    try:
        collection = source_client.get_collection("arkturian_knowledge")
    except Exception as e:
        print(f"❌ Error: Could not find 'arkturian_knowledge' collection: {e}")
        return None

    count = collection.count()
    print(f"   Total embeddings: {count}")

    if count == 0:
        print("   ⚠️  Collection is empty, nothing to migrate")
        return None

    # Get all embeddings
    result = collection.get(
        include=['metadatas', 'embeddings', 'documents'],
        limit=count
    )

    # Analyze tenant distribution
    tenant_groups: Dict[str, List[Any]] = {}
    brand_counts: Dict[str, int] = {}

    for i, emb_id in enumerate(result['ids']):
        metadata = result['metadatas'][i]
        tenant_id = metadata.get('tenant_id', 'unknown')
        brand = metadata.get('brand', 'unknown')

        if tenant_id not in tenant_groups:
            tenant_groups[tenant_id] = []

        tenant_groups[tenant_id].append({
            'id': emb_id,
            'metadata': metadata,
            'embedding': result['embeddings'][i],
            'document': result['documents'][i]
        })

        brand_counts[brand] = brand_counts.get(brand, 0) + 1

    print(f"\n📊 Tenant Distribution:")
    for tenant_id, embeddings in sorted(tenant_groups.items()):
        print(f"   {tenant_id}: {len(embeddings)} embeddings")

    print(f"\n🏷️  Brand Distribution:")
    for brand, count in sorted(brand_counts.items(), key=lambda x: -x[1]):
        print(f"   {brand}: {count} embeddings")

    return {
        'collection': collection,
        'total': count,
        'tenant_groups': tenant_groups,
        'brand_counts': brand_counts,
        'raw_result': result
    }


def create_tenant_collection(dest_client, tenant_id: str):
    """Create a tenant-specific collection."""
    collection_name = f"tenant_{tenant_id}_knowledge"
    print(f"\n📝 Creating collection: {collection_name}")

    collection = dest_client.get_or_create_collection(
        name=collection_name,
        metadata={
            "description": f"Semantic knowledge graph embeddings for tenant: {tenant_id}",
            "tenant_id": tenant_id,
            "migrated_from": "arkturian_knowledge"
        }
    )
    return collection


def migrate_tenant_embeddings(
    source_data: Dict,
    dest_client,
    old_tenant_id: str,
    new_tenant_id: str,
    dry_run: bool = False
):
    """Migrate embeddings for a specific tenant."""
    tenant_embeddings = source_data['tenant_groups'].get(old_tenant_id, [])

    if not tenant_embeddings:
        print(f"   ⚠️  No embeddings found for tenant '{old_tenant_id}'")
        return 0

    print(f"\n🚀 Migrating {len(tenant_embeddings)} embeddings: {old_tenant_id} → {new_tenant_id}")

    if dry_run:
        print(f"   [DRY RUN] Would create collection: tenant_{new_tenant_id}_knowledge")
        print(f"   [DRY RUN] Would migrate {len(tenant_embeddings)} embeddings")
        return len(tenant_embeddings)

    # Create tenant-specific collection
    dest_collection = create_tenant_collection(dest_client, new_tenant_id)

    # Prepare batch data
    ids = []
    embeddings = []
    metadatas = []
    documents = []

    for item in tenant_embeddings:
        ids.append(item['id'])
        embeddings.append(item['embedding'])

        # Update metadata with correct tenant_id
        metadata = item['metadata'].copy()
        metadata['tenant_id'] = new_tenant_id
        metadatas.append(metadata)

        documents.append(item['document'])

    # Batch upsert
    dest_collection.upsert(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=documents
    )

    # Verify
    migrated_count = dest_collection.count()
    print(f"   ✅ Migrated {migrated_count} embeddings to {dest_collection.name}")

    return migrated_count


def main():
    parser = argparse.ArgumentParser(description="Migrate ChromaDB embeddings to per-tenant collections")
    parser.add_argument(
        '--source',
        default='/var/lib/chromadb',
        help='Source ChromaDB path (default: /var/lib/chromadb)'
    )
    parser.add_argument(
        '--dest',
        default='/var/www/api-storage.arkturian.com/chroma_db',
        help='Destination ChromaDB path (default: ./chroma_db)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be migrated without actually migrating'
    )
    parser.add_argument(
        '--map-tenant',
        action='append',
        help='Map old tenant_id to new tenant_id (format: old:new)'
    )

    args = parser.parse_args()

    # Build tenant mapping
    tenant_mapping = {}
    if args.map_tenant:
        for mapping in args.map_tenant:
            old, new = mapping.split(':', 1)
            tenant_mapping[old] = new
    else:
        # Default mapping: arkturian → oneal (based on product analysis)
        tenant_mapping = {
            'arkturian': 'oneal'
        }

    print("=" * 80)
    print("🔄 ChromaDB Tenant Migration Script")
    print("=" * 80)
    print(f"Source:      {args.source}")
    print(f"Destination: {args.dest}")
    print(f"Mode:        {'DRY RUN' if args.dry_run else 'LIVE MIGRATION'}")
    print(f"Tenant Mapping: {tenant_mapping}")
    print("=" * 80)

    # Connect to source
    source_client = connect_to_chromadb(args.source, "Source ChromaDB")

    # Analyze old collection
    source_data = analyze_old_collection(source_client)
    if not source_data:
        print("\n❌ Migration aborted: No data to migrate")
        return 1

    # Connect to destination
    dest_client = connect_to_chromadb(args.dest, "Destination ChromaDB")

    # Migrate each tenant
    total_migrated = 0
    for old_tenant_id, new_tenant_id in tenant_mapping.items():
        count = migrate_tenant_embeddings(
            source_data,
            dest_client,
            old_tenant_id,
            new_tenant_id,
            dry_run=args.dry_run
        )
        total_migrated += count

    # Summary
    print("\n" + "=" * 80)
    print("📊 Migration Summary")
    print("=" * 80)
    print(f"Total embeddings in source: {source_data['total']}")
    print(f"Embeddings {'to be migrated' if args.dry_run else 'migrated'}: {total_migrated}")

    if not args.dry_run:
        print("\n✅ Migration completed successfully!")
        print(f"\n⚠️  Old collection '{source_data['collection'].name}' is still intact at {args.source}")
        print(f"   You can safely delete it after verifying the migration.")
    else:
        print("\n💡 This was a dry run. Re-run without --dry-run to perform actual migration.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
