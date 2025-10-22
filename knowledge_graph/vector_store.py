"""
Vector Store

ChromaDB integration for storing and querying semantic embeddings.
"""

# Fix for older SQLite versions - use pysqlite3 instead
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
from pathlib import Path
import os

from knowledge_graph.models import EmbeddingVector


class VectorStore:
    """
    ChromaDB-based vector store for semantic embeddings.

    Stores embeddings and enables fast similarity search.
    """

    def __init__(self, persist_directory: Optional[str] = None):
        """
        Initialize ChromaDB client.

        Args:
            persist_directory: Directory to persist ChromaDB data.
                              Defaults to /var/lib/chromadb on server, ./chromadb locally.
        """
        if persist_directory is None:
            # Default location based on environment
            if os.path.exists("/var/lib"):
                persist_directory = "/var/lib/chromadb"
            else:
                persist_directory = "./chromadb"

        persist_path = Path(persist_directory)
        persist_path.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(persist_path),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Collection for O'Neal/general knowledge
        self.collection_name = "arkturian_knowledge"
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Semantic knowledge graph embeddings"}
        )

    def upsert_embedding(self, embedding: EmbeddingVector) -> None:
        """
        Store or update an embedding in ChromaDB.

        If an embedding with the same ID exists, it will be updated.

        Args:
            embedding: EmbeddingVector to store
        """
        chroma_id, vector, metadata, document = embedding.to_chroma_document()

        self.collection.upsert(
            ids=[chroma_id],
            embeddings=[vector],
            metadatas=[metadata],
            documents=[document]
        )

    def get_embeddings_for_object(self, object_id: int) -> List[Dict[str, Any]]:
        """
        Get all embeddings for a storage object ID.

        Supports multiple embeddings per object (e.g., CSV rows).
        Returns metadata including row_hash for differential updates.

        Args:
            object_id: Storage object ID

        Returns:
            List of embedding metadata dicts:
            [
                {
                    "id": "obj_1888_0",
                    "metadata": {"object_id": 1888, "embedding_index": 0, "row_hash": "abc123", ...},
                    "document": "..."
                },
                ...
            ]
        """
        try:
            # Try querying by object_id metadata first (for newer embeddings)
            result = self.collection.get(
                where={"object_id": object_id},
                include=["metadatas", "documents"]
            )

            # If found via metadata, return those
            if len(result["ids"]) > 0:
                embeddings = []
                for i in range(len(result["ids"])):
                    embeddings.append({
                        "id": result["ids"][i],
                        "metadata": result["metadatas"][i],
                        "document": result["documents"][i]
                    })
                return embeddings

            # Fallback: query by ID pattern (for older embeddings without object_id metadata)
            # Get all embeddings and filter by ID pattern obj_{object_id}_*
            all_result = self.collection.get(include=["metadatas", "documents"])

            embeddings = []
            id_prefix = f"obj_{object_id}_"

            for i, emb_id in enumerate(all_result["ids"]):
                if emb_id.startswith(id_prefix):
                    embeddings.append({
                        "id": emb_id,
                        "metadata": all_result["metadatas"][i],
                        "document": all_result["documents"][i]
                    })

            return embeddings
        except Exception as e:
            print(f"Note: Could not load embeddings for object {object_id}: {e}")
            return []

    def delete_embedding(self, object_id: int) -> None:
        """
        Delete all embeddings for a storage object ID.

        Supports multiple embeddings per object (e.g., CSV rows).
        Uses metadata filter to delete all embeddings with matching object_id.

        Args:
            object_id: Storage object ID to delete
        """
        try:
            # Delete all embeddings with this object_id in metadata
            self.collection.delete(where={"object_id": object_id})
            print(f"✅ Deleted all embeddings for object: {object_id}")
        except Exception as e:
            # ChromaDB may raise if no embeddings exist, which is fine
            print(f"Note: Could not delete embeddings for object {object_id}: {e}")

    def find_similar(
        self,
        object_id: int,
        limit: int = 10,
        max_distance: float = 2.0,
        where: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Find objects similar to the given storage object.

        Args:
            object_id: Storage object ID to find similar items for
            limit: Maximum number of results
            max_distance: Maximum distance to include (0 = identical, 2 = completely different)
                         Only results with distance <= max_distance are returned.

        Returns:
            List of similar objects with scores:
            [
                {
                    "object_id": 123,
                    "distance": 0.15,
                    "metadata": {...},
                    "document": "..."
                },
                ...
            ]
        """
        # Try to get embedding - handle both single and multi-embedding objects
        chroma_id = f"obj_{object_id}"
        result = self.collection.get(ids=[chroma_id], include=["embeddings"])

        # If not found, try first embedding of multi-embedding object
        # Check length only to avoid numpy array boolean ambiguity
        if len(result["embeddings"]) == 0:
            chroma_id = f"obj_{object_id}_0"
            result = self.collection.get(ids=[chroma_id], include=["embeddings"])

        if len(result["embeddings"]) == 0:
            raise ValueError(f"No embedding found for object {object_id} (tried 'obj_{object_id}' and 'obj_{object_id}_0')")

        query_embedding = result["embeddings"][0]
        print(f"🔍 find_similar: Using embedding from {chroma_id} for similarity search")

        # Debug logging
        with open("/tmp/vector_search_debug.log", "a") as f:
            f.write(f"\n=== find_similar called for object_id={object_id} ===\n")
            f.write(f"Using embedding ID: {chroma_id}\n")
            f.write(f"Limit: {limit}, max_distance: {max_distance}\n")
            f.write(f"Where filter: {where}\n")

        # Search for similar vectors
        query_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": limit + 1,  # +1 because we'll filter out the query object itself
            "include": ["distances", "metadatas", "documents"],
        }
        # Apply metadata filter if provided
        if where:
            query_kwargs["where"] = where

        similar = self.collection.query(**query_kwargs)

        # Format results
        results = []
        print(f"🔍 find_similar: Found {len(similar['ids'][0])} candidates for object {object_id}")

        with open("/tmp/vector_search_debug.log", "a") as f:
            f.write(f"Found {len(similar['ids'][0])} candidates from ChromaDB\n")

        for i, result_chroma_id in enumerate(similar["ids"][0]):
            distance = similar["distances"][0][i]
            print(f"  Candidate {i}: {result_chroma_id}, distance={distance}")

            with open("/tmp/vector_search_debug.log", "a") as f:
                f.write(f"  Candidate {i}: ID={result_chroma_id}, distance={distance:.4f}\n")

            # Skip only the exact embedding we're querying
            # For multi-embedding objects (CSV rows), we want to return other rows from same parent
            if result_chroma_id == chroma_id:
                print(f"    → Skipping (same embedding as query)")
                with open("/tmp/vector_search_debug.log", "a") as f:
                    f.write(f"    SKIPPED: same embedding as query\n")
                continue

            # Skip if distance is too high (not similar enough)
            if distance > max_distance:
                print(f"    → Skipping (distance {distance} > max {max_distance})")
                with open("/tmp/vector_search_debug.log", "a") as f:
                    f.write(f"    SKIPPED: distance too high ({distance:.4f} > {max_distance})\n")
                continue

            print(f"    → Adding to results")

            # Extract object_id from metadata or chroma_id
            try:
                metadata = similar["metadatas"][0][i]
                with open("/tmp/vector_search_debug.log", "a") as f:
                    f.write(f"    Metadata keys: {list(metadata.keys())}\n")

                # Try to get object_id from metadata
                if "object_id" in metadata:
                    obj_id = metadata["object_id"]
                else:
                    # Extract from chroma_id (obj_1888_0 -> 1888)
                    obj_id = int(result_chroma_id.split("_")[1])
                    with open("/tmp/vector_search_debug.log", "a") as f:
                        f.write(f"    object_id not in metadata, extracted from ID: {obj_id}\n")

                with open("/tmp/vector_search_debug.log", "a") as f:
                    f.write(f"    ADDED to results (object_id={obj_id})\n")

                results.append({
                    "object_id": obj_id,
                    "distance": distance,
                    "metadata": metadata,
                    "document": similar["documents"][0][i]
                })
            except Exception as e:
                with open("/tmp/vector_search_debug.log", "a") as f:
                    f.write(f"    ERROR extracting object_id: {e}\n")
                import traceback
                with open("/tmp/vector_search_debug.log", "a") as f:
                    f.write(traceback.format_exc())
                raise

        print(f"🔍 find_similar: Returning {len(results)} results")
        with open("/tmp/vector_search_debug.log", "a") as f:
            f.write(f"Returning {len(results)} final results\n")
        return results[:limit]  # Return only up to limit

    def search_by_text(
        self,
        query_text: str,
        query_embedding: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for objects by semantic similarity to text query.

        Args:
            query_text: Text query (for logging)
            query_embedding: Pre-computed embedding vector
            limit: Maximum number of results
            filters: Optional metadata filters (e.g., {"category": "product"})

        Returns:
            List of matching objects with scores
        """
        where_filter = filters if filters else None

        similar = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where=where_filter,
            include=["distances", "metadatas", "documents"]  # IDs are always included by default
        )

        results = []
        for i in range(len(similar["ids"][0])):
            metadata = similar["metadatas"][0][i]

            # Extract object_id from metadata or from the ID format (obj_{object_id}_{idx})
            object_id = metadata.get("object_id")
            if object_id is None:
                # Fallback: parse from ChromaDB ID format "obj_{object_id}_{idx}"
                chroma_id = similar["ids"][0][i]
                if chroma_id.startswith("obj_"):
                    parts = chroma_id.split("_")
                    if len(parts) >= 2:
                        try:
                            object_id = int(parts[1])
                        except (ValueError, IndexError):
                            continue

            if object_id is not None:
                results.append({
                    "object_id": object_id,
                    "distance": similar["distances"][0][i],
                    "metadata": metadata,
                    "document": similar["documents"][0][i]
                })

        return results

    def count(self) -> int:
        """Get total number of embeddings in the collection"""
        return self.collection.count()

    def reset(self) -> None:
        """Delete all embeddings (use with caution!)"""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Semantic knowledge graph embeddings"}
        )


# Global vector store instance
vector_store = VectorStore()
