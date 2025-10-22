"""
Knowledge Graph Domain Models

Defines the core entities and relationships in the knowledge graph.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum


class EntityType(str, Enum):
    """Types of entities in the knowledge graph"""
    PRODUCT = "product"
    FEATURE = "feature"
    CERTIFICATION = "certification"
    SERIES = "series"
    ACTIVITY = "activity"
    PERSON = "person"
    EVENT = "event"
    LOCATION = "location"
    DOCUMENT = "document"
    MEDIA = "media"
    OTHER = "other"


class RelationType(str, Enum):
    """Types of relationships between entities"""
    HAS_FEATURE = "has_feature"
    BELONGS_TO_SERIES = "belongs_to_series"
    SIMILAR_TO = "similar_to"
    CERTIFIED_BY = "certified_by"
    USED_IN = "used_in"
    LOCATED_AT = "located_at"
    CREATED_BY = "created_by"
    RELATED_TO = "related_to"


@dataclass
class Entity:
    """
    Represents a semantic entity extracted from content.

    Entities are the nodes in the knowledge graph.
    Each entity has a unique ID, type, and properties.
    """
    id: str  # Unique identifier (e.g., "product:airframe-mx-123")
    type: EntityType
    name: str
    properties: Dict[str, Any]  # Flexible schema for entity-specific data
    source_object_id: int  # StorageObject ID that this entity was extracted from
    confidence: float = 1.0  # AI confidence in extraction (0.0 to 1.0)
    created_at: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary for storage"""
        return {
            "id": self.id,
            "type": self.type.value,
            "name": self.name,
            "properties": self.properties,
            "source_object_id": self.source_object_id,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


@dataclass
class Relation:
    """
    Represents a relationship between two entities.

    Relations are the edges in the knowledge graph.
    """
    from_entity_id: str
    to_entity_id: str
    relation_type: RelationType
    properties: Dict[str, Any] = None  # Additional relationship metadata
    confidence: float = 1.0
    created_at: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.properties is None:
            self.properties = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert relation to dictionary for storage"""
        return {
            "from": self.from_entity_id,
            "to": self.to_entity_id,
            "type": self.relation_type.value,
            "properties": self.properties,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


@dataclass
class EmbeddingVector:
    """
    Represents a semantic vector embedding for an entity or storage object.

    Embeddings enable semantic similarity search.
    """
    object_id: int  # StorageObject ID
    vector: List[float]  # 1536-dimensional embedding (OpenAI text-embedding-3-large)
    embedding_text: str  # The text that was embedded
    metadata: Dict[str, Any]  # Category, tags, etc.
    model: str = "text-embedding-3-large"
    created_at: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

    @property
    def chroma_id(self) -> str:
        """Generate ChromaDB-compatible ID"""
        return f"obj_{self.object_id}"

    def to_chroma_document(self) -> tuple:
        """
        Convert to ChromaDB document format.

        Returns:
            tuple: (id, embedding, metadata, document)
        """
        metadata = {
            "object_id": self.object_id,
            "model": self.model,
            **self.metadata
        }
        return (
            self.chroma_id,
            self.vector,
            metadata,
            self.embedding_text
        )


@dataclass
class KnowledgeGraphEntry:
    """
    Complete knowledge graph entry for a storage object.

    Combines entities, relations, and embedding for a single storage object.
    """
    storage_object_id: int
    entities: List[Entity]
    relations: List[Relation]
    embedding: Optional[EmbeddingVector] = None
    created_at: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire graph entry to dictionary"""
        return {
            "storage_object_id": self.storage_object_id,
            "entities": [e.to_dict() for e in self.entities],
            "relations": [r.to_dict() for r in self.relations],
            "embedding": {
                "chroma_id": self.embedding.chroma_id,
                "model": self.embedding.model,
                "metadata": self.embedding.metadata
            } if self.embedding else None,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }
