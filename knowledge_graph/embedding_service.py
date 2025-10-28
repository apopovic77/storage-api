"""
Embedding Service

Generates semantic vector embeddings using OpenAI's text-embedding-3-large model.
"""

import os
from typing import List
from openai import AsyncOpenAI
from models import StorageObject


class EmbeddingService:
    """
    Service for generating semantic embeddings.

    Uses OpenAI's text-embedding-3-large model to convert text into
    1536-dimensional vectors for semantic similarity search.
    """

    def __init__(self):
        self.model = "text-embedding-3-large"
        # text-embedding-3-large returns 3072-dimensional vectors by default
        self.dimensions = 3072
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector from text.

        Args:
            text: Text to embed

        Returns:
            List of 1536 floats representing the semantic vector
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        response = await self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return response.data[0].embedding

    def create_embedding_text(self, storage_obj: StorageObject) -> str:
        """
        Combine all semantic information from a StorageObject into embeddable text.

        This combines:
        - AI category
        - AI extracted tags
        - Context text
        - Collection/link IDs

        Args:
            storage_obj: Storage object with AI-analyzed data

        Returns:
            Combined text suitable for embedding
        """
        parts = []

        # Category
        if storage_obj.ai_category:
            parts.append(f"Category: {storage_obj.ai_category}")

        # Extracted tags (from AI analysis)
        if storage_obj.ai_tags:
            if isinstance(storage_obj.ai_tags, list):
                # ai_tags is a list of strings (new format)
                parts.append(f"Tags: {', '.join(str(t) for t in storage_obj.ai_tags)}")
            elif isinstance(storage_obj.ai_tags, dict):
                # ai_tags is a dict (legacy format)
                for key, value in storage_obj.ai_tags.items():
                    if value:
                        if isinstance(value, list):
                            parts.append(f"{key}: {', '.join(str(v) for v in value)}")
                        else:
                            parts.append(f"{key}: {value}")

        # Context text (if available)
        if storage_obj.ai_context_metadata:
            ctx = storage_obj.ai_context_metadata.get("context_text")
            if ctx:
                parts.append(f"Description: {ctx}")

        # Collection and link IDs (provide context about organization)
        if storage_obj.collection_id:
            parts.append(f"Collection: {storage_obj.collection_id}")

        if storage_obj.link_id:
            parts.append(f"Link ID: {storage_obj.link_id}")

        # Title and description (if available)
        if storage_obj.title:
            parts.append(f"Title: {storage_obj.title}")

        if storage_obj.description:
            parts.append(f"Description: {storage_obj.description}")

        return "\n".join(parts)


# Global service instance
embedding_service = EmbeddingService()
