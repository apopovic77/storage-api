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
        - AI-generated embedding text (if available - PRIORITY)
        - Visual and semantic properties for style/vibe-based search
        - AI category, tags, title, description
        - Context text and organizational metadata

        Args:
            storage_obj: Storage object with AI-analyzed data

        Returns:
            Combined text suitable for semantic embedding
        """

        # PRIORITY: Use AI-generated embedding text if available
        if storage_obj.ai_context_metadata:
            embedding_info = storage_obj.ai_context_metadata.get("embedding_info", {})
            ai_embedding_text = embedding_info.get("embeddingText")

            # If AI already generated rich embedding text, use it!
            if ai_embedding_text and len(ai_embedding_text.strip()) > 100:
                return ai_embedding_text

        # FALLBACK: Build rich embedding text from available metadata
        parts = []

        # Title and description first (most important)
        if storage_obj.ai_title or storage_obj.title:
            title = storage_obj.ai_title or storage_obj.title
            parts.append(f"{title}")

        if storage_obj.ai_subtitle or storage_obj.description:
            subtitle = storage_obj.ai_subtitle or storage_obj.description
            parts.append(f"{subtitle}")

        # Extract rich semantic properties from AI metadata
        if storage_obj.ai_context_metadata:
            metadata = storage_obj.ai_context_metadata

            # Product Analysis (style, mood, target audience)
            product_analysis = metadata.get("product_analysis", {})
            if product_analysis:
                style = product_analysis.get("style")
                if style:
                    parts.append(f"Style: {style}")

                target = product_analysis.get("targetAudience", {})
                if isinstance(target, dict):
                    audience_parts = []
                    if target.get("demographics"):
                        audience_parts.append(target["demographics"])
                    if target.get("interests"):
                        audience_parts.append(target["interests"])
                    if audience_parts:
                        parts.append(f"For: {' - '.join(audience_parts)}")

            # Visual Analysis (mood, aesthetic style)
            visual_analysis = metadata.get("visual_analysis", {})
            if visual_analysis:
                aesthetics = visual_analysis.get("aesthetics", {})
                if aesthetics:
                    mood = aesthetics.get("mood")
                    aesthetic_style = aesthetics.get("aestheticStyle")
                    if mood:
                        parts.append(f"Mood: {mood}")
                    if aesthetic_style:
                        parts.append(f"Aesthetic: {aesthetic_style}")

            # Layout Intelligence (visual harmony tags for style matching)
            layout = metadata.get("layout_intelligence", {})
            if layout:
                harmony_tags = layout.get("visualHarmonyTags", [])
                if harmony_tags:
                    # Convert to readable text
                    readable_tags = [tag.replace("_", " ") for tag in harmony_tags[:3]]
                    parts.append(f"Visual theme: {', '.join(readable_tags)}")

            # Semantic Properties (keywords, emotional appeal)
            semantic = metadata.get("semantic_properties", {})
            if semantic:
                emotional = semantic.get("emotionalAppeal", [])
                if emotional and isinstance(emotional, list):
                    parts.append(f"Emotional appeal: {', '.join(emotional[:5])}")

                brand_perception = semantic.get("brandPerception")
                if brand_perception:
                    parts.append(f"Brand feel: {brand_perception}")

        # Category
        if storage_obj.ai_category:
            parts.append(f"Category: {storage_obj.ai_category}")

        # AI Tags (semantic keywords)
        if storage_obj.ai_tags and isinstance(storage_obj.ai_tags, list):
            tags = [str(t) for t in storage_obj.ai_tags[:10]]  # Limit to top 10
            parts.append(f"Keywords: {', '.join(tags)}")

        # Collections (product lines, themes)
        if storage_obj.ai_collections and isinstance(storage_obj.ai_collections, list):
            parts.append(f"Collections: {', '.join(storage_obj.ai_collections[:3])}")

        return ". ".join(parts) if parts else ""


# Global service instance
embedding_service = EmbeddingService()
