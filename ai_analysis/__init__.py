"""
AI Analysis Module

Centralized AI analysis for content safety, classification, and embedding generation.

Supports two modes:
- UNIFIED: Single AI request for all analysis (fast, cheap)
- SPLIT: Separate requests for safety and embedding (flexible, powerful)

Usage:
    from ai_analysis.service import analyze_content

    result = await analyze_content(
        data=file_bytes,
        mime_type="image/jpeg",
        context={"brand": "O'Neal", "year": 2026}
    )
"""

from ai_analysis.config import ai_config, AIAnalysisMode
from ai_analysis.service import analyze_content

__all__ = ['analyze_content', 'ai_config', 'AIAnalysisMode']
