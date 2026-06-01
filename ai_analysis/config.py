"""
AI Analysis Configuration

Supports two modes:
1. UNIFIED: Single AI request for safety + embedding analysis (faster, cheaper)
2. SPLIT: Separate AI requests for safety and embedding (flexible model selection)

Environment Variables:
- AI_ANALYSIS_MODE: "unified" (default) or "split"
- SAFETY_MODEL: Model for safety checks (e.g., "gemini-flash", "gpt-4")
- EMBEDDING_MODEL: Model for embedding generation (e.g., "gemini-pro", "gpt-3.5-turbo")
"""

import os
from enum import Enum
from dotenv import load_dotenv
from pathlib import Path

# Load .env file from parent directory
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

class AIAnalysisMode(str, Enum):
    UNIFIED = "unified"  # Single request for everything (default)
    SPLIT = "split"      # Separate requests for safety and embedding

class AIConfig:
    """Configuration for AI analysis behavior"""

    def __init__(self):
        # Analysis mode
        self.mode = AIAnalysisMode(os.getenv("AI_ANALYSIS_MODE", "unified"))

        # Model selection.
        # NOTE: these strings are descriptive labels only (used for logging /
        # metadata, see _analyze_split_mode); the actual call routing is decided
        # by SAFETY_BACKEND / ANALYSIS_BACKEND in ai_analysis.service. The default
        # must NOT name a Google model — per project policy Gemini may only run
        # via CLI subprocess, never via the billable API/SDK. Defaulting to a
        # "gemini-*" label here was a latent footgun (it implied a Google path
        # even though the .env overrode it). Align the fallback with the chatgpt
        # backend default established in fd3eee3.
        self.safety_model = os.getenv("SAFETY_MODEL", "chatgpt")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "chatgpt")
        self.unified_model = os.getenv("UNIFIED_MODEL", "chatgpt")

    def is_unified(self) -> bool:
        """Check if running in unified mode"""
        return self.mode == AIAnalysisMode.UNIFIED

    def is_split(self) -> bool:
        """Check if running in split mode"""
        return self.mode == AIAnalysisMode.SPLIT

# Global configuration instance
ai_config = AIConfig()
