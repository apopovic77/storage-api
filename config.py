import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional

# Load .env robustly (try project root and CWD)
_ROOT = Path(__file__).resolve().parent  # /var/www/storage-api -> parent is project root
_candidates = [
    _ROOT / ".env",
    Path.cwd() / ".env",
]
for p in _candidates:
    try:
        if p.exists():
            load_dotenv(p, override=False)
    except Exception:
        pass

class Settings:
    # Database
    DATABASE_URL: str = os.getenv("STORAGE_DATABASE_URL", os.getenv("DATABASE_URL", "sqlite:///./storage.db"))

    # File Storage
    STORAGE_UPLOAD_DIR: str = os.getenv("STORAGE_UPLOAD_DIR", "./uploads/storage")
    BASE_URL: str = os.getenv("STORAGE_BASE_URL", os.getenv("BASE_URL", "https://api-storage.arkturian.com"))

    # ChromaDB / Knowledge Graph
    CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", "./chroma_db")

    # API Keys
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    API_KEY: str = os.getenv("API_KEY", "Inetpass1")

    # Media Settings
    MAX_FILE_SIZE: int = int(os.getenv("STORAGE_MAX_FILE_SIZE", os.getenv("MAX_FILE_SIZE", "524288000")))  # 500MB default
    ALLOWED_IMAGE_TYPES: list = ["image/jpeg", "image/png", "image/webp"]
    ALLOWED_VIDEO_TYPES: list = ["video/mp4", "video/quicktime", "video/x-msvideo", "video/webm"]
    ALLOWED_AUDIO_TYPES: list = ["audio/mpeg", "audio/wav", "audio/aac", "audio/mp4", "audio/webm"]

    # Analysis Settings
    ANALYSIS_TIMEOUT: int = int(os.getenv("STORAGE_ANALYSIS_TIMEOUT", "60"))  # seconds

settings = Settings()
