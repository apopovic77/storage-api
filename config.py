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

_DATA_DIR_ENV = os.getenv("STORAGE_DATA_DIR")
if _DATA_DIR_ENV:
    _DATA_DIR = Path(_DATA_DIR_ENV).expanduser()
else:
    _DATA_DIR = Path("/var/lib/storage-api")

_DATABASE_URL_ENV = os.getenv("STORAGE_DATABASE_URL") or os.getenv("DATABASE_URL")
if not _DATABASE_URL_ENV:
    try:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    default_sqlite_path = _DATA_DIR / "storage.db"
    _DATABASE_URL_ENV = f"sqlite:///{default_sqlite_path}"

_CHROMA_DB_PATH_ENV = os.getenv("CHROMA_DB_PATH")
if not _CHROMA_DB_PATH_ENV:
    default_chroma_dir = _DATA_DIR / "chroma_db"
    try:
        default_chroma_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    _CHROMA_DB_PATH_ENV = str(default_chroma_dir)

class Settings:
    # Database
    DATABASE_URL: str = _DATABASE_URL_ENV

    # File Storage
    STORAGE_UPLOAD_DIR: str = os.getenv("STORAGE_UPLOAD_DIR", "./uploads/storage")
    BASE_URL: str = os.getenv("STORAGE_BASE_URL", os.getenv("BASE_URL", "https://api-storage.arkturian.com"))

    # ChromaDB / Knowledge Graph
    CHROMA_DB_PATH: str = _CHROMA_DB_PATH_ENV

    DATA_DIR: str = str(_DATA_DIR)

    # API Keys
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    API_KEY: str = os.getenv("API_KEY", "Inetpass1")
    
    # User Settings
    NEW_USER_TRUST_LEVEL: str = os.getenv("NEW_USER_TRUST_LEVEL", "user")

    # Media Settings
    MAX_FILE_SIZE: int = int(os.getenv("STORAGE_MAX_FILE_SIZE", os.getenv("MAX_FILE_SIZE", "524288000")))  # 500MB default
    ALLOWED_IMAGE_TYPES: list = ["image/jpeg", "image/png", "image/webp"]
    ALLOWED_VIDEO_TYPES: list = ["video/mp4", "video/quicktime", "video/x-msvideo", "video/webm"]
    ALLOWED_AUDIO_TYPES: list = ["audio/mpeg", "audio/wav", "audio/aac", "audio/mp4", "audio/webm"]

    # Analysis Settings
    ANALYSIS_TIMEOUT: int = int(os.getenv("STORAGE_ANALYSIS_TIMEOUT", "60"))  # seconds

settings = Settings()
