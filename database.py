from sqlalchemy import create_engine, MetaData, text, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from databases import Database
import os
from config import settings

# Import Base from models (will be defined there)
# We'll import this after models.py is created
Base = None  # Will be set after models import

# Database setup
_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "20"))
_MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "40"))
_POOL_TIMEOUT = int(os.getenv("DB_POOL_TIMEOUT", "10"))
_POOL_RECYCLE = int(os.getenv("DB_POOL_RECYCLE", "1800"))
_ECHO_POOL = os.getenv("DB_ECHO_POOL", "false").lower() == "true"

engine = create_engine(
    settings.DATABASE_URL,
    pool_size=_POOL_SIZE,
    max_overflow=_MAX_OVERFLOW,
    pool_timeout=_POOL_TIMEOUT,
    pool_recycle=_POOL_RECYCLE,
    pool_pre_ping=True,
    echo_pool='debug' if _ECHO_POOL else False,
    connect_args={"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}
)

# Enable WAL and reasonable SQLite pragmas to improve concurrent access
if settings.DATABASE_URL.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        try:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL;")
            cursor.execute("PRAGMA synchronous=NORMAL;")
            cursor.execute("PRAGMA busy_timeout=5000;")
            cursor.execute("PRAGMA foreign_keys=ON;")
            cursor.close()
        except Exception:
            # Best-effort; do not crash if pragmas fail
            pass

# Create async database instance
_ASYNC_MIN_SIZE = int(os.getenv("DB_ASYNC_MIN_SIZE", "1"))
_ASYNC_MAX_SIZE = int(os.getenv("DB_ASYNC_MAX_SIZE", "10"))
database = Database(settings.DATABASE_URL, min_size=_ASYNC_MIN_SIZE, max_size=_ASYNC_MAX_SIZE)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def create_tables():
    """Create all database tables"""
    from models import Base  # Import here to avoid circular dependency
    Base.metadata.create_all(bind=engine)

def get_db():
    """Database dependency for FastAPI"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def connect_db():
    """Connect to database (for startup)"""
    await database.connect()
    create_tables()

    # Lightweight, idempotent migrations for SQLite deployments
    try:
        if settings.DATABASE_URL.startswith("sqlite"):
            with engine.connect() as conn:
                # Ensure storage_objects has AI analysis columns
                scols = {row[1] for row in conn.execute(text("PRAGMA table_info(storage_objects)"))}

                if "ai_category" not in scols:
                    conn.execute(text("ALTER TABLE storage_objects ADD COLUMN ai_category VARCHAR"))
                if "ai_danger_potential" not in scols:
                    conn.execute(text("ALTER TABLE storage_objects ADD COLUMN ai_danger_potential INTEGER"))

                # Ensure storage_objects has link_id column for linking related files
                if "link_id" not in scols:
                    conn.execute(text("ALTER TABLE storage_objects ADD COLUMN link_id VARCHAR"))
                    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_storage_link_id ON storage_objects (link_id)"))

                # Ensure storage_objects has webview_url column for web-optimized images
                if "webview_url" not in scols:
                    conn.execute(text("ALTER TABLE storage_objects ADD COLUMN webview_url VARCHAR"))

                # Ensure storage_objects has HLS transcoding columns
                if "hls_url" not in scols:
                    conn.execute(text("ALTER TABLE storage_objects ADD COLUMN hls_url VARCHAR"))
                if "transcoding_status" not in scols:
                    conn.execute(text("ALTER TABLE storage_objects ADD COLUMN transcoding_status VARCHAR"))
                if "transcoding_progress" not in scols:
                    conn.execute(text("ALTER TABLE storage_objects ADD COLUMN transcoding_progress INTEGER"))
                if "transcoding_error" not in scols:
                    conn.execute(text("ALTER TABLE storage_objects ADD COLUMN transcoding_error TEXT"))
                if "transcoded_file_size_bytes" not in scols:
                    conn.execute(text("ALTER TABLE storage_objects ADD COLUMN transcoded_file_size_bytes INTEGER"))
                if "ai_safety_status" not in scols:
                    conn.execute(text("ALTER TABLE storage_objects ADD COLUMN ai_safety_status VARCHAR"))
                if "ai_safety_error" not in scols:
                    conn.execute(text("ALTER TABLE storage_objects ADD COLUMN ai_safety_error TEXT"))

                # Ensure storage_objects has multi-tenancy column
                if "tenant_id" not in scols:
                    conn.execute(text("ALTER TABLE storage_objects ADD COLUMN tenant_id VARCHAR(50) DEFAULT 'arkturian'"))
                    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_storage_tenant ON storage_objects (tenant_id)"))

                # Ensure storage_objects has AI metadata fields (v2.0)
                if "ai_title" not in scols:
                    conn.execute(text("ALTER TABLE storage_objects ADD COLUMN ai_title VARCHAR(500)"))
                if "ai_subtitle" not in scols:
                    conn.execute(text("ALTER TABLE storage_objects ADD COLUMN ai_subtitle TEXT"))
                if "ai_tags" not in scols:
                    conn.execute(text("ALTER TABLE storage_objects ADD COLUMN ai_tags JSON"))
                if "ai_collections" not in scols:
                    conn.execute(text("ALTER TABLE storage_objects ADD COLUMN ai_collections JSON"))
                if "safety_info" not in scols:
                    conn.execute(text("ALTER TABLE storage_objects ADD COLUMN safety_info JSON"))

                # Ensure storage_objects has storage mode fields (v3.0)
                if "storage_mode" not in scols:
                    conn.execute(text("ALTER TABLE storage_objects ADD COLUMN storage_mode VARCHAR DEFAULT 'copy'"))
                if "reference_path" not in scols:
                    conn.execute(text("ALTER TABLE storage_objects ADD COLUMN reference_path VARCHAR"))
                    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_storage_reference_path ON storage_objects (reference_path)"))
                if "external_uri" not in scols:
                    conn.execute(text("ALTER TABLE storage_objects ADD COLUMN external_uri VARCHAR"))
                    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_storage_external_uri ON storage_objects (external_uri)"))
                if "ai_context_metadata" not in scols:
                    conn.execute(text("ALTER TABLE storage_objects ADD COLUMN ai_context_metadata JSON"))

                conn.commit()
    except Exception as e:
        print(f"Warning: Database migration failed: {e}")
        # Do not fail startup on migration errors

async def disconnect_db():
    """Disconnect from database (for shutdown)"""
    await database.disconnect()
