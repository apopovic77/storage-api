"""
Storage API - Media Storage & AI Analysis Service

Handles:
- Object Storage (files, images, videos)
- Media Transformation (resize, format conversion)
- AI Analysis (Gemini Vision API)
- Knowledge Graph (embeddings, similarity search)
- Multi-Tenancy (tenant isolation)
"""

# Configure logging FIRST - before any other imports
import logging
import sys

# Configure root logger to output to stderr (captured by gunicorn error log)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)

# Also configure uvicorn loggers
logging.getLogger("uvicorn.error").setLevel(logging.DEBUG)
logging.getLogger("uvicorn.access").setLevel(logging.INFO)

import re
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from storage import routes as storage_routes
from tenancy import routes as tenant_routes
from admin import routes as admin_routes
from database import connect_db, disconnect_db, SessionLocal
from config import settings
from models import StorageObject

app = FastAPI(
    title="Storage API",
    version="1.0.0",
    description="Media storage and AI analysis service with multi-tenancy support"
)

# Database lifecycle events
@app.on_event("startup")
async def startup():
    await connect_db()

@app.on_event("shutdown")
async def shutdown():
    await disconnect_db()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "Content-Length",
        "Content-Range",
        "X-HLS-URL",
        "X-Transcoding-Status",
        "X-Mime-Type",
    ],
)

_MEDIA_ID_RE = re.compile(r"^/storage/media/(\d+)/?$")
_EXPOSE_HEADERS = "Content-Length, Content-Range, X-HLS-URL, X-Transcoding-Status, X-Mime-Type"


def _apply_cors_for_head(headers: dict, request: Request) -> dict:
    """
    Manually attach CORS response headers because our HEAD-intercept middleware
    runs OUTER of CORSMiddleware in Starlette's stack — returning a Response
    here bypasses CORSMiddleware's send-wrap, so it never sees the response
    to inject Access-Control-Expose-Headers / Allow-Origin. We mirror the
    behavior here for HEAD /storage/media/{id} only.
    """
    origin = request.headers.get("origin")
    if origin:
        allowed = settings.CORS_ORIGINS or []
        if "*" in allowed or origin in allowed:
            headers["Access-Control-Allow-Origin"] = origin
            headers["Vary"] = "Origin"
            headers["Access-Control-Allow-Credentials"] = "true"
    headers["Access-Control-Expose-Headers"] = _EXPOSE_HEADERS
    return headers


@app.middleware("http")
async def storage_media_head_headers(request: Request, call_next):
    """
    For HEAD /storage/media/{id} requests, return a body-less response with
    X-Mime-Type, X-Transcoding-Status, X-HLS-URL headers so HLS-aware
    frontends can pick HLS vs progressive MP4 without a second round-trip.

    Bypasses FastAPI's Starlette auto-HEAD handler which strips custom
    headers from FileResponse. Runs before any route handler.
    """
    if request.method == "HEAD":
        m = _MEDIA_ID_RE.match(request.url.path)
        if m:
            object_id = int(m.group(1))
            db = SessionLocal()
            try:
                obj = db.query(StorageObject).filter(StorageObject.id == object_id).first()
            finally:
                db.close()
            if not obj:
                return Response(status_code=404, headers=_apply_cors_for_head({}, request))
            headers = {"Content-Disposition": "inline", "Accept-Ranges": "bytes"}
            if obj.mime_type:
                headers["X-Mime-Type"] = obj.mime_type
            if obj.transcoding_status:
                headers["X-Transcoding-Status"] = obj.transcoding_status
            if obj.hls_url and obj.transcoding_status == "completed":
                headers["X-HLS-URL"] = obj.hls_url
            if obj.file_size_bytes:
                headers["Content-Length"] = str(obj.file_size_bytes)
            _apply_cors_for_head(headers, request)
            return Response(
                status_code=200,
                media_type=(obj.mime_type or "application/octet-stream"),
                headers=headers,
            )
    return await call_next(request)


# Include routers
app.include_router(storage_routes.router, prefix="/storage", tags=["Storage"])
# Include tenant routes (router already has '/tenants' prefix)
app.include_router(tenant_routes.router, tags=["Tenancy"])
app.include_router(admin_routes.router, prefix="/admin", tags=["Admin"])

@app.get("/")
def root():
    return {
        "service": "storage-api",
        "version": "1.0.0",
        "description": "Media storage and AI analysis service"
    }

@app.get("/health")
def health():
    return {"status": "healthy", "service": "storage-api"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
