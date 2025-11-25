"""
Storage API - Media Storage & AI Analysis Service

Handles:
- Object Storage (files, images, videos)
- Media Transformation (resize, format conversion)
- AI Analysis (Gemini Vision API)
- Knowledge Graph (embeddings, similarity search)
- Multi-Tenancy (tenant isolation)
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from storage import routes as storage_routes
from tenancy import routes as tenant_routes
from admin import routes as admin_routes
from database import connect_db, disconnect_db
from config import settings

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
)

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
