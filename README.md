# Storage API

Media storage and AI analysis service with multi-tenancy support.

## Features

- 📦 Object Storage (files, images, videos)
- 🖼️ Media Transformation (resize, format conversion)
- 🤖 AI Analysis (Gemini Vision API)
- 🧠 Knowledge Graph (embeddings, similarity search)
- 🏢 Multi-Tenancy (isolated tenant data)
- 🔍 Semantic Search

## Tech Stack

- FastAPI
- SQLAlchemy + SQLite/PostgreSQL
- Pillow (image processing)
- Google Gemini AI
- OpenAI Embeddings
- ChromaDB (vector database)

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Start server
uvicorn main:app --host 0.0.0.0 --port 8002
```

## API Documentation

Once running, visit:
- OpenAPI docs: http://localhost:8002/docs
- ReDoc: http://localhost:8002/redoc

## Environment Variables

See `.env.example` for required configuration.

## Multi-Tenancy

This service supports multiple tenants with data isolation:
- Tenant ID mapping via API keys
- Isolated storage per tenant
- Shared AI analysis pipeline

## Features

### Media Transformation
```
GET /storage/media/{id}?width=400&format=webp&quality=80
```

### AI Analysis
- Automatic image analysis with Gemini Vision
- Safety rating
- Object detection
- Semantic tagging

### Knowledge Graph
- Vector embeddings (3072-dim)
- Similarity search
- Semantic text search

## Development

```bash
# Run locally
python main.py

# Run tests
pytest

# Backfill embeddings
python scripts/backfill_embeddings.py
```
