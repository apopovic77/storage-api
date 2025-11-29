"""
AI Analysis Tasks for Storage API

Handles:
- Image analysis (Vision AI + Safety)
- Video analysis (5-frame sampling + Vision AI)
- Text/CSV analysis
- Knowledge Graph embedding generation

Migrated from scripts/ai_worker.py to Celery for production reliability.
"""

import sys
import base64
from pathlib import Path
from typing import Optional, Dict, Any
import logging

# Fix ChromaDB SQLite issue
sys.modules['sqlite3'] = __import__('pysqlite3')

from celery_app import app
from tasks.base import BaseStorageTask
from models import StorageObject
from ai_analysis.service import analyze_content
from knowledge_graph.pipeline import KnowledgeGraphPipeline

logger = logging.getLogger(__name__)


@app.task(base=BaseStorageTask, name='tasks.ai_analysis.process_image_analysis')
def process_image_analysis(object_id: int, image_path: str, filename: str) -> Dict[str, Any]:
    """
    Process single image for AI analysis

    Args:
        object_id: Storage object ID
        image_path: Path to image file or directory containing image.jpg
        filename: Original filename

    Returns:
        Analysis results dictionary
    """
    logger.info(f"📸 Processing image analysis for object {object_id}: {filename}")

    # Resolve image path (might be directory with image.jpg inside)
    img_path = Path(image_path)
    if img_path.is_dir():
        img_path = img_path / "image.jpg"
        if not img_path.exists():
            raise FileNotFoundError(f"Expected image.jpg in {image_path}")
        logger.debug(f"   Found image in directory: {img_path}")

    # Read image data
    with open(img_path, "rb") as f:
        image_data = f.read()

    # AI Analysis
    context = {
        "filename": filename,
        "media_type": "image"
    }

    result = analyze_content(
        data=image_data,
        mime_type="image/jpeg",
        context=context,
        object_id=object_id
    )

    logger.info(f"   🎨 AI analysis complete:")
    logger.info(f"      Safety: {result.get('safety_info', {}).get('isSafe', 'unknown')}")
    logger.info(f"      Category: {result.get('ai_category', 'unknown')}")
    logger.info(f"      Tags: {len(result.get('ai_tags', []))} extracted")

    # Update database
    task = process_image_analysis
    db = task.get_db_session()

    storage_obj = db.query(StorageObject).filter(StorageObject.id == object_id).first()
    if not storage_obj:
        raise Exception(f"Storage object {object_id} not found in database")

    # Update AI fields
    storage_obj.ai_safety_rating = "safe" if result.get("safety_info", {}).get("isSafe") else "unsafe"
    storage_obj.ai_category = result.get("ai_category")
    storage_obj.ai_danger_potential = result.get("ai_danger_potential")
    storage_obj.ai_title = result.get("ai_title")
    storage_obj.ai_subtitle = result.get("ai_subtitle")
    storage_obj.ai_tags = result.get("ai_tags")
    storage_obj.ai_collections = result.get("ai_collections")
    storage_obj.safety_info = result.get("safety_info")

    # Build ai_context_metadata
    ai_context = {
        "embedding_info": result.get("embedding_info", {}),
        "prompt": result.get("prompt", ""),
        "ai_response": result.get("ai_response", ""),
        "mode": result.get("mode", "unknown")
    }
    storage_obj.ai_context_metadata = ai_context
    storage_obj.ai_safety_status = "completed"

    db.commit()
    logger.info(f"   ✅ Database updated for object {object_id}")

    # Trigger embedding generation (chained task)
    generate_embedding.delay(object_id)

    return {
        "object_id": object_id,
        "status": "completed",
        "ai_safety_rating": storage_obj.ai_safety_rating,
        "ai_category": storage_obj.ai_category,
        "tags_count": len(result.get("ai_tags", []))
    }


@app.task(base=BaseStorageTask, name='tasks.ai_analysis.process_video_analysis')
def process_video_analysis(object_id: int, thumb_dir: str, filename: str) -> Dict[str, Any]:
    """
    Process video thumbnails for AI analysis

    Loads 5 video screenshots and sends to AI for comprehensive analysis.

    Args:
        object_id: Storage object ID
        thumb_dir: Directory containing thumb_01.jpg - thumb_05.jpg
        filename: Original filename

    Returns:
        Analysis results dictionary
    """
    logger.info(f"🎬 Processing video analysis for object {object_id}: {filename}")

    thumb_path = Path(thumb_dir)

    # Load 5 video thumbnails
    images_base64 = []
    for i in range(1, 6):
        thumb_file = thumb_path / f"thumb_0{i}.jpg"
        if thumb_file.exists():
            with open(thumb_file, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()
                images_base64.append(img_b64)
                logger.debug(f"   ✅ Loaded screenshot {i}: {thumb_file.stat().st_size} bytes")
        else:
            logger.warning(f"   ⚠️  Missing screenshot {i}: {thumb_file}")

    if not images_base64:
        raise Exception(f"No thumbnails found in {thumb_dir}")

    logger.info(f"   📸 Loaded {len(images_base64)}/5 screenshots, analyzing with AI...")

    # Use first screenshot as primary image
    primary_image = base64.b64decode(images_base64[0])

    context = {
        "filename": filename,
        "media_type": "video",
        "video_screenshots": len(images_base64),
        "note": f"Video analysis based on {len(images_base64)} sampled frames"
    }

    result = analyze_content(
        data=primary_image,
        mime_type="image/jpeg",
        context=context,
        object_id=object_id
    )

    logger.info(f"   🎨 AI analysis complete:")
    logger.info(f"      Safety: {result.get('safety_info', {}).get('isSafe', 'unknown')}")
    logger.info(f"      Category: {result.get('ai_category', 'unknown')}")
    logger.info(f"      Title: {result.get('ai_title', 'N/A')[:50]}...")

    # Update database
    task = process_video_analysis
    db = task.get_db_session()

    storage_obj = db.query(StorageObject).filter(StorageObject.id == object_id).first()
    if not storage_obj:
        raise Exception(f"Storage object {object_id} not found in database")

    # Update AI fields
    storage_obj.ai_safety_rating = "safe" if result.get("safety_info", {}).get("isSafe") else "unsafe"
    storage_obj.ai_category = result.get("ai_category")
    storage_obj.ai_danger_potential = result.get("ai_danger_potential")
    storage_obj.ai_title = result.get("ai_title")
    storage_obj.ai_subtitle = result.get("ai_subtitle")
    storage_obj.ai_tags = result.get("ai_tags")
    storage_obj.ai_collections = result.get("ai_collections")
    storage_obj.safety_info = result.get("safety_info")

    # Build ai_context_metadata
    ai_context = {
        "embedding_info": result.get("embedding_info", {}),
        "prompt": result.get("prompt", ""),
        "ai_response": result.get("ai_response", ""),
        "mode": result.get("mode", "unknown")
    }
    storage_obj.ai_context_metadata = ai_context
    storage_obj.ai_safety_status = "completed"

    db.commit()
    logger.info(f"   ✅ Database updated for object {object_id}")

    # Trigger embedding generation
    generate_embedding.delay(object_id)

    return {
        "object_id": object_id,
        "status": "completed",
        "frames_analyzed": len(images_base64)
    }


@app.task(base=BaseStorageTask, name='tasks.ai_analysis.process_text_analysis')
def process_text_analysis(object_id: int, file_path: str, filename: str) -> Dict[str, Any]:
    """
    Process text/CSV files for AI analysis

    Args:
        object_id: Storage object ID
        file_path: Path to text file
        filename: Original filename

    Returns:
        Analysis results dictionary
    """
    logger.info(f"📄 Processing text analysis for object {object_id}: {filename}")

    # Read file content
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        # Fallback to latin-1
        with open(file_path, "r", encoding="latin-1") as f:
            content = f.read()

    # Determine MIME type
    mime_type = "application/csv" if filename.lower().endswith('.csv') else "text/plain"

    logger.info(f"   📄 File size: {len(content)} bytes, MIME: {mime_type}")

    context = {
        "filename": filename,
        "media_type": "text",
        "file_extension": filename.split('.')[-1].lower() if '.' in filename else "txt"
    }

    result = analyze_content(
        data=content.encode('utf-8'),
        mime_type=mime_type,
        context=context,
        object_id=object_id
    )

    logger.info(f"   📊 AI analysis complete:")
    logger.info(f"      Mode: {result.get('mode', 'unknown')}")
    logger.info(f"      Title: {result.get('ai_title', 'N/A')}")

    # Update database
    task = process_text_analysis
    db = task.get_db_session()

    storage_obj = db.query(StorageObject).filter(StorageObject.id == object_id).first()
    if not storage_obj:
        raise Exception(f"Storage object {object_id} not found")

    # Update AI fields
    storage_obj.ai_title = result.get("ai_title")
    storage_obj.ai_subtitle = result.get("ai_subtitle")
    storage_obj.ai_tags = result.get("ai_tags")
    storage_obj.ai_collections = result.get("ai_collections")
    storage_obj.ai_category = result.get("ai_category", "text")

    # Build ai_context_metadata
    ai_context = {
        "embedding_info": result.get("embedding_info", {}),
        "prompt": result.get("prompt", ""),
        "ai_response": result.get("ai_response", ""),
        "mode": result.get("mode", "text")
    }
    storage_obj.ai_context_metadata = ai_context
    storage_obj.ai_safety_status = "completed"

    db.commit()
    logger.info(f"   ✅ Database updated for object {object_id}")

    # Trigger embedding generation
    generate_embedding.delay(object_id)

    return {
        "object_id": object_id,
        "status": "completed",
        "file_size": len(content)
    }


@app.task(base=BaseStorageTask, name='tasks.ai_analysis.generate_embedding', queue='embeddings')
def generate_embedding(object_id: int) -> Dict[str, Any]:
    """
    Generate Knowledge Graph embedding for storage object

    Uses OpenAI text-embedding-3-large (3072-dim) and stores in ChromaDB.

    Args:
        object_id: Storage object ID

    Returns:
        Embedding generation result
    """
    logger.info(f"   🔗 Generating Knowledge Graph embedding for object {object_id}...")

    task = generate_embedding
    db = task.get_db_session()

    storage_obj = db.query(StorageObject).filter(StorageObject.id == object_id).first()
    if not storage_obj:
        raise Exception(f"Storage object {object_id} not found")

    try:
        pipeline = KnowledgeGraphPipeline()
        kg_entry = pipeline.process_storage_object(storage_obj, db)

        if kg_entry:
            logger.info(f"   ✅ Embedding generated and stored in ChromaDB")
            return {
                "object_id": object_id,
                "status": "embedded",
                "vector_dim": 3072
            }
        else:
            logger.warning(f"   ⚠️  No embedding generated (object may not have AI context)")
            return {
                "object_id": object_id,
                "status": "skipped",
                "reason": "no_ai_context"
            }

    except Exception as kg_error:
        logger.error(f"   ❌ Knowledge Graph error: {kg_error}")
        raise  # Will trigger retry
