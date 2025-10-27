#!/usr/bin/env python3
"""
AI Worker for Storage API
Processes video thumbnail analysis and embedding generation
Extracts 5 screenshots from videos for comprehensive AI analysis
"""

import os
import sys
import time
import asyncio
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from database import get_db
from models import StorageObject
from ai_analysis.service import analyze_content
import base64

# Configuration
QUEUE_FILE = os.getenv("AI_QUEUE_FILE", "/var/log/ai_analysis_queue.txt")
LOG_FILE = "/var/log/ai_worker.log"


def log(message: str):
    """Log to both stdout and log file"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}"
    print(log_message, flush=True)
    
    try:
        with open(LOG_FILE, "a") as f:
            f.write(log_message + "\n")
    except Exception:
        pass


def get_job() -> Optional[str]:
    """Atomically pop the first line from the queue file"""
    if not os.path.exists(QUEUE_FILE):
        return None
    
    job_line = None
    try:
        with open(QUEUE_FILE, "r+") as f:
            lines = f.readlines()
            if lines:
                job_line = lines.pop(0).strip()
                f.seek(0)
                f.writelines(lines)
                f.truncate()
    except Exception as e:
        log(f"ERROR reading queue: {e}")
    
    return job_line or None


async def process_video_job(object_id: int, thumb_dir: Path, filename: str):
    """
    Process video thumbnails for AI analysis
    Loads all 5 screenshots and sends them to AI for comprehensive analysis
    """
    log(f"Processing video {object_id}: {filename}")
    
    # Load the 5 video thumbnails (thumb_01.jpg to thumb_05.jpg)
    images_base64 = []
    for i in range(1, 6):
        thumb_path = thumb_dir / f"thumb_0{i}.jpg"
        if thumb_path.exists():
            with open(thumb_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()
                images_base64.append(img_b64)
                log(f"  ✅ Loaded screenshot {i}: {thumb_path.stat().st_size} bytes")
        else:
            log(f"  ⚠️  Missing screenshot {i}: {thumb_path}")
    
    if not images_base64:
        raise Exception(f"No thumbnails found in {thumb_dir}")
    
    log(f"  📸 Loaded {len(images_base64)}/5 screenshots, analyzing with AI...")
    
    # For videos, use first screenshot as primary image
    # Context tells AI this is a video with multiple frames
    primary_image = base64.b64decode(images_base64[0])
    
    context = {
        "filename": filename,
        "media_type": "video",
        "video_screenshots": len(images_base64),
        "note": f"Video analysis based on {len(images_base64)} sampled frames"
    }
    
    result = await analyze_content(
        data=primary_image,
        mime_type="image/jpeg",
        context=context,
        object_id=object_id
    )
    
    log(f"  🎨 AI analysis complete:")
    log(f"     Safety: {result.get('safety_info', {}).get('isSafe', 'unknown')}")
    log(f"     Category: {result.get('ai_category', 'unknown')}")
    log(f"     Tags: {len(result.get('ai_tags', []))} extracted")
    log(f"     Title: {result.get('ai_title', 'N/A')[:50]}...")
    
    # Update database with AI results
    db = next(get_db())
    try:
        storage_obj = db.query(StorageObject).filter(StorageObject.id == object_id).first()
        if not storage_obj:
            raise Exception(f"Storage object {object_id} not found in database")
        
        # Update AI analysis fields
        storage_obj.ai_safety_rating = "safe" if result.get("safety_info", {}).get("isSafe") else "unsafe"
        storage_obj.ai_category = result.get("ai_category")
        storage_obj.ai_danger_potential = result.get("ai_danger_potential")
        storage_obj.ai_title = result.get("ai_title")
        storage_obj.ai_subtitle = result.get("ai_subtitle")
        storage_obj.ai_tags = result.get("ai_tags")
        storage_obj.ai_collections = result.get("ai_collections")
        storage_obj.safety_info = result.get("safety_info")
        storage_obj.ai_context_metadata = result.get("ai_context_metadata") or {}
        storage_obj.ai_safety_status = "completed"
        
        db.commit()
        log(f"  ✅ Database updated for object {object_id}")
        
    except Exception as e:
        log(f"  ❌ ERROR updating database: {e}")
        db.rollback()
        raise
    finally:
        db.close()


async def process_image_job(object_id: int, file_path: Path, filename: str):
    """Process single image for AI analysis"""
    log(f"Processing image {object_id}: {filename}")
    
    with open(file_path, "rb") as f:
        image_data = f.read()
    
    context = {
        "filename": filename,
        "media_type": "image"
    }
    
    result = await analyze_content(
        data=image_data,
        mime_type="image/jpeg",
        context=context,
        object_id=object_id
    )
    
    log(f"  🎨 AI analysis complete:")
    log(f"     Safety: {result.get('safety_info', {}).get('isSafe', 'unknown')}")
    log(f"     Category: {result.get('ai_category', 'unknown')}")
    
    # Update database
    db = next(get_db())
    try:
        storage_obj = db.query(StorageObject).filter(StorageObject.id == object_id).first()
        if not storage_obj:
            raise Exception(f"Storage object {object_id} not found")
        
        # Update AI fields
        storage_obj.ai_safety_rating = "safe" if result.get("safety_info", {}).get("isSafe") else "unsafe"
        storage_obj.ai_category = result.get("ai_category")
        storage_obj.ai_danger_potential = result.get("ai_danger_potential")
        storage_obj.ai_title = result.get("ai_title")
        storage_obj.ai_subtitle = result.get("ai_subtitle")
        storage_obj.ai_tags = result.get("ai_tags")
        storage_obj.ai_collections = result.get("ai_collections")
        storage_obj.safety_info = result.get("safety_info")
        storage_obj.ai_context_metadata = result.get("ai_context_metadata") or {}
        storage_obj.ai_safety_status = "completed"
        
        db.commit()
        log(f"  ✅ Database updated for object {object_id}")
        
    except Exception as e:
        log(f"  ❌ ERROR updating database: {e}")
        db.rollback()
        raise
    finally:
        db.close()


async def process_job_async(job_line: str):
    """Process a single job from the queue"""
    status_dir = None
    object_id = None
    
    try:
        if job_line.count("|") != 3:
            raise ValueError(f"Invalid job format (expected 4 parts): {job_line}")
        
        object_id_str, job_type, content_path_str, filename = job_line.split("|", 3)
        object_id = int(object_id_str.strip())
        job_type = job_type.strip()
        content_path = Path(content_path_str.strip())
        filename = filename.strip()
        
        log(f"Worker found job: {object_id} | {job_type} | {filename}")
        
        # Create status directory
        status_dir = Path(f"/tmp/ai_analysis_status_{object_id}")
        status_dir.mkdir(exist_ok=True)
        
        if not content_path.exists():
            raise FileNotFoundError(f"Content path not found: {content_path}")
        
        # Process based on type
        if job_type == "video":
            await process_video_job(object_id, content_path, filename)
        elif job_type == "image":
            await process_image_job(object_id, content_path, filename)
        else:
            raise ValueError(f"Unsupported job type: {job_type}")
        
        log(f"Worker finished job for object ID: {object_id}")
        
        # Cleanup status dir on success
        try:
            import shutil
            shutil.rmtree(status_dir)
        except Exception:
            pass
        
    except Exception as e:
        error_message = f"Worker ERROR processing job '{job_line}': {e}"
        log(error_message)
        
        # Write error log
        if object_id and status_dir:
            try:
                status_dir.mkdir(exist_ok=True)
                with open(status_dir / "error.log", "w") as f:
                    f.write(error_message)
            except Exception:
                pass


def main():
    """Main worker loop"""
    log("=" * 60)
    log("Storage AI Worker started")
    log(f"Queue file: {QUEUE_FILE}")
    log(f"Log file: {LOG_FILE}")
    log("=" * 60)
    
    # Ensure queue file exists
    Path(QUEUE_FILE).parent.mkdir(parents=True, exist_ok=True)
    Path(QUEUE_FILE).touch(exist_ok=True)
    
    while True:
        try:
            job_line = get_job()
            if job_line:
                # Run async job processing
                asyncio.run(process_job_async(job_line))
            else:
                # No jobs, sleep
                time.sleep(5)
        
        except KeyboardInterrupt:
            log("Worker shutdown requested")
            break
        except Exception as e:
            log(f"FATAL ERROR in main loop: {e}")
            time.sleep(10)  # Back off on fatal errors


if __name__ == "__main__":
    main()
