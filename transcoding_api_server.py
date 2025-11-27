"""
Local Transcoding API Server

FastAPI server that wraps the transcoding package for local FFmpeg transcoding.
Provides the same API interface as the Mac Transcoding API.

Run with: uvicorn transcoding_api_server:app --host 0.0.0.0 --port 8087
"""

import asyncio
import logging
import uuid
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile, Form
from pydantic import BaseModel
import httpx

# Try to import transcoding package
try:
    from transcoding import TranscoderFactory, TranscodingConfig, TranscodingMode
    TRANSCODING_AVAILABLE = True
except ImportError:
    TRANSCODING_AVAILABLE = False
    logging.warning("Transcoding package not available - install with: pip install -e /path/to/mac_transcoding_api")

app = FastAPI(title="Local Transcoding API", version="1.0.0")

# Job storage (in-memory, simple implementation)
jobs: Dict[str, dict] = {}
current_job_id: Optional[str] = None


class TranscodeRequest(BaseModel):
    """Transcoding job request"""
    job_id: str
    source_url: str
    callback_url: Optional[str] = None
    file_size_bytes: int
    original_filename: str
    storage_object_id: str
    download_headers: Optional[Dict[str, str]] = None


class JobStatus(BaseModel):
    """Job status response"""
    job_id: str
    status: str  # queued, downloading, transcoding, uploading, completed, failed
    progress: int  # 0-100
    message: Optional[str] = None
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


@app.get("/")
async def root():
    """Health check and status"""
    return {
        "service": "Local Transcoding API",
        "status": "online",
        "active_jobs": 1 if current_job_id else 0,
        "queue_size": len([j for j in jobs.values() if j["status"] == "queued"]),
        "current_job": current_job_id,
        "mode": "local_ffmpeg"
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    if not TRANSCODING_AVAILABLE:
        raise HTTPException(status_code=503, detail="Transcoding package not available")

    # Check if ffmpeg is available
    try:
        config = TranscodingConfig(mode=TranscodingMode.LOCAL)
        transcoder = TranscoderFactory.create(config)
        available = await transcoder.check_availability()

        if not available:
            raise HTTPException(status_code=503, detail="FFmpeg not available")

        return {"status": "healthy", "ffmpeg": "available"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")


@app.post("/transcode")
async def transcode(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Submit a transcoding job via file upload

    The job will be processed in the background. Use GET /status/{job_id} to check progress.
    """
    if not TRANSCODING_AVAILABLE:
        raise HTTPException(status_code=503, detail="Transcoding package not available")

    # Generate job ID
    job_id = str(uuid.uuid4())

    # Save uploaded file to temp directory
    temp_dir = Path(f"/tmp/transcode_{job_id}")
    temp_dir.mkdir(parents=True, exist_ok=True)

    source_file = temp_dir / file.filename

    # Write uploaded file
    with open(source_file, "wb") as f:
        content = await file.read()
        f.write(content)

    logging.info(f"📥 Received file: {file.filename} ({len(content)} bytes)")

    # Create job record
    job = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0,
        "message": "Job queued",
        "source_file": str(source_file),
        "original_filename": file.filename,
        "file_size": len(content),
        "started_at": datetime.utcnow().isoformat(),
        "completed_at": None,
        "error": None
    }

    jobs[job_id] = job

    # Start processing in background
    background_tasks.add_task(process_job_from_file, job_id)

    logging.info(f"📤 Transcoding job queued: {job_id} - {file.filename}")

    return {
        "job_id": job_id,
        "status": "queued",
        "message": "Job queued for processing"
    }


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """Get job status"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]

    return JobStatus(
        job_id=job["job_id"],
        status=job["status"],
        progress=job["progress"],
        message=job.get("message"),
        error=job.get("error"),
        started_at=job.get("started_at"),
        completed_at=job.get("completed_at")
    )


@app.post("/cancel/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a job"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]

    if job["status"] in ["completed", "failed"]:
        return {"message": "Job already finished", "status": job["status"]}

    job["status"] = "cancelled"
    job["message"] = "Job cancelled by user"
    job["completed_at"] = datetime.utcnow().isoformat()

    logging.info(f"🚫 Job cancelled: {job_id}")

    return {"message": "Job cancelled", "job_id": job_id}


async def process_job_from_file(job_id: str):
    """
    Process a transcoding job from uploaded file

    Steps:
    1. File already uploaded
    2. Transcode using local FFmpeg
    3. Upload results back (TODO: implement upload to storage)
    4. Call callback URL (if provided)
    """
    global current_job_id

    job = jobs[job_id]
    current_job_id = job_id

    try:
        source_file = Path(job["source_file"])

        logging.info(f"📁 Processing file: {source_file} ({job['file_size']} bytes)")

        # Step 1: Transcode
        job["status"] = "transcoding"
        job["progress"] = 30
        job["message"] = "Transcoding video"
        logging.info(f"🎬 Transcoding: {source_file}")

        # Create output directory
        output_dir = source_file.parent / "output"
        output_dir.mkdir(exist_ok=True)

        # Create transcoder
        config = TranscodingConfig(mode=TranscodingMode.LOCAL)
        transcoder = TranscoderFactory.create(config)

        # Transcode
        result = await transcoder.transcode(source_file, output_dir)

        if not result.success:
            raise Exception(f"Transcoding failed: {result.error}")

        logging.info(f"✅ Transcoded {len(result.variants)} variants")
        for variant in result.variants:
            logging.info(f"   - {variant.name}: {variant.resolution} @ {variant.bitrate_mbps:.1f} Mbps")

        # Job completed
        job["status"] = "completed"
        job["progress"] = 100
        job["message"] = f"Transcoding completed - {len(result.variants)} variants created"
        job["completed_at"] = datetime.utcnow().isoformat()
        job["output_dir"] = str(output_dir)
        job["variants"] = [
            {
                "name": v.name,
                "resolution": v.resolution,
                "bitrate_mbps": v.bitrate_mbps,
                "file_path": str(v.output_path) if v.output_path else None
            }
            for v in result.variants
        ]

        logging.info(f"✅ Job completed: {job_id}")

    except Exception as e:
        logging.error(f"❌ Job failed: {job_id} - {str(e)}")
        import traceback
        traceback.print_exc()

        job["status"] = "failed"
        job["progress"] = 0
        job["error"] = str(e)
        job["message"] = "Transcoding failed"
        job["completed_at"] = datetime.utcnow().isoformat()

    finally:
        current_job_id = None


if __name__ == "__main__":
    import uvicorn

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Run server
    uvicorn.run(app, host="0.0.0.0", port=8087)
