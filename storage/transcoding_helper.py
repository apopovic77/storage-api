"""
Transcoding helper for storage API

Integrates the transcoding package with storage uploads
"""

import asyncio
from pathlib import Path
from typing import Optional
from config import settings
import logging

# Import our transcoding package
try:
    from transcoding import TranscoderFactory, TranscodingConfig, TranscodingMode
    TRANSCODING_AVAILABLE = True
except ImportError:
    TRANSCODING_AVAILABLE = False
    logging.warning("Transcoding package not available")


class TranscodingHelper:
    """Helper class for video transcoding integration"""

    @staticmethod
    def is_enabled() -> bool:
        """Check if transcoding is enabled"""
        if not TRANSCODING_AVAILABLE:
            return False
        return settings.TRANSCODING_MODE.lower() != "disabled"

    @staticmethod
    def should_transcode(mime_type: Optional[str]) -> bool:
        """Check if this file should be transcoded"""
        if not TranscodingHelper.is_enabled():
            return False

        if not mime_type:
            return False

        return mime_type.startswith("video/")

    @staticmethod
    async def transcode_video(
        source_path: Path,
        output_dir: Path,
        storage_object_id: int
    ) -> bool:
        """
        Transcode a video file

        Args:
            source_path: Path to source video
            output_dir: Directory for transcoded files
            storage_object_id: Storage object ID for reference

        Returns:
            bool: True if transcoding was started/succeeded
        """
        if not TRANSCODING_AVAILABLE:
            logging.error("Transcoding requested but package not available")
            return False

        try:
            # Create transcoding config
            config = TranscodingConfig.from_env(
                mode_str=settings.TRANSCODING_MODE,
                remote_url=settings.TRANSCODING_API_URL,
                remote_key=settings.TRANSCODING_API_KEY
            )

            # Create transcoder
            transcoder = TranscoderFactory.create(config)

            # Check availability
            if not await transcoder.check_availability():
                logging.error(f"Transcoder not available (mode: {settings.TRANSCODING_MODE})")
                return False

            logging.info(f"🎬 Starting transcoding for storage object {storage_object_id}")
            logging.info(f"   Mode: {settings.TRANSCODING_MODE}")
            logging.info(f"   Source: {source_path}")
            logging.info(f"   Output: {output_dir}")

            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)

            # Transcode (this will run async)
            result = await transcoder.transcode(source_path, output_dir)

            # Update database with transcoding result
            from database import SessionLocal
            from models import StorageObject

            db_session = SessionLocal()
            try:
                storage_obj = db_session.get(StorageObject, storage_object_id)
                if storage_obj:
                    if result.success:
                        logging.info(f"✅ Transcoding completed for storage object {storage_object_id}")
                        logging.info(f"   Created {len(result.variants)} variants")
                        for variant in result.variants:
                            logging.info(f"      - {variant.name}: {variant.resolution} @ {variant.bitrate_mbps:.1f} Mbps")

                        if result.thumbnails:
                            logging.info(f"   Generated {len(result.thumbnails)} thumbnails")

                        # Update database status
                        storage_obj.transcoding_status = "completed"
                        storage_obj.transcoding_progress = 100
                        storage_obj.transcoding_error = None

                        # Set HLS URL (master.m3u8 in output directory)
                        hls_master = output_dir / "master.m3u8"
                        if hls_master.exists():
                            # Construct public URL for HLS master playlist
                            # URL structure: /uploads/storage/media/{tenant}/{object_key}/master.m3u8
                            tenant_id = storage_obj.metadata_json.get("tenant_id", "arkturian") if storage_obj.metadata_json else "arkturian"
                            base_key = storage_obj.object_key.rsplit(".", 1)[0]  # Remove .mp4 extension
                            storage_obj.hls_url = f"/uploads/storage/media/{tenant_id}/{base_key}_transcoded/master.m3u8"
                            logging.info(f"   HLS URL set to: {storage_obj.hls_url}")

                        db_session.commit()
                        logging.info(f"   Database updated - status: completed")
                        return True
                    else:
                        logging.error(f"❌ Transcoding failed for storage object {storage_object_id}: {result.error}")

                        # Update database with failure
                        storage_obj.transcoding_status = "failed"
                        storage_obj.transcoding_error = str(result.error)
                        db_session.commit()
                        logging.info(f"   Database updated - status: failed")
                        return False
                else:
                    logging.error(f"Storage object {storage_object_id} not found in database")
                    return False
            finally:
                db_session.close()

        except Exception as e:
            logging.error(f"❌ Transcoding error for storage object {storage_object_id}: {e}")
            import traceback
            traceback.print_exc()
            return False

    @staticmethod
    async def start_background_transcoding(
        source_path: Path,
        output_dir: Path,
        storage_object_id: int
    ):
        """
        Start transcoding by sending request directly to transcoding-api

        NOTE: We send the request synchronously (awaited) instead of using asyncio.create_task()
        because background tasks created with create_task() get cancelled when the HTTP response
        is sent in gunicorn/uvicorn workers. The transcoding-api handles the actual transcoding
        asynchronously on its side.

        Args:
            source_path: Path to source video
            output_dir: Directory for transcoded files
            storage_object_id: Storage object ID for reference
        """
        import httpx

        try:
            logging.error(f"📤 START: Sending transcoding request for storage object {storage_object_id}")
            logging.error(f"   Source: {source_path}")
            logging.error(f"   API URL: {settings.TRANSCODING_API_URL}")

            # Check if transcoding is enabled
            if not TRANSCODING_AVAILABLE:
                logging.error("❌ Transcoding package not available")
                return

            if settings.TRANSCODING_MODE.lower() == "disabled":
                logging.error("❌ Transcoding is disabled")
                return

            # Send POST request directly to transcoding-api
            # The transcoding-api will queue the job and return immediately
            # Long timeout since transcoding is now synchronous (can take several minutes)
            async with httpx.AsyncClient(timeout=600.0) as client:
                # First check if transcoding-api is available
                try:
                    health_response = await client.get(f"{settings.TRANSCODING_API_URL}/health")
                    if health_response.status_code != 200:
                        logging.error(f"❌ Transcoding API not healthy: {health_response.status_code}")
                        return
                    logging.error(f"✅ Transcoding API is healthy")
                except Exception as health_err:
                    logging.error(f"❌ Cannot reach transcoding API: {health_err}")
                    return

                # Upload the video file to transcoding-api
                with open(source_path, "rb") as f:
                    files = {"file": (source_path.name, f, "video/mp4")}
                    headers = {}

                    if settings.TRANSCODING_API_KEY:
                        headers["X-API-Key"] = settings.TRANSCODING_API_KEY

                    logging.error(f"📤 Uploading {source_path.name} to transcoding-api...")

                    response = await client.post(
                        f"{settings.TRANSCODING_API_URL}/transcode",
                        files=files,
                        headers=headers
                    )

                    if response.status_code == 200:
                        result = response.json()
                        job_id = result.get("job_id")
                        status = result.get("status")
                        output_dir_str = result.get("output_dir")
                        variants = result.get("variants", [])
                        error = result.get("error")

                        logging.error(f"✅ Transcoding response received!")
                        logging.error(f"   Job ID: {job_id}")
                        logging.error(f"   Status: {status}")
                        logging.error(f"   Output Dir: {output_dir_str}")
                        logging.error(f"   Variants: {len(variants)}")

                        from database import SessionLocal
                        from models import StorageObject
                        import shutil

                        db_session = SessionLocal()
                        try:
                            storage_obj = db_session.get(StorageObject, storage_object_id)
                            if storage_obj:
                                if status == "completed" and output_dir_str:
                                    # Copy transcoded files to storage location
                                    output_source = Path(output_dir_str)

                                    # Determine destination directory (next to original file)
                                    tenant_id = storage_obj.metadata_json.get("tenant_id", "arkturian") if storage_obj.metadata_json else "arkturian"
                                    base_key = storage_obj.object_key.rsplit(".", 1)[0]  # Remove extension

                                    hls_dest_dir = Path(settings.STORAGE_UPLOAD_DIR) / "media" / tenant_id / f"{base_key}_transcoded"
                                    hls_dest_dir.mkdir(parents=True, exist_ok=True)

                                    logging.error(f"📁 Copying HLS files from {output_source} to {hls_dest_dir}")

                                    # Copy all HLS files
                                    for item in output_source.iterdir():
                                        dest_path = hls_dest_dir / item.name
                                        if item.is_file():
                                            shutil.copy2(item, dest_path)
                                            logging.error(f"   Copied: {item.name}")

                                    # Update database
                                    storage_obj.transcoding_status = "completed"
                                    storage_obj.transcoding_progress = 100
                                    storage_obj.transcoding_error = None
                                    # Use VOD_BASE_URL for full URL
                                    vod_base = settings.VOD_BASE_URL
                                    storage_obj.hls_url = f"{vod_base}/media/{tenant_id}/{base_key}_transcoded/master.m3u8"

                                    if not storage_obj.metadata_json:
                                        storage_obj.metadata_json = {}
                                    storage_obj.metadata_json['transcoding_job_id'] = job_id
                                    storage_obj.metadata_json['transcoding_variants'] = variants

                                    db_session.commit()
                                    logging.error(f"✅ Database updated - transcoding completed!")
                                    logging.error(f"   HLS URL: {storage_obj.hls_url}")

                                    # Cleanup temp directory
                                    try:
                                        shutil.rmtree(output_source.parent)
                                        logging.error(f"🧹 Cleaned up temp directory")
                                    except Exception as cleanup_err:
                                        logging.error(f"⚠️ Failed to cleanup temp dir: {cleanup_err}")

                                elif status == "failed":
                                    storage_obj.transcoding_status = "failed"
                                    storage_obj.transcoding_error = error or "Transcoding failed"
                                    db_session.commit()
                                    logging.error(f"❌ Transcoding failed: {error}")
                                else:
                                    logging.error(f"⚠️ Unexpected status: {status}")
                        finally:
                            db_session.close()
                    else:
                        logging.error(f"❌ Transcoding API error: {response.status_code} - {response.text}")

                        # Update database with failure
                        from database import SessionLocal
                        from models import StorageObject

                        db_session = SessionLocal()
                        try:
                            storage_obj = db_session.get(StorageObject, storage_object_id)
                            if storage_obj:
                                storage_obj.transcoding_status = "failed"
                                storage_obj.transcoding_error = f"API error: {response.status_code}"
                                db_session.commit()
                        finally:
                            db_session.close()

        except Exception as e:
            logging.error(f"❌ Failed to start transcoding: {e}")
            import traceback
            traceback.print_exc()


# Convenience function for easy import
async def transcode_if_needed(
    source_path: Path,
    mime_type: Optional[str],
    storage_object_id: int,
    output_dir: Optional[Path] = None
) -> bool:
    """
    Transcode video if needed

    Args:
        source_path: Path to uploaded file
        mime_type: MIME type of file
        storage_object_id: Storage object ID
        output_dir: Optional output directory (defaults to source_path parent / basename)

    Returns:
        bool: True if transcoding was triggered
    """
    if not TranscodingHelper.should_transcode(mime_type):
        return False

    if output_dir is None:
        # Default: create directory next to source file
        basename = source_path.stem
        output_dir = source_path.parent / basename

    # Start transcoding in background
    await TranscodingHelper.start_background_transcoding(
        source_path,
        output_dir,
        storage_object_id
    )

    return True
