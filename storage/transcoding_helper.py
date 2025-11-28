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
    def start_background_transcoding(
        source_path: Path,
        output_dir: Path,
        storage_object_id: int
    ):
        """
        Start transcoding in the background (fire and forget)

        Args:
            source_path: Path to source video
            output_dir: Directory for transcoded files
            storage_object_id: Storage object ID for reference
        """
        try:
            # Create a new event loop task
            loop = asyncio.get_event_loop()
            loop.create_task(
                TranscodingHelper.transcode_video(source_path, output_dir, storage_object_id)
            )
            logging.info(f"📤 Background transcoding queued for storage object {storage_object_id}")
        except Exception as e:
            logging.error(f"Failed to queue background transcoding: {e}")


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
    TranscodingHelper.start_background_transcoding(
        source_path,
        output_dir,
        storage_object_id
    )

    return True
