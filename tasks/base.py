"""
Abstract Base Task for Storage API

Provides common functionality for all tasks:
- Logging
- Error handling
- Database session management
- Metrics
"""

from celery import Task
from typing import Any, Dict, Optional
import logging
from database import SessionLocal
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class BaseStorageTask(Task):
    """
    Abstract base class for all Storage API tasks

    Features:
    - Automatic database session management
    - Structured logging
    - Error handling with retry logic
    - Performance metrics
    """

    # Retry configuration
    autoretry_for = (Exception,)
    max_retries = 3
    retry_backoff = True
    retry_backoff_max = 600  # 10 minutes max
    retry_jitter = True

    def __init__(self):
        super().__init__()
        self._db_session: Optional[Session] = None

    def before_start(self, task_id: str, args: tuple, kwargs: dict) -> None:
        """Called before task execution starts"""
        logger.info(f"🚀 Starting task {self.name} (ID: {task_id})")
        logger.debug(f"   Args: {args}")
        logger.debug(f"   Kwargs: {kwargs}")

    def on_success(self, retval: Any, task_id: str, args: tuple, kwargs: dict) -> None:
        """Called when task succeeds"""
        logger.info(f"✅ Task {self.name} completed successfully (ID: {task_id})")
        self._cleanup_db_session()

    def on_failure(self, exc: Exception, task_id: str, args: tuple, kwargs: dict, einfo) -> None:
        """Called when task fails (after all retries exhausted)."""
        logger.error(f"❌ Task {self.name} failed (ID: {task_id}): {exc}")
        logger.debug(f"   Exception info: {einfo}")

        # For AI-analysis tasks, mark the object as ai_safety_status='failed' so the
        # /storage/media quarantine logic treats the asset as untrusted until a
        # manual re-analysis succeeds. Fail-closed across the whole pipeline.
        ai_task_names = (
            "tasks.ai_analysis.process_safety_check_only",
            "tasks.ai_analysis.process_vision_analysis_only",
            "tasks.ai_analysis.process_image_analysis",
            "tasks.ai_analysis.process_video_analysis",
            "tasks.ai_analysis.process_text_analysis",
        )
        if self.name in ai_task_names and args:
            try:
                from models import StorageObject
                object_id = args[0]
                db = SessionLocal()
                try:
                    obj = db.query(StorageObject).filter(StorageObject.id == object_id).first()
                    if obj is not None:
                        obj.ai_safety_status = "failed"
                        obj.ai_safety_error = (str(exc) or "AI task failed")[:500]
                        db.commit()
                        logger.info(f"   ↳ Marked object {object_id} ai_safety_status=failed")
                finally:
                    db.close()
            except Exception as mark_exc:
                logger.error(f"   ↳ Could not mark ai_safety_status=failed: {mark_exc}")

        self._cleanup_db_session()

    def on_retry(self, exc: Exception, task_id: str, args: tuple, kwargs: dict, einfo) -> None:
        """Called when task is retried"""
        logger.warning(f"🔄 Retrying task {self.name} (ID: {task_id}) after error: {exc}")
        self._cleanup_db_session()

    def get_db_session(self) -> Session:
        """
        Get or create database session for this task

        Returns:
            SQLAlchemy Session
        """
        if not self._db_session:
            self._db_session = SessionLocal()
        return self._db_session

    def _cleanup_db_session(self) -> None:
        """Close database session if open"""
        if self._db_session:
            try:
                self._db_session.close()
            except Exception as e:
                logger.error(f"Error closing database session: {e}")
            finally:
                self._db_session = None
