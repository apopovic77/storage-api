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
        """Called when task fails"""
        logger.error(f"❌ Task {self.name} failed (ID: {task_id}): {exc}")
        logger.debug(f"   Exception info: {einfo}")
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
