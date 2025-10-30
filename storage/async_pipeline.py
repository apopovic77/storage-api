"""
Async Pipeline Manager for Arkturian Knowledge Graph System

This module provides async task management for AI analysis and knowledge graph building.
It supports two modes:
- fast: Single flash model for everything (quick results)
- quality: Flash for safety check, Pro with thinking for embeddings (best results)
"""

import asyncio
import uuid
import json
import traceback
from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum
from dataclasses import dataclass, asdict

from sqlalchemy.orm import Session
from models import StorageObject, AsyncTask
from datetime import datetime


class TaskStatus(str, Enum):
    """Task status enum"""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class AnalysisMode(str, Enum):
    """Analysis mode enum"""
    FAST = "fast"  # Single flash model for everything
    QUALITY = "quality"  # Flash safety + Pro embeddings with thinking


class ProcessingPhase(str, Enum):
    """Processing phase enum"""
    SAFETY_CHECK = "safety_check"
    AI_ANALYSIS = "ai_analysis"
    BUILDING_KNOWLEDGE_GRAPH = "building_knowledge_graph"
    PROCESSING_URIS = "processing_uris"
    COMPLETE = "complete"


@dataclass
class TaskInfo:
    """Task information"""
    task_id: str
    object_id: int
    status: TaskStatus
    mode: AnalysisMode
    current_phase: Optional[ProcessingPhase] = None
    progress: int = 0
    created_at: str = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow().isoformat()

    def to_dict(self):
        """Convert to dict for API responses"""
        return {k: v.value if isinstance(v, Enum) else v for k, v in asdict(self).items()}


class AsyncPipelineManager:
    """
    Manages async AI analysis and knowledge graph building.

    This is a singleton that maintains a task queue and executes
    AI analysis and KG building in the background.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # Only track running tasks in memory (not persistent)
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self._initialized = True

        # Log file for debugging
        self.log_file = "/tmp/async_pipeline.log"
        self._log("AsyncPipelineManager initialized (database-backed)")

    def _log(self, message: str):
        """Write log message to file"""
        try:
            with open(self.log_file, "a") as f:
                timestamp = datetime.utcnow().isoformat()
                f.write(f"[{timestamp}] {message}\n")
                f.flush()
        except Exception as e:
            print(f"Failed to write log: {e}")

    def _save_task(self, task_info: TaskInfo, db: Session):
        """Save or update task in database"""
        try:
            # Normalize timestamp fields for SQLAlchemy (expect datetime objects)
            def _to_dt(val):
                if val is None:
                    return None
                if isinstance(val, datetime):
                    return val
                try:
                    return datetime.fromisoformat(val)
                except Exception:
                    return None

            created_dt = _to_dt(task_info.created_at) or datetime.utcnow()
            started_dt = _to_dt(task_info.started_at)
            completed_dt = _to_dt(task_info.completed_at)

            # Check if task already exists
            existing = db.query(AsyncTask).filter(AsyncTask.task_id == task_info.task_id).first()

            if existing:
                # Update existing task
                existing.status = task_info.status.value if isinstance(task_info.status, Enum) else task_info.status
                existing.mode = task_info.mode.value if isinstance(task_info.mode, Enum) else task_info.mode
                existing.current_phase = task_info.current_phase.value if isinstance(task_info.current_phase, Enum) else task_info.current_phase
                existing.progress = task_info.progress
                existing.started_at = started_dt
                existing.completed_at = completed_dt
                existing.error = task_info.error
                existing.result = json.dumps(task_info.result) if task_info.result else None
            else:
                # Create new task
                new_task = AsyncTask(
                    task_id=task_info.task_id,
                    object_id=task_info.object_id,
                    status=task_info.status.value if isinstance(task_info.status, Enum) else task_info.status,
                    mode=task_info.mode.value if isinstance(task_info.mode, Enum) else task_info.mode,
                    current_phase=task_info.current_phase.value if isinstance(task_info.current_phase, Enum) and task_info.current_phase else None,
                    progress=task_info.progress,
                    created_at=created_dt,
                    started_at=started_dt,
                    completed_at=completed_dt,
                    error=task_info.error,
                    result=json.dumps(task_info.result) if task_info.result else None
                )
                db.add(new_task)

            db.commit()
        except Exception as e:
            self._log(f"Error saving task {task_info.task_id}: {e}")
            db.rollback()
            raise

    def _load_task(self, task_id: str, db: Session) -> Optional[TaskInfo]:
        """Load task from database"""
        try:
            db_task = db.query(AsyncTask).filter(AsyncTask.task_id == task_id).first()
            if not db_task:
                return None

            return TaskInfo(
                task_id=db_task.task_id,
                object_id=db_task.object_id,
                status=TaskStatus(db_task.status),
                mode=AnalysisMode(db_task.mode),
                current_phase=ProcessingPhase(db_task.current_phase) if db_task.current_phase else None,
                progress=db_task.progress,
                created_at=db_task.created_at,
                started_at=db_task.started_at,
                completed_at=db_task.completed_at,
                error=db_task.error,
                result=json.loads(db_task.result) if db_task.result else None
            )
        except Exception as e:
            self._log(f"Error loading task {task_id}: {e}")
            return None

    async def start_task(
        self,
        object_id: int,
        mode: str = "quality",
        db: Session = None,
        ai_tasks_str: Optional[str] = None,
        vision_mode: Optional[str] = None,
        context_role: Optional[str] = None
    ) -> str:
        """
        Start a new async processing task.

        Args:
            object_id: Storage object ID to process
            mode: "fast" or "quality"
            db: Database session

        Returns:
            Task ID for status tracking
        """
        task_id = str(uuid.uuid4())

        # Validate mode
        try:
            analysis_mode = AnalysisMode(mode)
        except ValueError:
            analysis_mode = AnalysisMode.QUALITY

        # Create task info
        task_info = TaskInfo(
            task_id=task_id,
            object_id=object_id,
            status=TaskStatus.QUEUED,
            mode=analysis_mode,
            result={
                "config": {
                    "ai_tasks": ai_tasks_str,
                    "vision_mode": vision_mode or "auto",
                    "context_role": context_role,
                }
            }
        )

        # Save to database
        self._save_task(task_info, db)
        self._log(f"Created task {task_id} for object {object_id} in {mode} mode")

        # Start background processing
        task = asyncio.create_task(self._process_task(task_id, db))
        self.running_tasks[task_id] = task

        # Clean up when done
        task.add_done_callback(lambda t: self._cleanup_task(task_id))

        return task_id

    def _cleanup_task(self, task_id: str):
        """Remove task from running tasks when complete"""
        if task_id in self.running_tasks:
            del self.running_tasks[task_id]
            self._log(f"Cleaned up task {task_id}")

    async def _process_task(self, task_id: str, db: Session):
        """
        Process a task through the complete pipeline.

        This is the main async processing function that orchestrates
        the entire AI analysis and KG building process.
        """
        # Load task from database
        task_info = self._load_task(task_id, db)
        if not task_info:
            self._log(f"Task {task_id}: Not found in database")
            return

        try:
            # Mark as processing
            task_info.status = TaskStatus.PROCESSING
            task_info.started_at = datetime.utcnow().isoformat()
            task_info.progress = 5
            self._save_task(task_info, db)

            self._log(f"Task {task_id}: Starting processing for object {task_info.object_id}")

            # Get storage object from database
            storage_obj = db.query(StorageObject).filter(
                StorageObject.id == task_info.object_id
            ).first()

            if not storage_obj:
                raise ValueError(f"Storage object {task_info.object_id} not found")

            # Process based on mode
            if task_info.mode == AnalysisMode.FAST:
                result = await self._process_fast_mode(task_info, storage_obj, db)
            else:
                result = await self._process_quality_mode(task_info, storage_obj, db)

            # Mark as completed
            task_info.status = TaskStatus.COMPLETED
            task_info.completed_at = datetime.utcnow().isoformat()
            task_info.current_phase = ProcessingPhase.COMPLETE
            task_info.progress = 100
            task_info.result = result
            self._save_task(task_info, db)

            self._log(f"Task {task_id}: Completed successfully")

        except Exception as e:
            # Mark as failed
            task_info.status = TaskStatus.FAILED
            task_info.completed_at = datetime.utcnow().isoformat()
            task_info.error = str(e)
            task_info.progress = 0
            self._save_task(task_info, db)

            error_trace = traceback.format_exc()
            self._log(f"Task {task_id}: Failed with error: {e}\n{error_trace}")

    async def _process_fast_mode(
        self,
        task_info: TaskInfo,
        storage_obj: StorageObject,
        db: Session
    ) -> Dict[str, Any]:
        """
        Fast mode: Process with AI analysis first, then knowledge graph pipeline.

        Steps:
        1. Run AI analysis (analyze_content)
        2. Save results to database
        3. Build knowledge graph from AI results
        """
        self._log(f"Task {task_info.task_id}: Fast mode processing")

        # STEP 1: AI ANALYSIS
        task_info.current_phase = ProcessingPhase.AI_ANALYSIS
        task_info.progress = 10
        self._save_task(task_info, db)

        self._log(f"Task {task_info.task_id}: Running AI analysis")

        # Load file from storage
        from storage.service import generic_storage
        tenant_id = getattr(storage_obj, 'tenant_id', None)
        if not tenant_id:
            meta = getattr(storage_obj, 'metadata_json', None) or {}
            tenant_id = meta.get('tenant_id')
        tenant_id = tenant_id or 'arkturian'
        file_path = generic_storage.absolute_path_for_key(storage_obj.object_key, tenant_id)
        if not file_path.exists():
            raise FileNotFoundError(f"File missing from storage: {file_path}")

        with open(file_path, "rb") as f:
            data = f.read()

        # Run AI analysis (respect pipeline config if present)
        from ai_analysis.service import analyze_content
        cfg = (task_info.result or {}).get("config", {}) if task_info.result else {}
        analysis_result = await analyze_content(
            data,
            storage_obj.mime_type,
            context=None,
            vision_mode=cfg.get("vision_mode", "auto"),
            ai_tasks_str=cfg.get("ai_tasks"),
            context_role=cfg.get("context_role")
        )

        # Ensure object is attached to this session
        storage_obj = db.merge(storage_obj)

        # Save AI analysis results to database (like sync route does)
        storage_obj.ai_category = analysis_result.get("category")
        storage_obj.ai_danger_potential = analysis_result.get("danger_potential")

        if "safety_info" in analysis_result:
            storage_obj.safety_info = analysis_result.get("safety_info")
            safety_info = analysis_result.get("safety_info", {})
            storage_obj.ai_safety_rating = "safe" if safety_info.get("isSafe", True) else "unsafe"

        storage_obj.ai_title = analysis_result.get("ai_title")
        storage_obj.ai_subtitle = analysis_result.get("ai_subtitle")
        storage_obj.ai_tags = analysis_result.get("ai_tags", [])
        storage_obj.ai_collections = analysis_result.get("ai_collections", [])

        # Store extracted tags and embedding info for Knowledge Graph
        if "extracted_tags" in analysis_result or "embedding_info" in analysis_result:
            context_meta = storage_obj.ai_context_metadata.copy() if storage_obj.ai_context_metadata else {}
            context_meta["extracted_tags"] = analysis_result.get("extracted_tags", {})
            context_meta["embedding_info"] = analysis_result.get("embedding_info", {})
            context_meta["mode"] = analysis_result.get("mode", "")
            context_meta["prompt"] = analysis_result.get("prompt", "")
            context_meta["response"] = analysis_result.get("ai_response", "")
            storage_obj.ai_context_metadata = context_meta

        db.commit()
        db.refresh(storage_obj)

        self._log(f"Task {task_info.task_id}: AI analysis complete, saved to database")

        # STEP 2: KNOWLEDGE GRAPH
        task_info.current_phase = ProcessingPhase.BUILDING_KNOWLEDGE_GRAPH
        task_info.progress = 50
        self._save_task(task_info, db)

        from knowledge_graph.pipeline import kg_pipeline

        self._log(f"Task {task_info.task_id}: Building knowledge graph")
        kg_result = await kg_pipeline.process_storage_object(storage_obj, db)

        task_info.progress = 90
        self._save_task(task_info, db)

        # Get stats from database
        external_objects = db.query(StorageObject).filter(
            StorageObject.link_id == str(storage_obj.id)
        ).count() if kg_result else 0

        return {
            "mode": "fast",
            "kg_result": kg_result,
            "embeddings_created": kg_result.get("embeddings_created", 0) if kg_result and isinstance(kg_result, dict) else 0,
            "external_objects_created": external_objects
        }

    async def _process_quality_mode(
        self,
        task_info: TaskInfo,
        storage_obj: StorageObject,
        db: Session
    ) -> Dict[str, Any]:
        """
        Quality mode: Process with AI analysis first, then knowledge graph pipeline.

        Steps:
        1. Run AI analysis (analyze_content)
        2. Save results to database
        3. Build knowledge graph from AI results

        In the future, this could use Pro model with thinking
        by passing parameters to analyze_content.
        """
        self._log(f"Task {task_info.task_id}: Quality mode processing")

        # STEP 1: AI ANALYSIS
        task_info.current_phase = ProcessingPhase.AI_ANALYSIS
        task_info.progress = 10
        self._save_task(task_info, db)

        self._log(f"Task {task_info.task_id}: Running AI analysis")

        # Load file from storage
        from storage.service import generic_storage
        tenant_id = getattr(storage_obj, 'tenant_id', None)
        if not tenant_id:
            meta = getattr(storage_obj, 'metadata_json', None) or {}
            tenant_id = meta.get('tenant_id')
        tenant_id = tenant_id or 'arkturian'
        file_path = generic_storage.absolute_path_for_key(storage_obj.object_key, tenant_id)
        if not file_path.exists():
            raise FileNotFoundError(f"File missing from storage: {file_path}")

        with open(file_path, "rb") as f:
            data = f.read()

        # Run AI analysis (respect pipeline config if present)
        from ai_analysis.service import analyze_content
        cfg = (task_info.result or {}).get("config", {}) if task_info.result else {}
        analysis_result = await analyze_content(
            data,
            storage_obj.mime_type,
            context=None,
            vision_mode=cfg.get("vision_mode", "auto"),
            ai_tasks_str=cfg.get("ai_tasks"),
            context_role=cfg.get("context_role")
        )

        # Ensure object is attached to this session
        storage_obj = db.merge(storage_obj)

        # Save AI analysis results to database (like sync route does)
        storage_obj.ai_category = analysis_result.get("category")
        storage_obj.ai_danger_potential = analysis_result.get("danger_potential")

        if "safety_info" in analysis_result:
            storage_obj.safety_info = analysis_result.get("safety_info")
            safety_info = analysis_result.get("safety_info", {})
            storage_obj.ai_safety_rating = "safe" if safety_info.get("isSafe", True) else "unsafe"

        storage_obj.ai_title = analysis_result.get("ai_title")
        storage_obj.ai_subtitle = analysis_result.get("ai_subtitle")
        storage_obj.ai_tags = analysis_result.get("ai_tags", [])
        storage_obj.ai_collections = analysis_result.get("ai_collections", [])

        # Store extracted tags and embedding info for Knowledge Graph
        if "extracted_tags" in analysis_result or "embedding_info" in analysis_result:
            context_meta = storage_obj.ai_context_metadata.copy() if storage_obj.ai_context_metadata else {}
            context_meta["extracted_tags"] = analysis_result.get("extracted_tags", {})
            context_meta["embedding_info"] = analysis_result.get("embedding_info", {})
            context_meta["mode"] = analysis_result.get("mode", "")
            context_meta["prompt"] = analysis_result.get("prompt", "")
            context_meta["response"] = analysis_result.get("ai_response", "")
            storage_obj.ai_context_metadata = context_meta

        db.commit()
        db.refresh(storage_obj)

        self._log(f"Task {task_info.task_id}: AI analysis complete, saved to database")

        task_info.progress = 40
        self._save_task(task_info, db)

        # STEP 2: KNOWLEDGE GRAPH
        task_info.current_phase = ProcessingPhase.BUILDING_KNOWLEDGE_GRAPH
        task_info.progress = 50
        self._save_task(task_info, db)

        from knowledge_graph.pipeline import kg_pipeline

        self._log(f"Task {task_info.task_id}: Building knowledge graph")
        kg_result = await kg_pipeline.process_storage_object(storage_obj, db)

        task_info.progress = 90
        self._save_task(task_info, db)

        # Get stats from database
        external_objects = db.query(StorageObject).filter(
            StorageObject.link_id == str(storage_obj.id)
        ).count() if kg_result else 0

        return {
            "mode": "quality",
            "kg_result": kg_result,
            "embeddings_created": kg_result.get("embeddings_created", 0) if kg_result and isinstance(kg_result, dict) else 0,
            "external_objects_created": external_objects
        }

    def get_task_status(self, task_id: str, db: Session) -> Optional[Dict[str, Any]]:
        """
        Get status of a task.

        Args:
            task_id: Task ID to check
            db: Database session

        Returns:
            Task info dict or None if not found
        """
        task_info = self._load_task(task_id, db)
        if not task_info:
            return None

        return task_info.to_dict()

    def get_all_tasks(self, db: Session, limit: int = 100, object_id: int = None) -> list:
        """Get all tasks (most recent first), optionally filtered by object_id"""
        try:
            query = db.query(AsyncTask)

            # Filter by object_id if provided
            if object_id is not None:
                query = query.filter(AsyncTask.object_id == object_id)

            db_tasks = query.order_by(
                AsyncTask.created_at.desc()
            ).limit(limit).all()

            return [
                TaskInfo(
                    task_id=t.task_id,
                    object_id=t.object_id,
                    status=TaskStatus(t.status),
                    mode=AnalysisMode(t.mode),
                    current_phase=ProcessingPhase(t.current_phase) if t.current_phase else None,
                    progress=t.progress,
                    created_at=t.created_at,
                    started_at=t.started_at,
                    completed_at=t.completed_at,
                    error=t.error,
                    result=json.loads(t.result) if t.result else None
                ).to_dict()
                for t in db_tasks
            ]
        except Exception as e:
            self._log(f"Error loading tasks: {e}")
            return []


# Global singleton instance
pipeline_manager = AsyncPipelineManager()
