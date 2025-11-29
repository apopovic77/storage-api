"""
Celery Application for Storage API

Centralized task queue for:
- AI Analysis (Vision, Safety, Embeddings)
- Video Transcoding
- Background Processing
- Scheduled Tasks
"""

from celery import Celery
from config import settings
import os

# Celery Application Singleton
app = Celery(
    'storage_api',
    broker=f'redis://localhost:6379/0',
    backend=f'redis://localhost:6379/1',
    include=[
        'tasks.ai_analysis',
        # Future: 'tasks.transcoding',
        # Future: 'tasks.notifications',
    ]
)

# Celery Configuration
app.conf.update(
    # Task Settings
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,

    # Worker Settings
    worker_prefetch_multiplier=1,  # Only fetch 1 task at a time (for long-running AI tasks)
    worker_max_tasks_per_child=50,  # Restart worker after 50 tasks (memory cleanup)

    # Retry Settings
    task_acks_late=True,  # Acknowledge task AFTER completion (ensures retry on crash)
    task_reject_on_worker_lost=True,  # Re-queue task if worker dies

    # Result Backend
    result_expires=3600,  # Keep results for 1 hour
    result_persistent=True,  # Persist results to Redis

    # Monitoring
    task_track_started=True,  # Track when task starts
    task_send_sent_event=True,  # Send events for monitoring

    # Beat Scheduler (for future scheduled tasks)
    beat_scheduler='celery.beat:PersistentScheduler',
    beat_schedule_filename='/var/lib/storage-api/celerybeat-schedule',
)

# Task Routes (Priority Queues)
app.conf.task_routes = {
    'tasks.ai_analysis.process_image_analysis': {'queue': 'ai_analysis'},
    'tasks.ai_analysis.process_video_analysis': {'queue': 'ai_analysis'},
    'tasks.ai_analysis.process_text_analysis': {'queue': 'ai_analysis'},
    'tasks.ai_analysis.generate_embedding': {'queue': 'embeddings'},
}

# Task Priority
app.conf.task_default_priority = 5  # 0 (highest) - 9 (lowest)

if __name__ == '__main__':
    app.start()
