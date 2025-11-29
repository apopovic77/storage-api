"""
Task Registry for Storage API Celery Tasks
"""

from .ai_analysis import (
    process_image_analysis,
    process_video_analysis,
    process_text_analysis,
    generate_embedding,
)

__all__ = [
    'process_image_analysis',
    'process_video_analysis',
    'process_text_analysis',
    'generate_embedding',
]
