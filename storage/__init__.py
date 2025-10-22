"""
Storage Module

Generic file storage system with AI-powered semantic analysis.

Components:
- service: GenericStorageService for file operations
- domain: Business logic for saving/updating storage objects
- routes: API endpoints for upload, download, bulk operations
- models: Database models (StorageObject)
"""

from .service import GenericStorageService, generic_storage, bulk_delete_objects

__all__ = [
    "GenericStorageService",
    "generic_storage",
    "bulk_delete_objects",
]
