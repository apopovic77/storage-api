from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import Optional, List
import shutil
from pathlib import Path
from database import get_db
from auth import get_current_user
from models import StorageObject, User
from pydantic import BaseModel
from storage.service import generic_storage

router = APIRouter()

class UserWithCollections(BaseModel):
    email: str
    display_name: str
    collection_count: int

class CollectionInfo(BaseModel):
    id: str
    name: Optional[str]
    item_count: int
    owner_email: Optional[str] = None

@router.get("/admin/users-with-collections", response_model=List[UserWithCollections])
def get_users_with_collections(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get all users who have collections in storage"""
    if current_user.trust_level != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    # Query users who have storage objects with collection_id
    users_with_collections = db.query(
        User.email,
        User.display_name,
        func.count(func.distinct(StorageObject.collection_id)).label('collection_count')
    ).join(
        StorageObject, User.id == StorageObject.owner_user_id
    ).filter(
        StorageObject.collection_id.isnot(None),
        StorageObject.collection_id != ""
    ).group_by(
        User.id, User.email, User.display_name
    ).having(
        func.count(func.distinct(StorageObject.collection_id)) > 0
    ).order_by(
        User.email
    ).all()
    
    return [
        UserWithCollections(
            email=user.email,
            display_name=user.display_name,
            collection_count=user.collection_count
        )
        for user in users_with_collections
    ]

@router.get("/admin/collections", response_model=List[CollectionInfo])
def get_collections_for_user(
    user_email: Optional[str] = Query(None),
    public_only: bool = Query(False),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get collections for a specific user or public collections"""
    if current_user.trust_level != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    if public_only:
        # Get collections with no owner (public)
        collections_query = db.query(
            StorageObject.collection_id,
            func.count(StorageObject.id).label('item_count')
        ).filter(
            StorageObject.collection_id.isnot(None),
            StorageObject.collection_id != "",
            StorageObject.owner_user_id.is_(None)
        ).group_by(
            StorageObject.collection_id
        )
    elif user_email:
        # Find the user
        user = db.query(User).filter(User.email == user_email).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Get collections for this user
        collections_query = db.query(
            StorageObject.collection_id,
            func.count(StorageObject.id).label('item_count')
        ).filter(
            StorageObject.collection_id.isnot(None),
            StorageObject.collection_id != "",
            StorageObject.owner_user_id == user.id
        ).group_by(
            StorageObject.collection_id
        )
    else:
        raise HTTPException(status_code=400, detail="Either user_email or public_only must be specified")
    
    collections = collections_query.order_by(StorageObject.collection_id).all()
    
    return [
        CollectionInfo(
            id=collection.collection_id,
            name=collection.collection_id,  # Using ID as name since we don't store names separately
            item_count=collection.item_count,
            owner_email=user_email if not public_only else None
        )
        for collection in collections
    ]


class CleanupResponse(BaseModel):
    deleted_count: int
    message: str


def cleanup_objects_older_than(db: Session, days: int) -> CleanupResponse:
    """Delete storage objects older than specified days"""
    from datetime import datetime, timedelta
    
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    
    # Get objects to delete
    objects_to_delete = db.query(StorageObject).filter(
        StorageObject.created_at < cutoff_date
    ).all()
    
    deleted_count = len(objects_to_delete)
    
    # Delete files and database records
    for obj in objects_to_delete:
        try:
            # Delete HLS directory if exists
            basename = Path(obj.object_key).stem
            hls_dir_path = generic_storage.absolute_path_for_key(obj.object_key).parent / basename
            if hls_dir_path.is_dir():
                shutil.rmtree(hls_dir_path)
        except Exception as e:
            print(f"Error deleting HLS directory for {obj.object_key}: {e}")
        
        # Delete file
        try:
            generic_storage.delete(obj.object_key)
        except Exception as e:
            print(f"Error deleting file {obj.object_key}: {e}")
        
        # Delete database record
        db.delete(obj)
    
    db.commit()
    
    return CleanupResponse(
        deleted_count=deleted_count,
        message=f"Deleted {deleted_count} objects older than {days} days"
    )


def purge_objects_by_user_email(db: Session, email: str) -> CleanupResponse:
    """Delete all storage objects owned by a specific user email"""
    # Find the user
    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise ValueError(f"User with email {email} not found")
    
    # Get objects to delete
    objects_to_delete = db.query(StorageObject).filter(
        StorageObject.owner_user_id == user.id
    ).all()
    
    deleted_count = len(objects_to_delete)
    
    # Delete files and database records
    for obj in objects_to_delete:
        try:
            # Delete HLS directory if exists
            basename = Path(obj.object_key).stem
            hls_dir_path = generic_storage.absolute_path_for_key(obj.object_key).parent / basename
            if hls_dir_path.is_dir():
                shutil.rmtree(hls_dir_path)
        except Exception as e:
            print(f"Error deleting HLS directory for {obj.object_key}: {e}")
        
        # Delete file
        try:
            generic_storage.delete(obj.object_key)
        except Exception as e:
            print(f"Error deleting file {obj.object_key}: {e}")
        
        # Delete database record
        db.delete(obj)
    
    db.commit()
    
    return CleanupResponse(
        deleted_count=deleted_count,
        message=f"Deleted {deleted_count} objects from user {email}"
    )


def purge_objects_by_collection(db: Session, collection_id: str) -> CleanupResponse:
    """Delete all storage objects within a specific collection"""
    # Get objects to delete
    objects_to_delete = db.query(StorageObject).filter(
        StorageObject.collection_id == collection_id
    ).all()
    
    deleted_count = len(objects_to_delete)
    
    # Delete files and database records
    for obj in objects_to_delete:
        try:
            # Delete HLS directory if exists
            basename = Path(obj.object_key).stem
            hls_dir_path = generic_storage.absolute_path_for_key(obj.object_key).parent / basename
            if hls_dir_path.is_dir():
                shutil.rmtree(hls_dir_path)
        except Exception as e:
            print(f"Error deleting HLS directory for {obj.object_key}: {e}")
        
        # Delete file
        try:
            generic_storage.delete(obj.object_key)
        except Exception as e:
            print(f"Error deleting file {obj.object_key}: {e}")
        
        # Delete database record
        db.delete(obj)
    
    db.commit()
    
    return CleanupResponse(
        deleted_count=deleted_count,
        message=f"Deleted {deleted_count} objects from collection {collection_id}"
    )