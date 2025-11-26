from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import Optional, List
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
    tenant_id: str

class CollectionInfo(BaseModel):
    id: str
    name: Optional[str]
    item_count: int
    owner_email: Optional[str] = None

class CollectionSearchResult(BaseModel):
    collection_id: str
    item_count: int
    owner_email: Optional[str] = None
    owner_display_name: Optional[str] = None
    tenant_id: str

@router.get("/users-with-collections", response_model=List[UserWithCollections])
def get_users_with_collections(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get all users who have collections in storage"""
    if current_user.trust_level != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    # Query users who have storage objects with collection_id, grouped by tenant
    users_with_collections = db.query(
        User.email,
        User.display_name,
        StorageObject.tenant_id,
        func.count(func.distinct(StorageObject.collection_id)).label('collection_count')
    ).join(
        StorageObject, User.id == StorageObject.owner_user_id
    ).filter(
        StorageObject.collection_id.isnot(None),
        StorageObject.collection_id != ""
    ).group_by(
        User.id, User.email, User.display_name, StorageObject.tenant_id
    ).having(
        func.count(func.distinct(StorageObject.collection_id)) > 0
    ).order_by(
        StorageObject.tenant_id, User.email
    ).all()

    return [
        UserWithCollections(
            email=user.email,
            display_name=user.display_name,
            collection_count=user.collection_count,
            tenant_id=user.tenant_id
        )
        for user in users_with_collections
    ]

@router.get("/collections", response_model=List[CollectionInfo])
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


@router.get("/collections/search", response_model=List[CollectionSearchResult])
def search_collections(
    query: str = Query(..., min_length=1, description="Search query for collection name"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Search for collections across all tenants and users by collection name"""
    if current_user.trust_level != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    # Search for collections with name matching the query (case-insensitive)
    # Join with User to get owner information
    results = db.query(
        StorageObject.collection_id,
        StorageObject.tenant_id,
        func.count(StorageObject.id).label('item_count'),
        User.email.label('owner_email'),
        User.display_name.label('owner_display_name')
    ).outerjoin(
        User, StorageObject.owner_user_id == User.id
    ).filter(
        StorageObject.collection_id.isnot(None),
        StorageObject.collection_id != "",
        StorageObject.collection_id.ilike(f"%{query}%")  # Case-insensitive partial match
    ).group_by(
        StorageObject.collection_id,
        StorageObject.tenant_id,
        User.email,
        User.display_name
    ).order_by(
        StorageObject.tenant_id,
        StorageObject.collection_id
    ).all()

    return [
        CollectionSearchResult(
            collection_id=result.collection_id,
            item_count=result.item_count,
            owner_email=result.owner_email,
            owner_display_name=result.owner_display_name,
            tenant_id=result.tenant_id
        )
        for result in results
    ]


class CleanupResponse(BaseModel):
    deleted_count: int
    message: str


class CleanupByCollectionRequest(BaseModel):
    collection_id: str


class CleanupByUserRequest(BaseModel):
    user_email: str


class CleanupByAgeRequest(BaseModel):
    days: int


@router.post("/cleanup/by-collection", response_model=CleanupResponse)
def cleanup_by_collection(
    request: CleanupByCollectionRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete all storage objects within a specific collection"""
    if current_user.trust_level != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    return purge_objects_by_collection(db, request.collection_id)


@router.post("/cleanup/by-user", response_model=CleanupResponse)
def cleanup_by_user(
    request: CleanupByUserRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete all storage objects owned by a specific user"""
    if current_user.trust_level != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    try:
        return purge_objects_by_user_email(db, request.user_email)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/cleanup/by-age", response_model=CleanupResponse)
def cleanup_by_age(
    request: CleanupByAgeRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete storage objects older than specified days"""
    if current_user.trust_level != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    return cleanup_objects_older_than(db, request.days)


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
        # Delete file and all associated assets (thumbnails, webview, HLS)
        # The generic_storage.delete() method calls _delete_physical_assets()
        # which handles deletion of all derived files
        try:
            generic_storage.delete(obj.object_key, obj.tenant_id)
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
        # Delete file and all associated assets (thumbnails, webview, HLS)
        # The generic_storage.delete() method calls _delete_physical_assets()
        # which handles deletion of all derived files
        try:
            generic_storage.delete(obj.object_key, obj.tenant_id)
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
        # Delete file and all associated assets (thumbnails, webview, HLS)
        # The generic_storage.delete() method calls _delete_physical_assets()
        # which handles deletion of all derived files
        try:
            generic_storage.delete(obj.object_key, obj.tenant_id)
        except Exception as e:
            print(f"Error deleting file {obj.object_key}: {e}")

        # Delete database record
        db.delete(obj)
    
    db.commit()
    
    return CleanupResponse(
        deleted_count=deleted_count,
        message=f"Deleted {deleted_count} objects from collection {collection_id}"
    )