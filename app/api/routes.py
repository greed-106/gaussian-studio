"""
FastAPI routes for the Gaussian Reconstruction Backend.
"""
import shutil
import requests
from pathlib import Path
from typing import List
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.task_queue import Task, TaskQueueManager
from app.database import TaskDatabase
from app.logger import WorkerLogger


router = APIRouter()

# Global instances (will be set by main app)
queue_manager: TaskQueueManager = None
db: TaskDatabase = None
config: dict = None


def init_routes(qm: TaskQueueManager, database: TaskDatabase, cfg: dict):
    """Initialize route dependencies."""
    global queue_manager, db, config
    queue_manager = qm
    db = database
    config = cfg


# Request/Response models
class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    exists: bool
    completed_at: str | None = None  # ISO format timestamp for finished/failed tasks
    intrinsic_matrix: List[float] | None = None  # Camera intrinsic matrix (3x3 flattened)
    extrinsic_matrix: List[float] | None = None  # Camera extrinsic matrix (4x4 flattened)


class BatchStatusRequest(BaseModel):
    task_ids: List[str]


class BatchStatusResponse(BaseModel):
    results: List[TaskStatusResponse]


class UploadResponse(BaseModel):
    message: str


class QueueStatsResponse(BaseModel):
    total: int
    waiting: int
    preprocessing: int
    sfm: int
    reconstruction: int
    compress: int


class UploadToStorageRequest(BaseModel):
    upload_url: str


class UploadToStorageResponse(BaseModel):
    message: str
    file_size: int


class DownloadFromStorageRequest(BaseModel):
    task_id: str
    video_url: str


class DownloadFromStorageResponse(BaseModel):
    task_id: str
    message: str
    file_size: int


# Supported video formats
SUPPORTED_FORMATS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}


@router.post("/upload", response_model=UploadResponse)
async def upload_video(task_id: str = Form(...), file: UploadFile = File(...)):
    """
    Upload a video file and create a reconstruction task.
    
    Args:
        task_id: Client-provided task ID (64-bit snowflake ID as string, sent as form data)
        file: Video file
        
    Returns:
        Success message
    """
    # Validate task_id format (should be numeric string)
    if not task_id or not task_id.isdigit():
        raise HTTPException(
            status_code=400,
            detail="Invalid task_id: must be a numeric string (e.g., snowflake ID)"
        )
    
    # Check if task_id already exists
    existing_task = queue_manager.get_task(task_id)
    if existing_task:
        raise HTTPException(
            status_code=409,
            detail=f"Task ID {task_id} already exists"
        )
    
    # Check in database history
    db_record = await db.get_task_history(task_id)
    if db_record:
        raise HTTPException(
            status_code=409,
            detail=f"Task ID {task_id} already exists in history"
        )
    
    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format. Supported: {', '.join(SUPPORTED_FORMATS)}"
        )
    
    # Create work directory
    work_root = Path(config["STORAGE"]["WORK_DIRECTORY"])
    work_dir = work_root / task_id
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # Save video file
    video_path = work_dir / f"video{file_ext}"
    try:
        with open(video_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        # Cleanup on failure
        shutil.rmtree(work_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Failed to save video: {str(e)}")
    
    # Create task and add to queue
    task = Task(
        task_id=task_id,
        video_path=str(video_path),
        work_dir=str(work_dir)
    )
    queue_manager.add_task(task)
    
    return UploadResponse(
        message="Task created successfully"
    )


@router.get("/status/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """
    Get status of a single task.
    
    Args:
        task_id: Task ID
        
    Returns:
        Task status information (includes completed_at and camera metadata for finished tasks)
    """
    # Check in queue first
    task_dict = queue_manager.get_task(task_id)
    if task_dict:
        return TaskStatusResponse(
            task_id=task_id,
            status=task_dict["status"],
            exists=True,
            completed_at=None,
            intrinsic_matrix=None,
            extrinsic_matrix=None
        )
    
    # Check in database
    db_record = await db.get_task_history(task_id)
    if db_record:
        # Get camera metadata if task is finished
        metadata = None
        if db_record["status"] == "finish":
            metadata = await db.get_metadata(task_id)
        
        return TaskStatusResponse(
            task_id=task_id,
            status=db_record["status"],
            exists=True,
            completed_at=db_record.get("completed_at"),
            intrinsic_matrix=metadata.get("intrinsic_matrix") if metadata else None,
            extrinsic_matrix=metadata.get("extrinsic_matrix") if metadata else None
        )
    
    # Task not found
    return TaskStatusResponse(
        task_id=task_id,
        status="not_found",
        exists=False,
        completed_at=None,
        intrinsic_matrix=None,
        extrinsic_matrix=None
    )


@router.post("/status/batch", response_model=BatchStatusResponse)
async def get_batch_status(request: BatchStatusRequest):
    """
    Get status of multiple tasks.
    
    Args:
        request: List of task IDs
        
    Returns:
        List of task status information (includes completed_at and camera metadata for finished tasks)
    """
    results = []
    
    for task_id in request.task_ids:
        # Check in queue
        task_dict = queue_manager.get_task(task_id)
        if task_dict:
            results.append(TaskStatusResponse(
                task_id=task_id,
                status=task_dict["status"],
                exists=True,
                completed_at=None,
                intrinsic_matrix=None,
                extrinsic_matrix=None
            ))
            continue
        
        # Check in database
        db_record = await db.get_task_history(task_id)
        if db_record:
            # Get camera metadata if task is finished
            metadata = None
            if db_record["status"] == "finish":
                metadata = await db.get_metadata(task_id)
            
            results.append(TaskStatusResponse(
                task_id=task_id,
                status=db_record["status"],
                exists=True,
                completed_at=db_record.get("completed_at"),
                intrinsic_matrix=metadata.get("intrinsic_matrix") if metadata else None,
                extrinsic_matrix=metadata.get("extrinsic_matrix") if metadata else None
            ))
            continue
        
        # Not found
        results.append(TaskStatusResponse(
            task_id=task_id,
            status="not_found",
            exists=False,
            completed_at=None,
            intrinsic_matrix=None,
            extrinsic_matrix=None
        ))
    
    return BatchStatusResponse(results=results)


@router.get("/queue/stats", response_model=QueueStatsResponse)
async def get_queue_stats():
    """
    Get statistics of tasks in the queue.
    
    Returns:
        Number of tasks in each status
    """
    stats = queue_manager.get_queue_stats()
    return QueueStatsResponse(**stats)


@router.get("/tasks/{task_id}/assets")
async def get_task_assets(task_id: str):
    """
    Download the reconstructed SOG file for a completed task.
    
    Args:
        task_id: Task ID
        
    Returns:
        SOG file for download
    """
    # Check if task exists and is completed
    task_dict = queue_manager.get_task(task_id)
    if task_dict:
        raise HTTPException(
            status_code=400,
            detail=f"Task is still in progress (status: {task_dict['status']})"
        )
    
    # Check in database
    db_record = await db.get_task_history(task_id)
    if not db_record:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if db_record["status"] != "finish":
        raise HTTPException(
            status_code=400,
            detail=f"Task did not complete successfully (status: {db_record['status']})"
        )
    
    # Find the SOG file
    work_root = Path(config["STORAGE"]["WORK_DIRECTORY"])
    work_dir = work_root / task_id
    
    if not work_dir.exists():
        raise HTTPException(status_code=404, detail="Task directory not found")
    
    # Find the latest SOG file
    sog_files = sorted(work_dir.glob("splat_*.sog"))
    if not sog_files:
        raise HTTPException(status_code=404, detail="SOG file not found")
    
    sog_file = sog_files[-1]  # Get the latest one
    
    return FileResponse(
        path=str(sog_file),
        media_type="application/octet-stream",
        filename=sog_file.name
    )


@router.get("/tasks/{task_id}/metadata")
async def get_task_metadata(task_id: str):
    """
    Get camera metadata (intrinsic and extrinsic parameters) for a task.
    
    Args:
        task_id: Task ID
        
    Returns:
        Camera metadata including intrinsic and extrinsic matrices
    """
    # Check if task exists
    task_dict = queue_manager.get_task(task_id)
    if task_dict:
        # Task is still in progress
        if task_dict["status"] in ["waiting", "preprocessing"]:
            raise HTTPException(
                status_code=400,
                detail=f"Metadata not available yet (status: {task_dict['status']})"
            )
    
    # Get metadata from database
    metadata = await db.get_metadata(task_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="Metadata not found")
    
    return metadata


@router.post("/tasks/{task_id}/upload-to-oss", response_model=UploadToStorageResponse)
async def upload_task_to_storage(task_id: str, request: UploadToStorageRequest):
    """
    Upload completed task's SOG file to object storage using a pre-signed URL.
    
    Args:
        task_id: Task ID from URL path
        request: Contains upload_url (pre-signed URL from client)
        
    Returns:
        Success message and file size
    """
    upload_url = request.upload_url
    
    # Check if task exists and is completed
    task_dict = queue_manager.get_task(task_id)
    if task_dict:
        raise HTTPException(
            status_code=400,
            detail=f"Task is still in progress (status: {task_dict['status']})"
        )
    
    # Check in database
    db_record = await db.get_task_history(task_id)
    if not db_record:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if db_record["status"] != "finish":
        raise HTTPException(
            status_code=400,
            detail=f"Task did not complete successfully (status: {db_record['status']})"
        )
    
    # Find the SOG file
    work_root = Path(config["STORAGE"]["WORK_DIRECTORY"])
    work_dir = work_root / task_id
    
    if not work_dir.exists():
        raise HTTPException(status_code=404, detail="Task directory not found")
    
    # Find the latest SOG file
    sog_files = sorted(work_dir.glob("splat_*.sog"))
    if not sog_files:
        raise HTTPException(status_code=404, detail="SOG file not found")
    
    sog_file = sog_files[-1]  # Get the latest one
    
    # Upload file to object storage
    try:
        with open(sog_file, "rb") as f:
            file_data = f.read()
            file_size = len(file_data)
            
            # Upload using PUT request to pre-signed URL
            response = requests.put(
                upload_url,
                data=file_data,
                headers={"Content-Type": "application/octet-stream"},
                timeout=300  # 5 minutes timeout
            )
            
            if response.status_code not in [200, 201, 204]:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to upload to storage: HTTP {response.status_code}"
                )
        
        # Upload successful, clean up task files
        try:
            shutil.rmtree(work_dir)
        except Exception as cleanup_error:
            # Log the cleanup error but don't fail the request since upload was successful
            WorkerLogger.log("Main", f"Warning: Failed to cleanup task directory {work_dir}: {cleanup_error}")
        
        return UploadToStorageResponse(
            message="File uploaded successfully and local files cleaned up",
            file_size=file_size
        )
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload to storage: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error reading or uploading file: {str(e)}"
        )

@router.post("/tasks/from-oss", response_model=DownloadFromStorageResponse)
async def create_task_from_storage(request: DownloadFromStorageRequest):
    """
    Download video from object storage and create a reconstruction task.
    
    Args:
        request: Contains task_id and video_url (direct download URL)
        
    Returns:
        Task ID, success message and downloaded file size
    """
    task_id = request.task_id
    video_url = request.video_url
    
    # Validate task_id format (should be numeric string)
    if not task_id or not task_id.isdigit():
        raise HTTPException(
            status_code=400,
            detail="Invalid task_id: must be a numeric string (e.g., snowflake ID)"
        )
    
    # Check if task_id already exists
    existing_task = queue_manager.get_task(task_id)
    if existing_task:
        raise HTTPException(
            status_code=409,
            detail=f"Task ID {task_id} already exists"
        )
    
    # Check in database history
    db_record = await db.get_task_history(task_id)
    if db_record:
        raise HTTPException(
            status_code=409,
            detail=f"Task ID {task_id} already exists in history"
        )
    
    # Create work directory
    work_root = Path(config["STORAGE"]["WORK_DIRECTORY"])
    work_dir = work_root / task_id
    work_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Download video from URL
        response = requests.get(
            video_url,
            stream=True,
            timeout=30,  # 30 seconds timeout for connection
            headers={"User-Agent": "Gaussian-Studio-Backend/1.0"}
        )
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to download video: HTTP {response.status_code}"
            )
        
        # Extract file extension from URL
        from urllib.parse import urlparse
        parsed_url = urlparse(video_url)
        url_path = parsed_url.path.lower()
        
        file_ext = None
        for ext in SUPPORTED_FORMATS:
            if url_path.endswith(ext):
                file_ext = ext
                break
        
        # Default to .mp4 if no extension detected
        if not file_ext:
            file_ext = ".mp4"
        
        # Validate file extension
        if file_ext not in SUPPORTED_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported video format: {file_ext}. Supported: {', '.join(SUPPORTED_FORMATS)}"
            )
        
        # Save video file
        video_path = work_dir / f"video{file_ext}"
        file_size = 0
        
        with open(video_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    file_size += len(chunk)
        
        # Create task and add to queue
        task = Task(
            task_id=task_id,
            video_path=str(video_path),
            work_dir=str(work_dir)
        )
        queue_manager.add_task(task)
        
        return DownloadFromStorageResponse(
            task_id=task_id,
            message="Video downloaded and task created successfully",
            file_size=file_size
        )
        
    except requests.exceptions.RequestException as e:
        # Cleanup on failure
        shutil.rmtree(work_dir, ignore_errors=True)
        raise HTTPException(
            status_code=400,
            detail=f"Failed to download video: {str(e)}"
        )
    except Exception as e:
        # Cleanup on failure
        shutil.rmtree(work_dir, ignore_errors=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error downloading or saving video: {str(e)}"
        )