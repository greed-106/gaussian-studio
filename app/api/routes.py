"""
FastAPI routes for the Gaussian Reconstruction Backend.
"""
import uuid
import shutil
from pathlib import Path
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

from app.task_queue import Task, TaskQueueManager
from app.database import TaskDatabase


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


class BatchStatusRequest(BaseModel):
    task_ids: List[str]


class BatchStatusResponse(BaseModel):
    results: List[TaskStatusResponse]


class UploadResponse(BaseModel):
    task_id: str
    message: str


class QueueStatsResponse(BaseModel):
    total: int
    waiting: int
    preprocessing: int
    sfm: int
    reconstruction: int
    compress: int


# Supported video formats
SUPPORTED_FORMATS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}


@router.post("/upload", response_model=UploadResponse)
async def upload_video(file: UploadFile = File(...)):
    """
    Upload a video file and create a reconstruction task.
    
    Args:
        file: Video file
        
    Returns:
        Task ID and success message
    """
    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format. Supported: {', '.join(SUPPORTED_FORMATS)}"
        )
    
    # Generate task ID
    task_id = str(uuid.uuid4())
    
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
        task_id=task_id,
        message="Task created successfully"
    )


@router.get("/status/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """
    Get status of a single task.
    
    Args:
        task_id: Task ID
        
    Returns:
        Task status information
    """
    # Check in queue first
    task_dict = queue_manager.get_task(task_id)
    if task_dict:
        return TaskStatusResponse(
            task_id=task_id,
            status=task_dict["status"],
            exists=True
        )
    
    # Check in database
    db_record = await db.get_task_history(task_id)
    if db_record:
        return TaskStatusResponse(
            task_id=task_id,
            status=db_record["status"],
            exists=True
        )
    
    # Task not found
    return TaskStatusResponse(
        task_id=task_id,
        status="not_found",
        exists=False
    )


@router.post("/status/batch", response_model=BatchStatusResponse)
async def get_batch_status(request: BatchStatusRequest):
    """
    Get status of multiple tasks.
    
    Args:
        request: List of task IDs
        
    Returns:
        List of task status information
    """
    results = []
    
    for task_id in request.task_ids:
        # Check in queue
        task_dict = queue_manager.get_task(task_id)
        if task_dict:
            results.append(TaskStatusResponse(
                task_id=task_id,
                status=task_dict["status"],
                exists=True
            ))
            continue
        
        # Check in database
        db_record = await db.get_task_history(task_id)
        if db_record:
            results.append(TaskStatusResponse(
                task_id=task_id,
                status=db_record["status"],
                exists=True
            ))
            continue
        
        # Not found
        results.append(TaskStatusResponse(
            task_id=task_id,
            status="not_found",
            exists=False
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
