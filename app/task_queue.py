"""
Task queue management with multiprocessing support.
"""
from multiprocessing import Queue
from typing import Dict, Any, Optional
from datetime import datetime
from enum import Enum


class TaskStatus(str, Enum):
    """Task status enumeration."""
    WAITING = "waiting"
    PREPROCESSING = "preprocessing"
    SFM = "sfm"
    RECONSTRUCTION = "reconstruction"
    COMPRESS = "compress"
    FINISH = "finish"
    FAILURE = "failure"


class Task:
    """Task data structure."""
    
    def __init__(self, task_id: str, video_path: str, work_dir: str):
        self.task_id = task_id
        self.video_path = video_path
        self.work_dir = work_dir
        self.status = TaskStatus.WAITING
        self.created_at = datetime.now()
        self.error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary."""
        return {
            "task_id": self.task_id,
            "video_path": self.video_path,
            "work_dir": self.work_dir,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "error": self.error
        }


class TaskQueueManager:
    """
    Manages task queues for the pipeline.
    Uses TaskDatabase for state storage (Windows-friendly).
    """
    
    def __init__(self, db):
        self.db = db
        
        # Queues for each stage
        self.preprocessing_queue = Queue()
        self.sfm_queue = Queue()
        self.reconstruction_queue = Queue()
        self.compress_queue = Queue()
    
    def add_task(self, task: Task):
        """Add a new task to the queue."""
        self.db.add_active_task(
            task_id=task.task_id,
            video_path=task.video_path,
            work_dir=task.work_dir,
            status=task.status,
            created_at=task.created_at
        )
        self.preprocessing_queue.put(task.task_id)
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task by ID."""
        return self.db.get_active_task(task_id)
    
    def update_task_status(self, task_id: str, status: TaskStatus, error: Optional[str] = None):
        """Update task status."""
        self.db.update_active_task_status(task_id, status, error)
    
    def remove_task(self, task_id: str):
        """Remove task from queue (when finished or failed)."""
        self.db.remove_active_task(task_id)
    
    def get_queue_stats(self) -> Dict[str, int]:
        """Get statistics of tasks in each status."""
        return self.db.get_queue_stats()
