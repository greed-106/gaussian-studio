"""
Utility functions for the Gaussian Reconstruction Backend.
"""
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

from app.logger import WorkerLogger
from app.database import TaskDatabase


def format_utc_time(dt: datetime) -> str:
    """
    Format datetime to ISO 8601 string without microseconds and with Z suffix.
    
    Args:
        dt: datetime object (should be in UTC)
        
    Returns:
        ISO 8601 string like "2026-03-10T14:30:05Z"
    """
    # Ensure datetime is in UTC
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    
    # Format without microseconds and with Z suffix
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def get_log_file_path(work_dir: Path, stage: str) -> Path:
    """
    Get log file path for a specific pipeline stage.
    
    Args:
        work_dir: Task work directory
        stage: Pipeline stage name (preprocessing, sfm, reconstruction, compress)
        
    Returns:
        Path to log file
    """
    return work_dir / f"{stage}.log"


def handle_task_failure(
    db: TaskDatabase,
    stage_name: str,
    task_id: str,
    task_dict: Dict[str, Any],
    error: Exception
) -> None:
    """
    Handle task failure: log error and save to history.
    
    Args:
        db: Database instance
        stage_name: Name of the pipeline stage
        task_id: Task ID
        task_dict: Task dictionary from database
        error: Exception that caused the failure
    """
    WorkerLogger.log_task_failed(stage_name, task_id, str(error))
    
    # Save failure to history
    created_at = task_dict["created_at"]  # Already formatted as string
    completed_at = format_utc_time(datetime.now(timezone.utc))
    
    db.save_task_history_sync(
        task_id=task_id,
        status="failure",
        created_at=created_at,
        completed_at=completed_at
    )
    
    db.remove_active_task(task_id)


def handle_task_success(
    db: TaskDatabase,
    stage_name: str,
    task_id: str,
    task_dict: Dict[str, Any]
) -> None:
    """
    Handle task success: log completion and save to history.
    
    Args:
        db: Database instance
        stage_name: Name of the pipeline stage
        task_id: Task ID
        task_dict: Task dictionary from database
    """
    WorkerLogger.log_task_finish(stage_name, task_id)
    
    # Save to history
    created_at = task_dict["created_at"]  # Already formatted as string
    completed_at = format_utc_time(datetime.now(timezone.utc))
    
    db.save_task_history_sync(
        task_id=task_id,
        status="finish",
        created_at=created_at,
        completed_at=completed_at
    )
    
    db.remove_active_task(task_id)