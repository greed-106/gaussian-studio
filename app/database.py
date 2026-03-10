"""
SQLite database for storing task state and completed tasks.
"""
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any


class TaskDatabase:
    """Unified database for task state and history."""
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
    
    def init_db_sync(self):
        """Initialize database schema (synchronous)."""
        conn = sqlite3.connect(self.db_path)
        try:
            # Active tasks (in queue)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS active_tasks (
                    task_id TEXT PRIMARY KEY,
                    video_path TEXT NOT NULL,
                    work_dir TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    error TEXT
                )
            """)
            
            # Completed/failed tasks (history)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS task_history (
                    task_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    completed_at TEXT NOT NULL
                )
            """)
            
            # Camera metadata (intrinsic and extrinsic parameters)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metadata (
                    task_id TEXT PRIMARY KEY,
                    intrinsic_matrix TEXT NOT NULL,
                    extrinsic_matrix TEXT NOT NULL
                )
            """)
            conn.commit()
        finally:
            conn.close()
    
    async def init_db(self):
        """Initialize database schema (async wrapper)."""
        self.init_db_sync()
    
    # === Active Tasks (Queue State) ===
    
    def add_active_task(self, task_id: str, video_path: str, work_dir: str, status: str, created_at: str):
        """Add a new active task."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT INTO active_tasks (task_id, video_path, work_dir, status, created_at, error) VALUES (?, ?, ?, ?, ?, ?)",
                (task_id, video_path, work_dir, status, created_at, None)
            )
            conn.commit()
        finally:
            conn.close()
    
    def get_active_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get active task by ID."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM active_tasks WHERE task_id = ?", (task_id,))
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
        finally:
            conn.close()
    
    def update_active_task_status(self, task_id: str, status: str, error: Optional[str] = None):
        """Update active task status."""
        conn = sqlite3.connect(self.db_path)
        try:
            if error:
                conn.execute(
                    "UPDATE active_tasks SET status = ?, error = ? WHERE task_id = ?",
                    (status, error, task_id)
                )
            else:
                conn.execute(
                    "UPDATE active_tasks SET status = ? WHERE task_id = ?",
                    (status, task_id)
                )
            conn.commit()
        finally:
            conn.close()
    
    def remove_active_task(self, task_id: str):
        """Remove task from active queue."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("DELETE FROM active_tasks WHERE task_id = ?", (task_id,))
            conn.commit()
        finally:
            conn.close()
    
    def get_queue_stats(self) -> Dict[str, int]:
        """Get statistics of active tasks."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute("SELECT status, COUNT(*) as count FROM active_tasks GROUP BY status")
            rows = cursor.fetchall()
            
            stats = {
                "total": 0,
                "waiting": 0,
                "preprocessing": 0,
                "sfm": 0,
                "reconstruction": 0,
                "compress": 0
            }
            
            for status, count in rows:
                stats["total"] += count
                if status in stats:
                    stats[status] = count
            
            return stats
        finally:
            conn.close()
    
    # === Task History (Completed/Failed) ===
    
    def save_task_history_sync(self, task_id: str, status: str, created_at: str, completed_at: str):
        """Save a completed or failed task to history (synchronous)."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT OR REPLACE INTO task_history (task_id, status, created_at, completed_at) VALUES (?, ?, ?, ?)",
                (task_id, status, created_at, completed_at)
            )
            conn.commit()
        finally:
            conn.close()
    
    async def get_task_history(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task from history (async)."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM task_history WHERE task_id = ?", (task_id,))
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
        finally:
            conn.close()
    
    # === Camera Metadata ===
    
    def save_metadata_sync(self, task_id: str, intrinsic_matrix: list, extrinsic_matrix: list):
        """Save camera metadata (synchronous)."""
        import json
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT OR REPLACE INTO metadata (task_id, intrinsic_matrix, extrinsic_matrix) VALUES (?, ?, ?)",
                (task_id, json.dumps(intrinsic_matrix), json.dumps(extrinsic_matrix))
            )
            conn.commit()
        finally:
            conn.close()
    
    async def get_metadata(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get camera metadata (async)."""
        import json
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                "SELECT intrinsic_matrix, extrinsic_matrix FROM metadata WHERE task_id = ?",
                (task_id,)
            )
            row = cursor.fetchone()
            if row:
                return {
                    "task_id": task_id,
                    "intrinsic_matrix": json.loads(row[0]),
                    "extrinsic_matrix": json.loads(row[1])
                }
            return None
        finally:
            conn.close()
