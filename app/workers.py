"""
Worker processes for each pipeline stage.
"""
import signal
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
from multiprocessing import Queue

from app.task_queue import TaskStatus, TaskQueueManager
from app.database import TaskDatabase
from app.pipeline import extract_frames, run_colmap_sfm, run_lichtfeld_training, compress_splat


# Global flag for graceful shutdown
shutdown_flag = False


def signal_handler(signum, frame):
    """Handle Ctrl+C for graceful shutdown."""
    global shutdown_flag
    print(f"\n[Worker] Received signal {signum}, shutting down gracefully...")
    shutdown_flag = True


def preprocessing_worker(
    db_path: str,
    preprocessing_queue: Queue,
    sfm_queue: Queue,
    config: Dict[str, Any]
):
    """Worker for frame extraction stage."""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    from app.database import TaskDatabase
    from app.task_queue import TaskStatus
    db = TaskDatabase(db_path)
    
    print("[Preprocessing Worker] Started")
    
    while not shutdown_flag:
        try:
            # Blocking dequeue with timeout
            task_id = preprocessing_queue.get(timeout=1)
            
            task_dict = db.get_active_task(task_id)
            if not task_dict:
                continue
            
            print(f"[Preprocessing] Processing task: {task_id}")
            db.update_active_task_status(task_id, TaskStatus.PREPROCESSING)
            
            try:
                # Extract frames
                work_dir = Path(task_dict["work_dir"])
                video_path = task_dict["video_path"]
                output_dir = work_dir / "images"
                
                frame_config = config["FRAME_EXTRACTION"]
                result = extract_frames(
                    video_path=video_path,
                    output_dir=output_dir,
                    ffmpeg_exe=config["BINARIES"]["FFMPEG_PATH"],
                    ratio=frame_config["ratio"],
                    min_buffer=frame_config["min_buffer"],
                    resize_factor=frame_config["resize_factor"],
                    log=lambda msg: print(f"[Preprocessing] {msg}")
                )
                
                if not result.get("success"):
                    raise RuntimeError(result.get("error", "Frame extraction failed"))
                
                # Move to next stage
                sfm_queue.put(task_id)
                print(f"[Preprocessing] Task {task_id} completed")
                
            except Exception as e:
                print(f"[Preprocessing] Task {task_id} failed: {e}")
                db.update_active_task_status(task_id, TaskStatus.FAILURE, str(e))
                # Will be saved to DB by main process
                
        except Exception:
            # Timeout or other errors, continue
            continue
    
    print("[Preprocessing Worker] Stopped")


def sfm_worker(
    db_path: str,
    sfm_queue: Queue,
    reconstruction_queue: Queue,
    config: Dict[str, Any]
):
    """Worker for SfM stage."""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    from app.database import TaskDatabase
    from app.task_queue import TaskStatus
    db = TaskDatabase(db_path)
    
    print("[SfM Worker] Started")
    
    while not shutdown_flag:
        try:
            task_id = sfm_queue.get(timeout=1)
            
            task_dict = db.get_active_task(task_id)
            if not task_dict:
                continue
            
            print(f"[SfM] Processing task: {task_id}")
            db.update_active_task_status(task_id, TaskStatus.SFM)
            
            try:
                work_dir = Path(task_dict["work_dir"])
                
                result = run_colmap_sfm(
                    source=work_dir,
                    colmap_exe=config["BINARIES"]["COLMAP_PATH"],
                    log=lambda msg: print(f"[SfM] {msg}")
                )
                
                if not result.get("success"):
                    raise RuntimeError("SfM failed")
                
                reconstruction_queue.put(task_id)
                print(f"[SfM] Task {task_id} completed")
                
            except Exception as e:
                print(f"[SfM] Task {task_id} failed: {e}")
                db.update_active_task_status(task_id, TaskStatus.FAILURE, str(e))
                
        except Exception:
            continue
    
    print("[SfM Worker] Stopped")


def reconstruction_worker(
    db_path: str,
    reconstruction_queue: Queue,
    compress_queue: Queue,
    config: Dict[str, Any]
):
    """Worker for Gaussian Splatting reconstruction stage."""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    from app.database import TaskDatabase
    from app.task_queue import TaskStatus
    db = TaskDatabase(db_path)
    
    print("[Reconstruction Worker] Started")
    
    while not shutdown_flag:
        try:
            task_id = reconstruction_queue.get(timeout=1)
            
            task_dict = db.get_active_task(task_id)
            if not task_dict:
                continue
            
            print(f"[Reconstruction] Processing task: {task_id}")
            db.update_active_task_status(task_id, TaskStatus.RECONSTRUCTION)
            
            try:
                work_dir = Path(task_dict["work_dir"])
                lf_params = config["LICHTFELD_PARAMS"]
                
                result = run_lichtfeld_training(
                    executable=config["BINARIES"]["LICHTFELD_PATH"],
                    data_path=work_dir,
                    output_path=work_dir,
                    iterations=lf_params["iterations"],
                    max_cap=lf_params["max_cap"],
                    headless=lf_params["headless"],
                    ppisp=lf_params["ppisp"],
                    enable_mip=lf_params["enable_mip"]
                )
                
                if not result.get("success"):
                    raise RuntimeError("Reconstruction failed")
                
                compress_queue.put(task_id)
                print(f"[Reconstruction] Task {task_id} completed")
                
            except Exception as e:
                print(f"[Reconstruction] Task {task_id} failed: {e}")
                db.update_active_task_status(task_id, TaskStatus.FAILURE, str(e))
                
        except Exception:
            continue
    
    print("[Reconstruction Worker] Stopped")


def compress_worker(
    db_path: str,
    compress_queue: Queue,
    config: Dict[str, Any]
):
    """Worker for compression stage."""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create database instance in worker process
    from app.database import TaskDatabase
    from app.task_queue import TaskStatus
    db = TaskDatabase(db_path)
    
    print("[Compress Worker] Started")
    
    while not shutdown_flag:
        try:
            task_id = compress_queue.get(timeout=1)
            
            task_dict = db.get_active_task(task_id)
            if not task_dict:
                continue
            
            print(f"[Compress] Processing task: {task_id}")
            db.update_active_task_status(task_id, TaskStatus.COMPRESS)
            
            try:
                work_dir = Path(task_dict["work_dir"])
                
                # Find the latest splat_*.ply file
                ply_files = sorted(work_dir.glob("splat_*.ply"))
                if not ply_files:
                    raise RuntimeError("No splat PLY file found")
                
                ply_file = ply_files[-1]  # Get the latest one
                iteration = ply_file.stem.split("_")[1]
                sog_file = work_dir / f"splat_{iteration}.sog"
                
                result = compress_splat(
                    input_path=ply_file,
                    output_path=sog_file
                )
                
                if not result.get("success"):
                    raise RuntimeError("Compression failed")
                
                # Task completed successfully
                print(f"[Compress] Task {task_id} completed")
                
                # Save to history
                created_at = datetime.fromisoformat(task_dict["created_at"])
                db.save_task_history_sync(
                    task_id=task_id,
                    status="finish",
                    created_at=created_at,
                    completed_at=datetime.now()
                )
                
                # Remove from active queue
                db.remove_active_task(task_id)
                
            except Exception as e:
                print(f"[Compress] Task {task_id} failed: {e}")
                
                # Save failure to history
                created_at = datetime.fromisoformat(task_dict["created_at"])
                db.save_task_history_sync(
                    task_id=task_id,
                    status="failure",
                    created_at=created_at,
                    completed_at=datetime.now()
                )
                
                db.remove_active_task(task_id)
                
        except Exception:
            continue
    
    print("[Compress Worker] Stopped")
