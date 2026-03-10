"""
Worker processes for each pipeline stage.
"""
import signal
import sys
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any
from multiprocessing import Queue
from contextlib import redirect_stdout, redirect_stderr

from app.task_queue import TaskStatus, TaskQueueManager
from app.utils import format_utc_time, get_log_file_path, handle_task_failure, handle_task_success
from app.database import TaskDatabase
from app.pipeline import extract_frames, run_colmap_sfm, run_lichtfeld_training, compress_splat
from app.pipeline.colmap_sfm import extract_camera_parameters
from app.logger import WorkerLogger


# Global flag for graceful shutdown
shutdown_flag = False


def signal_handler(signum, frame):
    """Handle Ctrl+C for graceful shutdown."""
    global shutdown_flag
    WorkerLogger.log("Main", f"Received signal {signum}, shutting down gracefully...")
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
    
    WorkerLogger.log_worker_start("Preprocessing")
    
    while not shutdown_flag:
        try:
            # Blocking dequeue with timeout
            task_id = preprocessing_queue.get(timeout=1)
            
            task_dict = db.get_active_task(task_id)
            if not task_dict:
                continue
            
            WorkerLogger.log_task_begin("Preprocessing", task_id)
            db.update_active_task_status(task_id, TaskStatus.PREPROCESSING)
            
            try:
                # Extract frames
                work_dir = Path(task_dict["work_dir"])
                video_path = task_dict["video_path"]
                output_dir = work_dir / "images"
                log_file = get_log_file_path(work_dir, "preprocessing")
                
                frame_config = config["FRAME_EXTRACTION"]
                
                # Redirect stdout/stderr to log file
                with open(log_file, "a", encoding="utf-8") as f:
                    with redirect_stdout(f), redirect_stderr(f):
                        result = extract_frames(
                            video_path=video_path,
                            output_dir=output_dir,
                            ffmpeg_exe=config["BINARIES"]["FFMPEG_PATH"],
                            ratio=frame_config["ratio"],
                            min_buffer=frame_config["min_buffer"],
                            resize_factor=frame_config["resize_factor"],
                            log=lambda msg: print(msg),  # Will be redirected to file
                            log_file=log_file
                        )
                
                if not result.get("success"):
                    raise RuntimeError(result.get("error", "Frame extraction failed"))
                
                # Move to next stage
                sfm_queue.put(task_id)
                WorkerLogger.log_task_finish("Preprocessing", task_id)
                
            except Exception as e:
                handle_task_failure(db, "Preprocessing", task_id, task_dict, e)
                
        except Exception:
            # Timeout or other errors, continue
            continue
    
    WorkerLogger.log("Preprocessing", "Worker Stopped")


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
    
    WorkerLogger.log_worker_start("SfM")
    
    while not shutdown_flag:
        try:
            task_id = sfm_queue.get(timeout=1)
            
            task_dict = db.get_active_task(task_id)
            if not task_dict:
                continue
            
            WorkerLogger.log_task_begin("SfM", task_id)
            db.update_active_task_status(task_id, TaskStatus.SFM)
            
            try:
                work_dir = Path(task_dict["work_dir"])
                log_file = get_log_file_path(work_dir, "sfm")
                
                result = run_colmap_sfm(
                    source=work_dir,
                    colmap_exe=config["BINARIES"]["COLMAP_PATH"],
                    log=lambda msg: None,  # Suppress console output
                    log_file=log_file
                )
                
                if not result.get("success"):
                    raise RuntimeError("SfM failed")
                
                # Extract camera parameters
                sparse_dir = work_dir / "sparse" / "0"
                camera_params = extract_camera_parameters(sparse_dir)
                if camera_params:
                    intrinsic_matrix, extrinsic_matrix = camera_params
                    db.save_metadata_sync(task_id, intrinsic_matrix, extrinsic_matrix)
                else:
                    WorkerLogger.log("SfM", f"Warning: Could not extract camera parameters for task {task_id[:8]}")
                
                reconstruction_queue.put(task_id)
                WorkerLogger.log_task_finish("SfM", task_id)
                
            except Exception as e:
                handle_task_failure(db, "SfM", task_id, task_dict, e)
                
        except Exception:
            continue
    
    WorkerLogger.log("SfM", "Worker Stopped")


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
    
    WorkerLogger.log_worker_start("Reconstruction")
    
    while not shutdown_flag:
        try:
            task_id = reconstruction_queue.get(timeout=1)
            
            task_dict = db.get_active_task(task_id)
            if not task_dict:
                continue
            
            WorkerLogger.log_task_begin("Reconstruction", task_id)
            db.update_active_task_status(task_id, TaskStatus.RECONSTRUCTION)
            
            try:
                work_dir = Path(task_dict["work_dir"])
                lf_params = config["LICHTFELD_PARAMS"]
                log_file = get_log_file_path(work_dir, "reconstruction")
                
                result = run_lichtfeld_training(
                    executable=config["BINARIES"]["LICHTFELD_PATH"],
                    data_path=work_dir,
                    output_path=work_dir,
                    iterations=lf_params["iterations"],
                    max_cap=lf_params["max_cap"],
                    headless=lf_params["headless"],
                    ppisp=lf_params["ppisp"],
                    enable_mip=lf_params["enable_mip"],
                    log_file=log_file
                )
                
                if not result.get("success"):
                    raise RuntimeError("Reconstruction failed")
                
                compress_queue.put(task_id)
                WorkerLogger.log_task_finish("Reconstruction", task_id)
                
            except Exception as e:
                handle_task_failure(db, "Reconstruction", task_id, task_dict, e)
                
        except Exception:
            continue
    
    WorkerLogger.log("Reconstruction", "Worker Stopped")


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
    
    WorkerLogger.log_worker_start("Compress")
    
    while not shutdown_flag:
        try:
            task_id = compress_queue.get(timeout=1)
            
            task_dict = db.get_active_task(task_id)
            if not task_dict:
                continue
            
            WorkerLogger.log_task_begin("Compress", task_id)
            db.update_active_task_status(task_id, TaskStatus.COMPRESS)
            
            try:
                work_dir = Path(task_dict["work_dir"])
                log_file = get_log_file_path(work_dir, "compress")
                
                # Find the latest splat_*.ply file
                ply_files = sorted(work_dir.glob("splat_*.ply"))
                if not ply_files:
                    raise RuntimeError("No splat PLY file found")
                
                ply_file = ply_files[-1]  # Get the latest one
                iteration = ply_file.stem.split("_")[1]
                sog_file = work_dir / f"splat_{iteration}.sog"
                
                result = compress_splat(
                    input_path=ply_file,
                    output_path=sog_file,
                    log_file=log_file
                )
                
                if not result.get("success"):
                    raise RuntimeError("Compression failed")
                
                # Task completed successfully
                handle_task_success(db, "Compress", task_id, task_dict)
                
            except Exception as e:
                handle_task_failure(db, "Compress", task_id, task_dict, e)
                
        except Exception:
            continue
    
    WorkerLogger.log("Compress", "Worker Stopped")
