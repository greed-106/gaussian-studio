#!/usr/bin/env python3
"""
Gaussian Reconstruction Backend - Main Entry Point

启动 FastAPI 服务和后台 worker 进程。
"""
import sys
import signal
import multiprocessing
from multiprocessing import Process
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_config
from app.database import TaskDatabase
from app.task_queue import TaskQueueManager
from app.api.routes import router, init_routes
from app.logger import WorkerLogger
from app.workers import (
    preprocessing_worker,
    sfm_worker,
    reconstruction_worker,
    compress_worker
)


# Global worker processes
workers = []


def signal_handler(signum, frame):
    """Handle Ctrl+C for graceful shutdown."""
    WorkerLogger.log("Main", "Shutting down...")
    for worker in workers:
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=5)
    sys.exit(0)


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="Gaussian Reconstruction Backend",
        description="Backend API for video-to-3D Gaussian Splatting reconstruction",
        version="1.0.0"
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routes
    app.include_router(router)
    
    return app


async def startup_event(app: FastAPI, config: dict, db: TaskDatabase, queue_manager: TaskQueueManager):
    """Initialize database and start worker processes."""
    WorkerLogger.log("Main", "Initializing database...")
    await db.init_db()
    
    WorkerLogger.log("Main", "Starting worker processes...")
    
    db_path = config["STORAGE"]["DATABASE_PATH"]
    
    # Start workers
    workers.append(Process(
        target=preprocessing_worker,
        args=(
            db_path,
            queue_manager.preprocessing_queue,
            queue_manager.sfm_queue,
            config
        ),
        daemon=True
    ))
    
    workers.append(Process(
        target=sfm_worker,
        args=(
            db_path,
            queue_manager.sfm_queue,
            queue_manager.reconstruction_queue,
            config
        ),
        daemon=True
    ))
    
    workers.append(Process(
        target=reconstruction_worker,
        args=(
            db_path,
            queue_manager.reconstruction_queue,
            queue_manager.compress_queue,
            config
        ),
        daemon=True
    ))
    
    workers.append(Process(
        target=compress_worker,
        args=(
            db_path,
            queue_manager.compress_queue,
            config
        ),
        daemon=True
    ))
    
    for worker in workers:
        worker.start()
    
    WorkerLogger.log("Main", f"Started {len(workers)} worker processes")


def main():
    """Main entry point."""
    # Parse port from command line
    port = 4000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Invalid port: {sys.argv[1]}, using default 4000")
    
    # Load configuration
    WorkerLogger.log("Main", "Loading configuration...")
    config = get_config()
    
    # Override port from config if specified
    if "PORT" in config:
        port = config["PORT"]
    
    # Initialize database and queue manager
    db = TaskDatabase(config["STORAGE"]["DATABASE_PATH"])
    queue_manager = TaskQueueManager(db)
    
    # Create app
    app = create_app()
    
    # Initialize routes with dependencies
    init_routes(queue_manager, db, config)
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Startup event
    import asyncio
    asyncio.run(startup_event(app, config, db, queue_manager))
    
    WorkerLogger.log("Main", f"Starting server on http://localhost:{port}")
    WorkerLogger.log("Main", f"API docs: http://localhost:{port}/docs")
    WorkerLogger.log("Main", f"ReDoc: http://localhost:{port}/redoc")
    WorkerLogger.log("Main", "Press Ctrl+C to stop")
    
    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    # Required for Windows multiprocessing
    multiprocessing.freeze_support()
    main()
