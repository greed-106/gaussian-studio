"""
Colored logger for worker processes.
"""
import sys
from datetime import datetime
from colorama import Fore, Style, init

# Initialize colorama for Windows support
init(autoreset=True)


class WorkerLogger:
    """Colored logger for worker processes."""
    
    # Color mapping for different modules
    COLORS = {
        "Main": Fore.CYAN,
        "Preprocessing": Fore.GREEN,
        "SfM": Fore.YELLOW,
        "Reconstruction": Fore.MAGENTA,
        "Compress": Fore.BLUE,
    }
    
    @staticmethod
    def log(module: str, message: str):
        """
        Log a message with timestamp and colored module name.
        
        Args:
            module: Module name (e.g., "Preprocessing", "SfM")
            message: Log message
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        color = WorkerLogger.COLORS.get(module, Fore.WHITE)
        
        # Use sys.stdout.write with flush to ensure atomic output
        output = f"{Style.DIM}{timestamp}{Style.RESET_ALL} {color}[{module}]{Style.RESET_ALL} {message}\n"
        sys.stdout.write(output)
        sys.stdout.flush()
    
    @staticmethod
    def log_worker_start(module: str):
        """Log worker start message."""
        WorkerLogger.log(module, f"{module} Worker Started")
    
    @staticmethod
    def log_task_begin(module: str, task_id: str):
        """Log task begin message."""
        short_id = task_id[:8]
        WorkerLogger.log(module, f"Task {Fore.CYAN}{short_id}{Style.RESET_ALL}: Begin")
    
    @staticmethod
    def log_task_finish(module: str, task_id: str):
        """Log task finish message."""
        short_id = task_id[:8]
        WorkerLogger.log(module, f"Task {Fore.CYAN}{short_id}{Style.RESET_ALL}: {Fore.GREEN}Finish{Style.RESET_ALL}")
    
    @staticmethod
    def log_task_failed(module: str, task_id: str, error: str):
        """Log task failure message."""
        short_id = task_id[:8]
        WorkerLogger.log(module, f"Task {Fore.CYAN}{short_id}{Style.RESET_ALL}: {Fore.RED}Failed{Style.RESET_ALL} - {error}")
