"""
Configuration loader for the backend.
"""
import json
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = "config.jsonc") -> Dict[str, Any]:
    """
    Load configuration from JSONC file (JSON with comments).
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Read and strip comments
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            # Remove line comments
            if "//" in line:
                line = line[:line.index("//")]
            lines.append(line)
    
    content = "\n".join(lines)
    return json.loads(content)


# Global config instance
_config = None


def get_config() -> Dict[str, Any]:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config
