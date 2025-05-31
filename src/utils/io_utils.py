"""
I/O utility functions for the project
"""

import os
import json
import yaml
import pickle
from pathlib import Path
from typing import Any, Dict, Union, Optional


def ensure_dir(path: Union[str, Path]) -> Path:
    """Create directory if it doesn't exist"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Union[str, Path]) -> Dict:
    """Load JSON file"""
    with open(path, "r") as f:
        return json.load(f)


def save_json(data: Dict, path: Union[str, Path], indent: int = 2):
    """Save data to JSON file"""
    ensure_dir(Path(path).parent)
    with open(path, "w") as f:
        json.dump(data, f, indent=indent)


def load_yaml(path: Union[str, Path]) -> Dict:
    """Load YAML file"""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict, path: Union[str, Path]):
    """Save data to YAML file"""
    ensure_dir(Path(path).parent)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)


def load_pickle(path: Union[str, Path]) -> Any:
    """Load pickle file"""
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(data: Any, path: Union[str, Path]):
    """Save data to pickle file"""
    ensure_dir(Path(path).parent)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_text(path: Union[str, Path]) -> str:
    """Load text file"""
    with open(path, "r") as f:
        return f.read()


def save_text(text: str, path: Union[str, Path]):
    """Save text to file"""
    ensure_dir(Path(path).parent)
    with open(path, "w") as f:
        f.write(text)


def file_exists(path: Union[str, Path]) -> bool:
    """Check if file exists"""
    return Path(path).exists()


def get_file_size(path: Union[str, Path]) -> int:
    """Get file size in bytes"""
    return Path(path).stat().st_size


def list_files(
    directory: Union[str, Path],
    extension: Optional[str] = None,
    recursive: bool = False,
) -> list:
    """List files in directory"""
    path = Path(directory)

    if extension:
        if not extension.startswith("."):
            extension = "." + extension

        if recursive:
            return sorted(list(path.rglob(f"*{extension}")))
        else:
            return sorted(list(path.glob(f"*{extension}")))
    else:
        if recursive:
            return sorted([f for f in path.rglob("*") if f.is_file()])
        else:
            return sorted([f for f in path.glob("*") if f.is_file()])
