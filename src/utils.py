"""
Utility helper functions for ITM-390 ASD Screening project.
"""

import os
import random
import json
import logging
from pathlib import Path
from typing import Any, Dict, Union
import numpy as np
from tabulate import tabulate


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across random, numpy, etc."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_logger(name: str = "asd_screening", log_level: int = logging.INFO) -> logging.Logger:
    """Configure a standard project logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(log_level)
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure a directory exists and return Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: Dict[str, Any], filepath: Union[str, Path], indent: int = 2) -> None:
    """Save dictionary to a formatted JSON file."""
    p = Path(filepath)
    ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)


def load_json(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Load dictionary from JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def format_markdown_table(data, headers="keys") -> str:
    """Format data as GitHub-compatible markdown table."""
    return tabulate(data, headers=headers, tablefmt="github", floatfmt=".4f")
