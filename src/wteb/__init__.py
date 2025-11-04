"""
WTEB - A benchmarking framework for evaluating models on various tasks.
"""

__version__ = "0.1.0"

from wteb import metrics, models, tasks
from wteb.logging import setup_logger
from wteb.wteb import WTEB

__all__ = [
    "WTEB",
    "metrics",
    "models",
    "setup_logger",
    "tasks",
]
