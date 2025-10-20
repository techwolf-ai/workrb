"""
WorkBench - A benchmarking framework for evaluating models on various tasks.
"""

__version__ = "0.1.0"

from workbench import metrics, models, tasks
from workbench.logging import setup_logger
from workbench.workbench import WorkBench

__all__ = [
    "WorkBench",
    "metrics",
    "models",
    "setup_logger",
    "tasks",
]
