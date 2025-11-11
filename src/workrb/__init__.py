"""
WTEB - A benchmarking framework for evaluating models on various tasks.
"""

from wteb import data, metrics, models, tasks
from wteb.evaluate import (
    evaluate,
    evaluate_multiple_models,
    get_tasks_overview,
    list_available_tasks,
    load_results,
)

__all__ = [
    "data",
    "evaluate",
    "evaluate_multiple_models",
    "get_tasks_overview",
    "list_available_tasks",
    "load_results",
    "metrics",
    "models",
    "tasks",
]
