"""
WTEB - A benchmarking framework for evaluating models on various tasks.
"""

from wteb import metrics, models, tasks
from wteb.evaluate import (
    evaluate,
    evaluate_multiple_models,
    get_tasks_overview,
    list_available_tasks,
    load_results,
)

__all__ = [
    "evaluate",
    "evaluate_multiple_models",
    "get_tasks_overview",
    "list_available_tasks",
    "load_results",
    "metrics",
    "models",
    "tasks",
]
