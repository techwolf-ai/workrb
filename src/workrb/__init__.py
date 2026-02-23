"""
WorkRB - A benchmarking framework for evaluating models on various tasks.
"""

from workrb import data, metrics, models, tasks
from workrb.registry import list_available_tasks
from workrb.results import load_results
from workrb.run import evaluate, evaluate_multiple_models, get_tasks_overview
from workrb.types import ExecutionMode, LanguageAggregationMode

__all__ = [
    "ExecutionMode",
    "LanguageAggregationMode",
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
