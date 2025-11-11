"""
Abstract base classes for tasks.

This module contains the base classes that define the interface
for all task implementations.
"""

from .base import LabelType, Task
from .classification_base import (
    ClassificationTask,
    MulticlassClassificationTask,
    MultiLabelClassificationTask,
)
from .ranking_base import RankingTask

__all__ = [
    "ClassificationTask",
    "LabelType",
    "MultiLabelClassificationTask",
    "MulticlassClassificationTask",
    "RankingTask",
    "Task",
]
