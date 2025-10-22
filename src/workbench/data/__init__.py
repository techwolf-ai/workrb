"""
Simplified data management for WorkBench.

This module provides:
- ESCODataManager: Handles ESCO data downloads and caching
- utils: Simple utilities for loading local data files
"""

from workbench.data.esco import ESCO
from workbench.data.input_types import ModelInputType

__all__ = [
    "ESCO",
    "ModelInputType",
]
