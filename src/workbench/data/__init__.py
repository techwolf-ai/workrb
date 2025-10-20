"""
Simplified data management for WorkBench.

This module provides:
- ESCODataManager: Handles ESCO data downloads and caching
- utils: Simple utilities for loading local data files
"""

from workbench.data.esco import ESCO
from workbench.data.input_types import ModelInputType
from workbench.data.utils import (
    ensure_path_exists,
    get_data_path,
    load_csv,
    load_json,
    load_jsonl,
    load_parquet,
)

__all__ = [
    "ESCO",
    "ModelInputType",
    "ensure_path_exists",
    "get_data_path",
    "load_csv",
    "load_json",
    "load_jsonl",
    "load_parquet",
]
