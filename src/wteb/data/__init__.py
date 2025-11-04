"""
Simplified data management for WTEB.

This module provides:
- ESCODataManager: Handles ESCO data downloads and caching
- utils: Simple utilities for loading local data files
"""

from wteb.data.esco import ESCO
from wteb.types import ModelInputType

__all__ = [
    "ESCO",
    "ModelInputType",
]
