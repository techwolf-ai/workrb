"""
Simplified data management for WorkRB.

This module provides:
- ESCODataManager: Handles ESCO data downloads and caching
- utils: Simple utilities for loading local data files
"""

from workrb.data.esco import ESCO
from workrb.types import ModelInputType

__all__ = [
    "ESCO",
    "ModelInputType",
]
