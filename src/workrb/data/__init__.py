"""
Simplified data management for WorkRB.

This module provides:
- ESCODataManager: Handles ESCO data downloads and caching
- ONET: Handles O*NET data downloads and caching
- utils: Simple utilities for loading local data files
"""

from workrb.data.esco import ESCO
from workrb.data.onet import ONET
from workrb.types import ModelInputType

__all__ = [
    "ESCO",
    "ONET",
    "ModelInputType",
]
