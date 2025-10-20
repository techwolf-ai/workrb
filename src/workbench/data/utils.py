"""
Simple data loading utilities for WorkBench tasks.

This module provides basic utilities for loading local data files.
For ESCO data, use the ESCODataManager directly.
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd


def get_data_path(*parts: str) -> Path:
    """
    Get path to data file in the package data directory.

    Args:
        *parts: Path components relative to the data directory

    Returns
    -------
        Full path to the data file
    """
    # Get package root (workbench-toolbox/)
    dataset_root = Path(__file__).parent / "datasets"
    return dataset_root / Path(*parts)


def load_json(path: Path | str) -> Any:
    """
    Load JSON file.

    Args:
        path: Path to JSON file

    Returns
    -------
        Loaded JSON data
    """
    if isinstance(path, str):
        path = Path(path)

    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_csv(path: Path | str, **kwargs) -> pd.DataFrame:
    """
    Load CSV file.

    Args:
        path: Path to CSV file
        **kwargs: Additional arguments passed to pd.read_csv

    Returns
    -------
        Loaded DataFrame
    """
    if isinstance(path, str):
        path = Path(path)

    return pd.read_csv(path, **kwargs)


def load_parquet(path: Path | str, **kwargs) -> pd.DataFrame:
    """
    Load Parquet file.

    Args:
        path: Path to Parquet file
        **kwargs: Additional arguments passed to pd.read_parquet

    Returns
    -------
        Loaded DataFrame
    """
    if isinstance(path, str):
        path = Path(path)

    return pd.read_parquet(path, **kwargs)


def load_jsonl(path: Path | str) -> list[dict]:
    """
    Load JSONL (JSON Lines) file.

    Args:
        path: Path to JSONL file

    Returns
    -------
        List of JSON objects
    """
    if isinstance(path, str):
        path = Path(path)

    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def ensure_path_exists(path: Path | str) -> Path:
    """
    Ensure a file path exists, raising a clear error if not.

    Args:
        path: Path to check

    Returns
    -------
        Path object

    Raises
    ------
        FileNotFoundError: If path doesn't exist
    """
    if isinstance(path, str):
        path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    return path
