"""
Logging configuration for the WorkRB package.

Provides a unified logging setup with console output for user-facing messages
and optional file output for debugging.
"""

import logging
import sys


def setup_logger(name: str = "workrb", verbose: bool = False) -> logging.Logger:
    """
    Setup package logger with console and optional file handlers.

    Args:
        name: Logger name (usually package name)
        verbose: If True, show DEBUG logs on console, otherwise only INFO and above

    Returns
    -------
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Prevent duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    # Set base level (captures everything)
    logger.setLevel(logging.DEBUG)

    # Console handler - for user-facing messages
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    # Clean output format (no timestamp/level for user-facing messages)
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger
