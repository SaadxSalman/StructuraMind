"""Structured logging configuration for StructuraMind."""

from __future__ import annotations

import logging
import sys

from app.core.config import settings


def setup_logging() -> None:
    """Configure the root logger once at application startup."""
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )

    # Quieten noisy third-party loggers unless in debug mode.
    if not settings.debug:
        for name in ("uvicorn", "httpx", "httpcore", "qdrant_client"):
            logging.getLogger(name).setLevel(logging.WARNING)


logger = logging.getLogger("structuramind")

__all__ = ["setup_logging", "logger"]
