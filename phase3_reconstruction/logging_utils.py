"""Small logging helpers so every module logs consistently and the CLI can set
one verbosity knob.
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager

_CONFIGURED = False


def setup_logging(level: str = "INFO") -> None:
    """Configure the root ``phase3_reconstruction`` logger once."""
    global _CONFIGURED
    lvl = getattr(logging, str(level).upper(), logging.INFO)
    logger = logging.getLogger("phase3_reconstruction")
    logger.setLevel(lvl)
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)-7s %(name)s | %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
        _CONFIGURED = True
    for h in logger.handlers:
        h.setLevel(lvl)


def get_logger(name: str) -> logging.Logger:
    """``get_logger(__name__)`` -> a child of the package logger."""
    if not name.startswith("phase3_reconstruction"):
        name = "phase3_reconstruction." + name.split(".")[-1]
    return logging.getLogger(name)


@contextmanager
def log_stage(logger: logging.Logger, title: str):
    """Context manager that brackets a pipeline stage with timing lines."""
    logger.info("=== %s ===", title)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        logger.info("--- %s done in %.2fs ---", title, time.perf_counter() - t0)
