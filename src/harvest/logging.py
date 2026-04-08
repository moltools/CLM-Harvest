"""Utility functions for logging."""

import logging
import sys
from pathlib import Path

PACKAGE_LOGGER = "harvest"

STANDARD_FMT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
STANDARD_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: str | int = "INFO",
    *,
    fmt: str = STANDARD_FMT,
    datefmt: str = STANDARD_DATEFMT,
    stream: None | int | str | object = None,
) -> None:
    """
    Set up logging.

    :param level: Log level for console output.
    :param fmt: Log message format.
    :param datefmt: Date format for log messages.
    :param stream: Output stream for console logs; defaults to sys.stderr.
    .. note:: Safe to call multiple times. Library code should not call this function;
        it is intended for use by applications using the library.
    """
    if stream is None:
        stream = sys.stderr

    if isinstance(level, str):
        level = level.upper()

    root = logging.getLogger()
    root.setLevel(level)

    handler = logging.StreamHandler(stream)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))

    # Avoid duplicate handlers if called repeatedly (common in notebooks)
    # Keep it simple: remove existing handlers created by previous setup calls.
    root.handlers = [handler]

    # Make sure package logger propagates to root
    logging.getLogger(PACKAGE_LOGGER).propagate = True


def add_file_handler(
    logfile: Path | str,
    *,
    level: str | int = "DEBUG",
    fmt: str = STANDARD_FMT,
    datefmt: str = STANDARD_DATEFMT,
) -> None:
    """
    Add a file handler to the root logger.

    :param logfile: Path to log file.
    :param level: Log level for file output.
    :param fmt: Log message format.
    :param datefmt: Date format for log messages.
    .. note:: Intended to be called after setup_logger(); safe to call multiple times for the same logfile.
    """
    logfile = Path(logfile)

    if isinstance(level, str):
        level = level.upper()

    root = logging.getLogger()

    # Prevent duplicate file handlers for the same path
    for h in root.handlers:
        if isinstance(h, logging.FileHandler) and h.baseFilename == logfile:
            return

    fh = logging.FileHandler(logfile)
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))

    root.addHandler(fh)
