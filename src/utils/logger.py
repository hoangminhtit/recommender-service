"""
Centralized logging configuration for the recommender service.
Uses tqdm-safe output to avoid progress bar corruption.
"""

import logging
import sys
from tqdm import tqdm


class TqdmLoggingHandler(logging.Handler):
    """Log handler that writes through tqdm to avoid progress bar corruption."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger configured with:
    - tqdm-safe console handler
    - formatted output: [LEVEL] [name] message
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        # Avoid adding duplicate handlers on repeated calls
        return logger

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = TqdmLoggingHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Prevent propagation to root logger (avoids duplicate output)
    logger.propagate = False

    return logger
