"""
logging_config.py — Configure logging for the application.
Import this module once (in app.py) to set up consistent log formatting.
"""
import logging


def setup_logging(level: int = logging.INFO) -> None:
    """
    Configure the root logger with a consistent format.

    Args:
        level: The logging level (default: INFO).
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )