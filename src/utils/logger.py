"""
Logging utilities for MLE-STAR project.
Provides consistent logging across all modules.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str,
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
    console_output: bool = True,
) -> logging.Logger:
    """
    Set up a logger with file and console handlers.

    Args:
        name: Logger name (typically __name__)
        log_file: Path to log file (optional)
        level: Logging level
        console_output: Whether to output to console

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str, log_dir: Optional[Path] = None) -> logging.Logger:
    """
    Get or create a logger with standard configuration.

    Args:
        name: Logger name
        log_dir: Directory for log files

    Returns:
        Logger instance
    """
    if log_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{name}_{timestamp}.log"
    else:
        log_file = None

    return setup_logger(name, log_file)


class LoggerContext:
    """Context manager for temporary logging configuration."""

    def __init__(self, logger: logging.Logger, level: int):
        self.logger = logger
        self.level = level
        self.old_level = None

    def __enter__(self):
        self.old_level = self.logger.level
        self.logger.setLevel(self.level)
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.old_level)


def log_execution_time(logger: logging.Logger):
    """
    Decorator to log function execution time.

    Args:
        logger: Logger instance
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            logger.info(f"Starting {func.__name__}...")

            try:
                result = func(*args, **kwargs)
                elapsed = (datetime.now() - start_time).total_seconds()
                logger.info(f"Completed {func.__name__} in {elapsed:.2f} seconds")
                return result
            except Exception as e:
                elapsed = (datetime.now() - start_time).total_seconds()
                logger.error(f"Failed {func.__name__} after {elapsed:.2f} seconds: {str(e)}")
                raise

        return wrapper
    return decorator


# Example usage
if __name__ == "__main__":
    from .config import LOGS_DIR

    # Create logger
    logger = get_logger("test_logger", LOGS_DIR)

    # Test logging
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")

    # Test decorator
    @log_execution_time(logger)
    def example_function():
        import time
        time.sleep(1)
        return "Done"

    example_function()
