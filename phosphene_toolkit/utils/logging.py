"""Structured logging for phosphene toolkit."""

import logging
from typing import Any, Optional


class StructuredLogger:
    """Simple structured logger."""

    def __init__(self, name: str, level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        if not self.logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
            self.logger.addHandler(h)

    def info(self, msg: str, **extra: Any):
        self.logger.info(msg, extra=extra)

    def warning(self, msg: str, **extra: Any):
        self.logger.warning(msg, extra=extra)

    def error(self, msg: str, **extra: Any):
        self.logger.error(msg, extra=extra)
