"""Base task policy interface."""

from abc import ABC, abstractmethod
from .schemas import TaskParams


class TaskPolicy(ABC):
    """Abstract base for task policy."""

    @abstractmethod
    def parse_task(self, task_description: str) -> TaskParams:
        """Parse natural language task into TaskParams."""
        pass
