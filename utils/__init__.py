"""Utilities package."""

from .losses import HybridLoss

# Keep this package import side-effect free.
# Importing `utils.phosphene` can trigger optional heavy deps (e.g. pulse2percept).
__all__ = ["HybridLoss"]
