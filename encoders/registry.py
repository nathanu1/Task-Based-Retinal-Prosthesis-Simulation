"""
Backend registry — discovers which backends are available and returns
the appropriate instance.

Usage
-----
    from encoders.registry import get_backend, list_backends, BACKEND_NAMES

    # All backend display names (always shown, even if unavailable)
    names = BACKEND_NAMES

    # Check availability and reasons
    for name, (backend, avail, reason) in list_backends().items():
        print(name, avail, reason)

    # Get a usable backend instance
    backend = get_backend("Dots Baseline")
    result = backend.run(context, params)
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .dots_backend import DotsBackend
from .toolkit_backend import ToolkitBackend
from .p2p_retinal_backend import P2PRetinalBackend
from .dynaphos_backend import DynaphosBackend
from .learned_backend import LearnedEncoderBackend
from .base import PhospheneBackend

# Canonical ordered list of backend display names
BACKEND_NAMES: List[str] = [
    "Dots Baseline",
    "Toolkit Pipeline",
    "Learned Encoder (E2E)",
    "p2p Retinal (Argus II)",
    "Dynaphos Cortical (p2p)",
]

# Singleton instances
_BACKENDS: Dict[str, PhospheneBackend] = {
    "Dots Baseline": DotsBackend(),
    "Toolkit Pipeline": ToolkitBackend(),
    "Learned Encoder (E2E)": LearnedEncoderBackend(),
    "p2p Retinal (Argus II)": P2PRetinalBackend(),
    "Dynaphos Cortical (p2p)": DynaphosBackend(),
}


def get_backend(name: str) -> PhospheneBackend:
    """Return the backend instance for *name*.  Always returns something."""
    return _BACKENDS[name]


def list_backends() -> Dict[str, Tuple[PhospheneBackend, bool, str]]:
    """Return dict of name → (instance, available, unavailable_reason)."""
    return {
        name: (b, b.available, b.unavailable_reason)
        for name, b in _BACKENDS.items()
    }


def available_backend_names() -> List[str]:
    """Subset of BACKEND_NAMES where backend.available is True."""
    return [n for n in BACKEND_NAMES if _BACKENDS[n].available]


def set_toolkit_encoder(enc: Any) -> None:
    """Register the Streamlit-cached PhospheneEncoderTool instance."""
    ToolkitBackend.set_encoder(enc)
