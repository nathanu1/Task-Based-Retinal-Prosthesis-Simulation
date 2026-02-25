"""Device constraint model for phosphene stimulation."""

import numpy as np
from typing import Dict
from ..task_policy.schemas import DeviceConfig


class DeviceModel:
    """Applies device constraints to stimulation."""

    def __init__(self, config: DeviceConfig):
        self.config = config

    def process_stimulation(self, allocated: np.ndarray) -> Dict:
        """Apply device constraints; return constrained_stim and diagnostics."""
        constrained = np.clip(allocated, 0, self.config.max_amplitude_per_electrode)
        total = np.sum(constrained)
        cap = self.config.global_power_cap
        if total > cap and cap > 0:
            constrained = constrained * (cap / total)
        diagnostics = {
            "total_power": float(np.sum(constrained)),
            "active_electrodes": int(np.sum(constrained > 0.01)),
        }
        return {"constrained_stim": constrained, "diagnostics": diagnostics}
