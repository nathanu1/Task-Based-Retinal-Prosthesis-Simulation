"""Task parameter schemas with validation."""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class RegionPriority(BaseModel):
    label: str = Field(..., description="Semantic label")
    weight: float = Field(..., ge=0.0, le=10.0)


class FoveaConfig(BaseModel):
    mode: Literal["center", "gaze", "task_driven", "disabled"] = "center"
    center_hint: Optional[tuple] = None
    radius: float = Field(0.3, ge=0.1, le=0.8)
    strength: float = Field(2.0, ge=1.0, le=5.0)


class TemporalConfig(BaseModel):
    base_smoothing: float = Field(0.7, ge=0.0, le=1.0)
    hysteresis: float = Field(0.1, ge=0.0, le=0.5)
    persistence_half_life: float = Field(0.5, ge=0.1, le=2.0)


class SafetyConfig(BaseModel):
    near_field_boost: float = Field(1.5, ge=1.0, le=3.0)
    hazard_labels: List[str] = Field(default=["person", "car", "bicycle", "motorcycle", "bus", "truck"])


class TaskParams(BaseModel):
    task_type: Literal["navigation", "grasping", "avoidance", "exploration"] = "navigation"
    region_priorities: List[RegionPriority] = Field(default_factory=list)
    edge_weight: float = Field(1.0, ge=0.0, le=5.0)
    floor_weight: float = Field(0.5, ge=0.0, le=5.0)
    obstacle_weight: float = Field(2.0, ge=0.0, le=5.0)
    motion_weight: float = Field(1.5, ge=0.0, le=5.0)
    fovea: FoveaConfig = Field(default_factory=FoveaConfig)
    temporal: TemporalConfig = Field(default_factory=TemporalConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)


class DeviceConfig(BaseModel):
    grid_size: tuple = Field((60, 60))
    amplitude_levels: int = Field(256, ge=2, le=1024)
    max_amplitude_per_electrode: float = Field(1.0, ge=0.1, le=2.0)
    global_power_cap: float = Field(100.0, ge=10.0, le=1000.0)
    spatial_spread_sigma: float = Field(1.2, ge=0.5, le=3.0)
    temporal_freq_hz: float = Field(20.0, ge=1.0, le=100.0)
    duty_cycle: float = Field(0.1, ge=0.01, le=0.5)
    dropout_rate: float = Field(0.0, ge=0.0, le=0.3)


class ObserverConfig(BaseModel):
    phosphene_size_mean: float = Field(2.0, ge=0.5, le=5.0)
    phosphene_size_std: float = Field(0.5, ge=0.1, le=2.0)
    elongation_factor: float = Field(1.5, ge=1.0, le=3.0)
    spatial_jitter_std: float = Field(0.3, ge=0.0, le=1.0)
    brightness_gamma: float = Field(0.8, ge=0.3, le=2.0)
    adaptation_rate: float = Field(0.1, ge=0.01, le=1.0)
    noise_level: float = Field(0.05, ge=0.0, le=0.3)
