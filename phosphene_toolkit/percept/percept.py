"""Phosphene perception model and renderer."""

import numpy as np
from typing import Dict, Optional
import cv2
from scipy.ndimage import gaussian_filter
from ..task_policy.schemas import ObserverConfig


class PhospheneRenderer:
    """Renders phosphene percepts from stimulation patterns."""

    def __init__(self, config: ObserverConfig, output_size: tuple = (480, 640)):
        self.config = config
        self.output_h, self.output_w = output_size
        self.phosphene_templates = self._create_phosphene_templates()
        self.adaptation_state = None

    def _create_phosphene_templates(self) -> Dict[str, np.ndarray]:
        templates = {}
        size = int(self.config.phosphene_size_mean * 4)
        if size % 2 == 0:
            size += 1
        center = size // 2
        y, x = np.ogrid[:size, :size]
        radius = self.config.phosphene_size_mean
        circular = np.exp(-((x - center)**2 + (y - center)**2) / (2 * radius**2))
        templates['circular'] = circular / circular.max()
        elongation = self.config.elongation_factor
        elongated = np.exp(-((x - center)**2 / (elongation * radius)**2 + (y - center)**2 / radius**2) / 2)
        templates['elongated'] = elongated / elongated.max()
        return templates

    def render_percept(self, stimulation: np.ndarray, frame_time: Optional[float] = None) -> Dict[str, np.ndarray]:
        stim_h, stim_w = stimulation.shape
        percept = np.zeros((self.output_h, self.output_w), dtype=np.float32)
        stim_nonlinear = np.power(stimulation, self.config.brightness_gamma)
        stim_adapted = self._apply_adaptation(stim_nonlinear, frame_time) if frame_time is not None else stim_nonlinear
        for i in range(stim_h):
            for j in range(stim_w):
                intensity = stim_adapted[i, j]
                if intensity < 0.005:
                    continue
                y_pos = int((i / stim_h) * self.output_h)
                x_pos = int((j / stim_w) * self.output_w)
                if self.config.spatial_jitter_std > 0:
                    y_pos = int(y_pos + np.random.normal(0, self.config.spatial_jitter_std))
                    x_pos = int(x_pos + np.random.normal(0, self.config.spatial_jitter_std))
                template_key = np.random.choice(['circular', 'elongated'], p=[0.7, 0.3])
                template = self.phosphene_templates[template_key]
                scaled_template = template * intensity
                size_factor = np.clip(np.random.normal(1.0, self.config.phosphene_size_std / self.config.phosphene_size_mean), 0.5, 2.0)
                if size_factor != 1.0:
                    new_size = int(template.shape[0] * size_factor)
                    if new_size > 0:
                        scaled_template = cv2.resize(scaled_template, (new_size, new_size))
                self._blend_phosphene(percept, scaled_template, y_pos, x_pos)
        if self.config.noise_level > 0:
            noise = np.random.normal(0, self.config.noise_level, percept.shape)
            percept = np.clip(percept + noise, 0, 1)
        percept_blurred = gaussian_filter(percept, sigma=0.5)
        pmin, pmax = percept_blurred.min(), percept_blurred.max()
        if pmax - pmin > 1e-6:
            percept_blurred = (percept_blurred - pmin) / (pmax - pmin)
        elif pmax > 0:
            percept_blurred = percept_blurred / pmax
        percept_uint8 = (np.clip(percept_blurred, 0, 1) * 255).astype(np.uint8)
        return {'percept': percept_blurred, 'percept_uint8': percept_uint8, 'raw_percept': percept,
                'active_phosphenes': int(np.sum(stimulation > 0.01))}

    def _apply_adaptation(self, stimulation: np.ndarray, frame_time: float) -> np.ndarray:
        if self.adaptation_state is None:
            self.adaptation_state = np.zeros_like(stimulation)
        r = self.config.adaptation_rate
        self.adaptation_state = (1 - r) * self.adaptation_state + r * stimulation
        return stimulation / (1 + 0.5 * self.adaptation_state)

    def _blend_phosphene(self, canvas: np.ndarray, phosphene: np.ndarray, center_y: int, center_x: int):
        ph_h, ph_w = phosphene.shape
        canvas_h, canvas_w = canvas.shape
        top, left = center_y - ph_h // 2, center_x - ph_w // 2
        bottom, right = top + ph_h, left + ph_w
        y1, x1 = max(0, top), max(0, left)
        y2, x2 = min(canvas_h, bottom), min(canvas_w, right)
        py1, px1 = y1 - top, x1 - left
        py2, px2 = py1 + (y2 - y1), px1 + (x2 - x1)
        if y2 > y1 and x2 > x1 and py2 > py1 and px2 > px1:
            canvas[y1:y2, x1:x2] = np.clip(canvas[y1:y2, x1:x2] + phosphene[py1:py2, px1:px2], 0, 1)


class ObserverModel:
    """Complete observer model for phosphene perception."""

    def __init__(self, config: ObserverConfig):
        self.config = config
        self.renderer = PhospheneRenderer(config)

    def predict_percept(self, stimulation: np.ndarray, **kwargs) -> Dict:
        return self.renderer.render_percept(stimulation, **kwargs)

    def reset_adaptation(self):
        self.renderer.adaptation_state = None
