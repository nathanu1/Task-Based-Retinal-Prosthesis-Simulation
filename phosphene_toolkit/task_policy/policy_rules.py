"""Rule-based task policy implementation."""

import re
from typing import Dict, List
from .policy_base import TaskPolicy
from .schemas import TaskParams, RegionPriority, FoveaConfig, TemporalConfig, SafetyConfig


class RuleBasedTaskPolicy(TaskPolicy):
    """Rule-based task parameter generation."""

    def __init__(self):
        self.task_presets = self._load_task_presets()
        self.keyword_mappings = self._load_keyword_mappings()

    def _copy_params(self, params: TaskParams) -> TaskParams:
        try:
            return params.model_copy(deep=True)
        except AttributeError:
            return params.copy(deep=True)

    def _load_task_presets(self) -> Dict[str, TaskParams]:
        presets = {}
        presets['navigation'] = TaskParams(
            task_type='navigation',
            region_priorities=[
                RegionPriority(label='person', weight=3.0),
                RegionPriority(label='car', weight=2.5),
                RegionPriority(label='bicycle', weight=2.0),
                RegionPriority(label='chair', weight=1.5),
                RegionPriority(label='door', weight=2.0),
            ],
            edge_weight=2.0,
            floor_weight=1.0,
            obstacle_weight=3.0,
            motion_weight=2.5,
            fovea=FoveaConfig(mode='center', radius=0.4, strength=1.5),
            safety=SafetyConfig(near_field_boost=2.0, hazard_labels=['person', 'car', 'bicycle', 'motorcycle', 'bus', 'truck'])
        )
        presets['grasping'] = TaskParams(
            task_type='grasping',
            region_priorities=[
                RegionPriority(label='cup', weight=4.0),
                RegionPriority(label='bottle', weight=4.0),
                RegionPriority(label='book', weight=3.0),
                RegionPriority(label='cell phone', weight=3.5),
                RegionPriority(label='remote', weight=3.0),
                RegionPriority(label='keys', weight=4.0),
            ],
            edge_weight=3.0,
            floor_weight=0.5,
            obstacle_weight=1.0,
            motion_weight=1.0,
            fovea=FoveaConfig(mode='task_driven', radius=0.3, strength=3.0),
            temporal=TemporalConfig(base_smoothing=0.8, persistence_half_life=1.0)
        )
        presets['avoidance'] = TaskParams(
            task_type='avoidance',
            region_priorities=[
                RegionPriority(label='person', weight=5.0),
                RegionPriority(label='car', weight=4.0),
                RegionPriority(label='bicycle', weight=3.5),
                RegionPriority(label='motorcycle', weight=4.0),
            ],
            edge_weight=1.5,
            floor_weight=0.8,
            obstacle_weight=4.0,
            motion_weight=4.0,
            fovea=FoveaConfig(mode='center', radius=0.5, strength=2.0),
            safety=SafetyConfig(near_field_boost=3.0, hazard_labels=['person', 'car', 'bicycle', 'motorcycle', 'bus', 'truck', 'train'])
        )
        presets['exploration'] = TaskParams(
            task_type='exploration',
            region_priorities=[
                RegionPriority(label='door', weight=2.0),
                RegionPriority(label='window', weight=1.5),
                RegionPriority(label='sign', weight=2.5),
                RegionPriority(label='text', weight=2.0),
            ],
            edge_weight=1.5,
            floor_weight=1.2,
            obstacle_weight=1.5,
            motion_weight=1.8,
            fovea=FoveaConfig(mode='disabled'),
            temporal=TemporalConfig(base_smoothing=0.6, hysteresis=0.2)
        )
        return presets

    def _load_keyword_mappings(self) -> Dict:
        return {
            'navigate': {'task_type': 'navigation'},
            'walk': {'task_type': 'navigation'},
            'move': {'task_type': 'navigation'},
            'go': {'task_type': 'navigation'},
            'grab': {'task_type': 'grasping'},
            'grasp': {'task_type': 'grasping'},
            'pick': {'task_type': 'grasping'},
            'take': {'task_type': 'grasping'},
            'reach': {'task_type': 'grasping'},
            'avoid': {'task_type': 'avoidance'},
            'dodge': {'task_type': 'avoidance'},
            'explore': {'task_type': 'exploration'},
            'look': {'task_type': 'exploration'},
            'search': {'task_type': 'exploration'},
            'find': {'task_type': 'exploration'},
            'chair': {'add_priority': ('chair', 3.0)},
            'table': {'add_priority': ('dining table', 2.5)},
            'door': {'add_priority': ('door', 3.0)},
            'person': {'add_priority': ('person', 4.0)},
            'car': {'add_priority': ('car', 3.0)},
            'cup': {'add_priority': ('cup', 4.0)},
            'bottle': {'add_priority': ('bottle', 4.0)},
            'phone': {'add_priority': ('cell phone', 3.5)},
        }

    def parse_task(self, task_description: str) -> TaskParams:
        task_lower = task_description.lower().strip()
        base_params = self._copy_params(self.task_presets['navigation'])
        detected_type = self._detect_task_type(task_lower)
        if detected_type in self.task_presets:
            base_params = self._copy_params(self.task_presets[detected_type])
        base_params = self._apply_keyword_modifications(base_params, task_lower)
        base_params = self._extract_object_priorities(base_params, task_lower)
        base_params = self._apply_contextual_modifiers(base_params, task_lower)
        return base_params

    def _detect_task_type(self, task_lower: str) -> str:
        for keyword, mapping in self.keyword_mappings.items():
            if 'task_type' in mapping and keyword in task_lower:
                return mapping['task_type']
        if any(w in task_lower for w in ['navigate', 'walk', 'move']):
            return 'navigation'
        elif any(w in task_lower for w in ['grab', 'pick', 'grasp', 'take']):
            return 'grasping'
        elif any(w in task_lower for w in ['avoid', 'dodge']):
            return 'avoidance'
        elif any(w in task_lower for w in ['explore', 'find', 'search']):
            return 'exploration'
        return 'navigation'

    def _apply_keyword_modifications(self, params: TaskParams, task_lower: str) -> TaskParams:
        for keyword, mapping in self.keyword_mappings.items():
            if keyword in task_lower:
                if 'edge_weight' in mapping:
                    params.edge_weight = mapping['edge_weight']
                if 'motion_weight' in mapping:
                    params.motion_weight = mapping['motion_weight']
        return params

    def _extract_object_priorities(self, params: TaskParams, task_lower: str) -> TaskParams:
        for keyword, mapping in self.keyword_mappings.items():
            if 'add_priority' in mapping and keyword in task_lower:
                label, weight = mapping['add_priority']
                found = False
                for i, p in enumerate(params.region_priorities):
                    if p.label == label:
                        params.region_priorities[i] = RegionPriority(label=label, weight=max(p.weight, weight))
                        found = True
                        break
                if not found:
                    params.region_priorities.append(RegionPriority(label=label, weight=weight))
        return params

    def _apply_contextual_modifiers(self, params: TaskParams, task_lower: str) -> TaskParams:
        if any(w in task_lower for w in ['indoor', 'inside', 'room', 'kitchen']):
            params.motion_weight *= 0.8
            params.fovea.radius *= 0.9
        if any(w in task_lower for w in ['outdoor', 'outside', 'street']):
            params.motion_weight *= 1.2
            params.safety.near_field_boost *= 1.3
        return params

    def get_available_presets(self) -> List[str]:
        return list(self.task_presets.keys())

    def get_preset(self, preset_name: str) -> TaskParams:
        if preset_name not in self.task_presets:
            raise ValueError(f"Unknown preset: {preset_name}")
        return self._copy_params(self.task_presets[preset_name])
