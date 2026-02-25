"""Main PhospheneEncoderTool class."""

import time
import numpy as np
from typing import Dict, Optional, Any
import yaml
import logging

from .task_policy.schemas import TaskParams, DeviceConfig, ObserverConfig
from .task_policy.policy_rules import RuleBasedTaskPolicy
from .perception.segmentation import DeepLabV3Segmentation
from .perception.saliency import SpectralResidualSaliency
from .perception.motion import FarnebackMotionDetector
from .fusion.fusion import TaskConditionalFusion
from .fusion.allocation import BandwidthAllocator
from .temporal.temporal import TemporalStabilizer
from .device.device import DeviceModel
from .percept.percept import ObserverModel
from .utils.logging import StructuredLogger


class PhospheneEncoderTool:
    """Main tool for task-conditioned phosphene encoding."""
    
    def __init__(self, config_path: Optional[str] = None, config: Optional[Dict] = None):
        """Initialize with configuration."""
        if config is not None:
            self.config = config
        else:
            self.config = self._load_config(config_path)
        self.logger = StructuredLogger("phosphene_encoder")
        
        # Initialize components
        try:
            self._init_components()
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}", extra={'error': str(e)})
            raise
        
        # State
        self.frame_count = 0
        self.start_time = time.time()
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from file or use defaults."""
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load config from {config_path}: {e}, using defaults")
                config = self._get_default_config()
        else:
            config = self._get_default_config()
        
        return config
    
    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            'device': {
                'grid_size': [60, 60],
                'amplitude_levels': 256,
                'max_amplitude_per_electrode': 1.0,
                'global_power_cap': 100.0,
                'spatial_spread_sigma': 1.2,
                'temporal_freq_hz': 20.0,
                'duty_cycle': 0.1,
                'dropout_rate': 0.0
            },
            'observer': {
                'phosphene_size_mean': 2.0,
                'phosphene_size_std': 0.5,
                'elongation_factor': 1.5,
                'spatial_jitter_std': 0.3,
                'brightness_gamma': 0.8,
                'adaptation_rate': 0.1,
                'noise_level': 0.05
            },
            'perception': {
                'segmentation_model': 'deeplabv3_resnet50',
                'input_size': [480, 640],
                'fast_mode': False
            },
            'fusion': {
                'allocation_strategy': 'foveated',
                'max_active_phosphenes': 200
            }
        }
    
    def _init_components(self):
        """Initialize all processing components."""
        print("Initializing PhospheneEncoderTool components...")
        
        # Parse configs
        device_config = DeviceConfig(**self.config['device'])
        observer_config = ObserverConfig(**self.config['observer'])
        
        # Core components
        print("  → Loading task policy...")
        self.task_policy = RuleBasedTaskPolicy()
        
        print("  → Loading segmentation model (this may take 1-2 minutes on first run)...")
        self.segmentation = DeepLabV3Segmentation(
            model_name=self.config['perception']['segmentation_model'],
            input_size=tuple(self.config['perception']['input_size'])
        )
        
        print("  → Initializing saliency detector...")
        self.saliency = SpectralResidualSaliency()
        
        print("  → Initializing motion detector...")
        self.motion_detector = FarnebackMotionDetector()
        
        print("  → Setting up fusion layer...")
        self.fusion = TaskConditionalFusion()
        
        print("  → Setting up bandwidth allocator...")
        self.allocator = BandwidthAllocator(
            strategy=self.config['fusion']['allocation_strategy'],
            max_active=self.config['fusion']['max_active_phosphenes']
        )
        
        print("  → Initializing temporal stabilizer...")
        self.temporal_stabilizer = TemporalStabilizer()
        
        print("  → Setting up device model...")
        self.device_model = DeviceModel(device_config)
        
        print("  → Setting up observer model...")
        self.observer_model = ObserverModel(observer_config)
        
        print("✓ All components initialized successfully!")
        
        self.logger.info("PhospheneEncoderTool initialized", extra={
            'device_grid': device_config.grid_size,
            'components': ['segmentation', 'saliency', 'motion', 'fusion', 'device', 'observer']
        })
    
    def process_frame(self, frame: np.ndarray, task: str, 
                     return_intermediates: bool = False) -> Dict[str, Any]:
        """
        Process single frame through complete pipeline.
        
        Args:
            frame: Input frame (H, W, 3) uint8
            task: Task description string
            return_intermediates: Whether to return intermediate results
            
        Returns:
            Dict with stimulation_plan, percept, timings, and optionally intermediates
        """
        timings = {}
        intermediates = {} if return_intermediates else None
        
        try:
            # Parse task
            t0 = time.time()
            task_params = self.task_policy.parse_task(task)
            timings['task_parsing'] = (time.time() - t0) * 1000
            
            # Perception stage
            t0 = time.time()
            perception_results = self._run_perception(frame)
            timings['perception'] = (time.time() - t0) * 1000
            
            if return_intermediates:
                intermediates.update(perception_results)
            
            # Fusion stage
            t0 = time.time()
            class_names = perception_results.get('class_names')
            fusion_result = self.fusion.fuse_maps(
                segmentation=perception_results['segmentation'],
                saliency=perception_results['saliency'],
                motion=perception_results['motion_magnitude'],
                task_params=task_params,
                class_names=class_names
            )
            timings['fusion'] = (time.time() - t0) * 1000
            
            if return_intermediates:
                intermediates['fusion_map'] = fusion_result
            
            # Temporal stabilization
            t0 = time.time()
            stabilized = self.temporal_stabilizer.stabilize(
                fusion_result,
                motion_magnitude=perception_results.get('motion_magnitude'),
                frame_time=time.time() - self.start_time
            )
            timings['temporal'] = (time.time() - t0) * 1000
        
            # Bandwidth allocation
            t0 = time.time()
            allocated = self.allocator.allocate(stabilized, task_params)
            timings['allocation'] = (time.time() - t0) * 1000
            
            if return_intermediates:
                intermediates['allocated_map'] = allocated
            
            # Device constraints
            t0 = time.time()
            device_result = self.device_model.process_stimulation(allocated)
            timings['device_constraints'] = (time.time() - t0) * 1000
            
            # Phosphene rendering
            t0 = time.time()
            percept_result = self.observer_model.predict_percept(
                device_result['constrained_stim'],
                frame_time=time.time() - self.start_time
            )
            timings['rendering'] = (time.time() - t0) * 1000
            
            # Total timing
            timings['total'] = sum([v for k, v in timings.items() if k != 'total'])
        
            # Log frame processing
            self.frame_count += 1
            if self.frame_count % 30 == 0:  # Log every 30 frames
                self.logger.info("Frame processed", extra={
                    'frame_count': self.frame_count,
                    'total_time_ms': timings['total'],
                    'fps': 1000.0 / timings['total'] if timings['total'] > 0 else 0,
                    'task': task
                })
            
            # Prepare output
            output = {
                'stimulation_plan': device_result['constrained_stim'],
                'percept': percept_result['percept_uint8'],
                'timings': timings,
                'device_diagnostics': device_result['diagnostics'],
                'percept_info': {
                    'active_phosphenes': percept_result['active_phosphenes']
                }
            }
            
            if return_intermediates:
                output['intermediates'] = intermediates
            
            return output
            
        except Exception as e:
            self.logger.error(f"Error processing frame: {e}", extra={'error': str(e), 'frame_count': self.frame_count})
            raise
    
    def _run_perception(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        """Run all perception modules."""
        results = {}
        
        # Segmentation
        seg_result = self.segmentation.segment(frame)
        results['segmentation'] = seg_result['segmentation']
        results['class_probs'] = seg_result.get('class_probs')
        results['class_names'] = seg_result.get('class_names')
        
        # Saliency
        saliency = self.saliency.compute_saliency(frame)
        results['saliency'] = saliency
        
        # Motion
        motion_result = self.motion_detector.detect_motion(frame)
        results['motion_magnitude'] = motion_result['magnitude']
        results['motion_direction'] = motion_result['direction']
        
        return results
    
    def reset_state(self):
        """Reset temporal state."""
        self.temporal_stabilizer.reset()
        self.observer_model.reset_adaptation()
        self.motion_detector.reset()
        self.frame_count = 0
        self.start_time = time.time()
        
        self.logger.info("State reset")
    
    def get_stats(self) -> Dict:
        """Get processing statistics."""
        return {
            'frames_processed': self.frame_count,
            'uptime_seconds': time.time() - self.start_time,
            'device_config': (lambda c: c.model_dump() if hasattr(c, 'model_dump') else c.dict())(self.device_model.config),
            'observer_config': (lambda c: c.model_dump() if hasattr(c, 'model_dump') else c.dict())(self.observer_model.config)
        }