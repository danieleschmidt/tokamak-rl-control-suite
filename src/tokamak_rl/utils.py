"""
Utility functions and helper classes for tokamak RL control.

This module provides common utilities, data processing functions,
and helper classes used throughout the tokamak control system.
"""

import numpy as np
import torch
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
import json
import yaml
import logging
import time
import random
from pathlib import Path
from dataclasses import dataclass, asdict
import hashlib
from contextlib import contextmanager

from .physics import TokamakConfig, PlasmaState


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass 
class ValidationResult:
    """Result of input validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    
    def __bool__(self) -> bool:
        return self.is_valid


class ConfigValidator:
    """Validates tokamak and training configurations."""
    
    @staticmethod
    def validate_tokamak_config(config: TokamakConfig) -> ValidationResult:
        """Validate tokamak configuration parameters."""
        errors = []
        warnings = []
        
        # Physical parameter validation
        if config.major_radius <= 0:
            errors.append("Major radius must be positive")
            
        if config.minor_radius <= 0:
            errors.append("Minor radius must be positive")
            
        if config.minor_radius >= config.major_radius:
            errors.append("Minor radius must be less than major radius")
            
        if config.aspect_ratio < 1.1:
            warnings.append("Very low aspect ratio may be unrealistic")
            
        if config.toroidal_field <= 0:
            errors.append("Toroidal field must be positive")
            
        if config.plasma_current <= 0:
            errors.append("Plasma current must be positive")
            
        # Plasma parameter validation
        if config.elongation < 1.0:
            errors.append("Elongation must be >= 1.0")
            
        if config.elongation > 3.0:
            warnings.append("Very high elongation may be unstable")
            
        if not -1.0 <= config.triangularity <= 1.0:
            errors.append("Triangularity must be between -1.0 and 1.0")
            
        if config.beta_n <= 0:
            errors.append("Normalized beta must be positive")
            
        if config.beta_n > 5.0:
            warnings.append("Very high beta_n may lead to disruptions")
            
        if config.q95 < 2.0:
            warnings.append("Low q95 may be unstable")
            
        # Control parameter validation
        if config.control_frequency <= 0:
            errors.append("Control frequency must be positive")
            
        if config.control_frequency > 10000:
            warnings.append("Very high control frequency may be computationally expensive")
            
        if config.simulation_timestep <= 0:
            errors.append("Simulation timestep must be positive")
            
        if config.simulation_timestep > 1.0 / config.control_frequency:
            warnings.append("Simulation timestep larger than control period")
            
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
        
    @staticmethod
    def validate_action(action: np.ndarray, action_space) -> ValidationResult:
        """Validate RL action against action space."""
        errors = []
        warnings = []
        
        # Check dimensions
        if len(action) != action_space.shape[0]:
            errors.append(f"Action dimension mismatch: {len(action)} vs {action_space.shape[0]}")
            
        # Check bounds
        if hasattr(action_space, 'low') and hasattr(action_space, 'high'):
            low_violations = action < action_space.low
            high_violations = action > action_space.high
            
            if np.any(low_violations):
                violations = np.where(low_violations)[0]
                errors.append(f"Action values below lower bound at indices: {violations.tolist()}")
                
            if np.any(high_violations):
                violations = np.where(high_violations)[0]
                errors.append(f"Action values above upper bound at indices: {violations.tolist()}")
                
        # Check for NaN or infinite values
        if np.any(np.isnan(action)):
            errors.append("Action contains NaN values")
            
        if np.any(np.isinf(action)):
            errors.append("Action contains infinite values")
            
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )


class StateProcessor:
    """Processes and normalizes plasma state observations."""
    
    def __init__(self, config: TokamakConfig):
        self.config = config
        self._setup_normalization_params()
        
    def _setup_normalization_params(self) -> None:
        """Setup normalization parameters based on tokamak configuration."""
        self.normalization = {
            'plasma_current': {'mean': self.config.plasma_current, 'std': self.config.plasma_current * 0.2},
            'plasma_beta': {'mean': 0.025, 'std': 0.01},
            'q_profile': {'mean': 3.0, 'std': 2.0},
            'shape_error': {'mean': 0.0, 'std': 2.0},
            'elongation': {'mean': self.config.elongation, 'std': 0.2},
            'triangularity': {'mean': self.config.triangularity, 'std': 0.1},
            'pf_currents': {'mean': 0.0, 'std': 1.0},
            'density': {'mean': 1e20, 'std': 5e19},
            'temperature': {'mean': 10.0, 'std': 5.0}
        }
        
    def normalize_observation(self, observation: np.ndarray) -> np.ndarray:
        """Normalize observation for RL training."""
        # Create normalized copy
        normalized = observation.copy()
        
        # Apply component-wise normalization
        # This is a simplified version - production would use proper statistics
        normalized = (normalized - 0.5) / 0.5  # Simple [-1, 1] normalization
        
        return normalized
        
    def denormalize_observation(self, normalized_obs: np.ndarray) -> np.ndarray:
        """Convert normalized observation back to physical units."""
        return normalized_obs * 0.5 + 0.5
        
    def process_state_for_analysis(self, state: PlasmaState) -> Dict[str, Any]:
        """Process plasma state for analysis and visualization."""
        return {
            'basic_parameters': {
                'q_min': state.q_min,
                'plasma_beta': state.plasma_beta,
                'shape_error': state.shape_error,
                'elongation': state.elongation,
                'triangularity': state.triangularity
            },
            'profiles': {
                'q_profile': state.q_profile.tolist(),
                'pressure_profile': state.pressure_profile.tolist(),
                'density_profile': state.density_profile.tolist(),
                'temperature_profile': state.temperature_profile.tolist()
            },
            'control': {
                'pf_coil_currents': state.pf_coil_currents.tolist(),
                'plasma_current': state.plasma_current
            },
            'safety': {
                'disruption_probability': state.disruption_probability,
                'q_min': state.q_min
            }
        }


class PerformanceMetrics:
    """Calculates and tracks performance metrics."""
    
    def __init__(self):
        self.reset()
        
    def reset(self) -> None:
        """Reset all metrics."""
        self.shape_errors = []
        self.q_mins = []
        self.disruption_risks = []
        self.control_efforts = []
        self.safety_violations = []
        
    def update(self, state: PlasmaState, action: np.ndarray, 
              safety_info: Dict[str, Any]) -> None:
        """Update metrics with new step data."""
        self.shape_errors.append(state.shape_error)
        self.q_mins.append(state.q_min)
        self.disruption_risks.append(state.disruption_probability)
        self.control_efforts.append(np.sum(action[:6]**2))  # PF coil effort
        self.safety_violations.append(len(safety_info.get('violations', [])))
        
    def compute_statistics(self) -> Dict[str, float]:
        """Compute statistical summary of metrics."""
        if not self.shape_errors:
            return {}
            
        return {
            'shape_error_mean': np.mean(self.shape_errors),
            'shape_error_std': np.std(self.shape_errors),
            'shape_error_max': np.max(self.shape_errors),
            'q_min_mean': np.mean(self.q_mins),
            'q_min_min': np.min(self.q_mins),
            'disruption_risk_mean': np.mean(self.disruption_risks),
            'disruption_risk_max': np.max(self.disruption_risks),
            'control_effort_mean': np.mean(self.control_efforts),
            'safety_violations_total': np.sum(self.safety_violations),
            'safety_violation_rate': np.mean([v > 0 for v in self.safety_violations])
        }


class ExperimentTracker:
    """Tracks and manages experiment metadata."""
    
    def __init__(self, experiment_name: str, output_dir: str = "./experiments"):
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.start_time = time.time()
        self.metadata = {
            'experiment_name': experiment_name,
            'start_time': self.start_time,
            'parameters': {},
            'metrics': {},
            'checkpoints': []
        }
        
    def log_parameters(self, parameters: Dict[str, Any]) -> None:
        """Log experiment parameters."""
        self.metadata['parameters'].update(parameters)
        self._save_metadata()
        
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log experiment metrics."""
        timestamp = time.time()
        metric_entry = {
            'timestamp': timestamp,
            'step': step,
            'metrics': metrics
        }
        
        if 'history' not in self.metadata['metrics']:
            self.metadata['metrics']['history'] = []
            
        self.metadata['metrics']['history'].append(metric_entry)
        self.metadata['metrics']['latest'] = metrics
        self._save_metadata()
        
    def log_checkpoint(self, checkpoint_path: str, metrics: Dict[str, Any]) -> None:
        """Log model checkpoint."""
        checkpoint_entry = {
            'timestamp': time.time(),
            'path': checkpoint_path,
            'metrics': metrics
        }
        self.metadata['checkpoints'].append(checkpoint_entry)
        self._save_metadata()
        
    def _save_metadata(self) -> None:
        """Save metadata to file."""
        metadata_file = self.output_dir / f"{self.experiment_name}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
            
    def finalize(self) -> None:
        """Finalize experiment tracking."""
        self.metadata['end_time'] = time.time()
        self.metadata['duration'] = self.metadata['end_time'] - self.start_time
        self._save_metadata()


@contextmanager
def timing_context(name: str = "operation"):
    """Context manager for timing operations."""
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        logger.info(f"{name} took {end_time - start_time:.3f} seconds")


def set_random_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_config_file(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file."""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    with open(config_path, 'r') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            return yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """Save configuration to file."""
    output_path = Path(output_path)
    
    with open(output_path, 'w') as f:
        if output_path.suffix.lower() in ['.yaml', '.yml']:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        elif output_path.suffix.lower() == '.json':
            json.dump(config, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported output format: {output_path.suffix}")


def compute_hash(data: Union[str, bytes, Dict, List]) -> str:
    """Compute SHA-256 hash of data."""
    if isinstance(data, (dict, list)):
        data = json.dumps(data, sort_keys=True)
    if isinstance(data, str):
        data = data.encode('utf-8')
    return hashlib.sha256(data).hexdigest()


def moving_average(data: List[float], window_size: int) -> List[float]:
    """Compute moving average of data."""
    if len(data) < window_size:
        return data
        
    averaged = []
    for i in range(len(data)):
        start_idx = max(0, i - window_size + 1)
        window = data[start_idx:i+1]
        averaged.append(sum(window) / len(window))
        
    return averaged


def interpolate_profile(profile: np.ndarray, target_size: int) -> np.ndarray:
    """Interpolate 1D profile to target size."""
    if len(profile) == target_size:
        return profile
        
    old_indices = np.linspace(0, 1, len(profile))
    new_indices = np.linspace(0, 1, target_size)
    
    return np.interp(new_indices, old_indices, profile)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division with default value for zero denominator."""
    return numerator / denominator if denominator != 0 else default


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value to range [min_val, max_val]."""
    return max(min_val, min(value, max_val))


def format_time(seconds: float) -> str:
    """Format time duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{int(minutes)}m {seconds:.1f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{int(hours)}h {int(minutes)}m"


def create_experiment_tracker(name: str, output_dir: str = "./experiments") -> ExperimentTracker:
    """Factory function to create experiment tracker."""
    return ExperimentTracker(name, output_dir)