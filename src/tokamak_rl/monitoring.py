"""
Monitoring and visualization components for tokamak plasma control.

This module provides real-time monitoring, logging, TensorBoard integration,
and visualization tools for RL training and plasma state analysis.
"""

try:
    import numpy as np
except ImportError:
    # Fallback for environments without numpy
    print("NumPy not available - using basic Python implementations")
    import math
    
    class np:
        @staticmethod
        def array(x):
            return list(x) if hasattr(x, '__iter__') else [x]
        
        @staticmethod
        def mean(arr):
            return sum(arr) / len(arr)
        
        @staticmethod
        def std(arr):
            mean_val = sum(arr) / len(arr)
            return math.sqrt(sum((x - mean_val)**2 for x in arr) / len(arr))
        
        @staticmethod
        def zeros(shape):
            if isinstance(shape, int):
                return [0.0] * shape
            return [[0.0] * shape[1] for _ in range(shape[0])]
        
        float32 = float
        ndarray = list
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
from typing import Dict, Any, Optional, List, Tuple
import json
import time
from datetime import datetime
from pathlib import Path
import warnings
from dataclasses import dataclass
from .physics import PlasmaState


@dataclass
class AlertThresholds:
    """Alert thresholds for plasma monitoring."""
    q_min: float = 1.5
    shape_error: float = 5.0  # cm
    stored_energy: float = 500  # MJ
    disruption_probability: float = 0.1
    beta_limit: float = 0.04
    density_limit: float = 1.2e20  # m^-3


class PlasmaMonitor:
    """Real-time plasma monitoring and alerting system."""
    
    def __init__(self, log_dir: str = "./plasma_logs",
                 alert_thresholds: Optional[AlertThresholds] = None,
                 enable_alerts: bool = True):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.thresholds = alert_thresholds or AlertThresholds()
        self.enable_alerts = enable_alerts
        
        # Initialize logging
        self.log_file = self.log_dir / f"plasma_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        self.alert_file = self.log_dir / f"alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        
        # Statistics tracking
        self.step_count = 0
        self.episode_count = 0
        self.alert_count = 0
        
        # Performance tracking
        self.performance_buffer = {
            'shape_errors': [],
            'q_mins': [],
            'disruption_risks': [],
            'timestamps': []
        }
        self.buffer_size = 1000
        
    def log_step(self, state: PlasmaState, action: np.ndarray, 
                reward: float, info: Dict[str, Any]) -> None:
        """Log a single environment step."""
        timestamp = time.time()
        
        # Create log entry
        log_entry = {
            'timestamp': timestamp,
            'step': self.step_count,
            'episode': self.episode_count,
            'plasma_state': {
                'q_min': state.q_min,
                'plasma_beta': state.plasma_beta,
                'shape_error': state.shape_error,
                'elongation': state.elongation,
                'triangularity': state.triangularity,
                'disruption_probability': state.disruption_probability,
                'plasma_current': state.plasma_current,
                'pf_coil_currents': state.pf_coil_currents.tolist()
            },
            'action': action.tolist(),
            'reward': reward,
            'info': self._sanitize_info(info)
        }
        
        # Write to log file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
            
        # Update performance buffer
        self._update_performance_buffer(state, timestamp)
        
        # Check for alerts
        if self.enable_alerts:
            self._check_alerts(state, log_entry)
            
        self.step_count += 1
        
    def log_episode_end(self, episode_metrics: Dict[str, float]) -> None:
        """Log episode completion."""
        timestamp = time.time()
        
        episode_entry = {
            'timestamp': timestamp,
            'episode': self.episode_count,
            'type': 'episode_end',
            'metrics': episode_metrics
        }
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(episode_entry) + '\n')
            
        self.episode_count += 1
        
    def _sanitize_info(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize info dict for JSON serialization."""
        sanitized = {}
        for key, value in info.items():
            if isinstance(value, (dict, list, str, int, float, bool, type(None))):
                sanitized[key] = value
            elif isinstance(value, np.ndarray):
                sanitized[key] = value.tolist()
            else:
                sanitized[key] = str(value)
        return sanitized
        
    def _update_performance_buffer(self, state: PlasmaState, timestamp: float) -> None:
        """Update performance tracking buffer."""
        self.performance_buffer['shape_errors'].append(state.shape_error)
        self.performance_buffer['q_mins'].append(state.q_min)
        self.performance_buffer['disruption_risks'].append(state.disruption_probability)
        self.performance_buffer['timestamps'].append(timestamp)
        
        # Maintain buffer size
        for key in self.performance_buffer:
            if len(self.performance_buffer[key]) > self.buffer_size:
                self.performance_buffer[key].pop(0)
                
    def _check_alerts(self, state: PlasmaState, log_entry: Dict[str, Any]) -> None:
        """Check for alert conditions and log alerts."""
        alerts = []
        
        # Q-min alert
        if state.q_min < self.thresholds.q_min:
            alerts.append({
                'type': 'low_q_min',
                'severity': 'critical' if state.q_min < 1.0 else 'warning',
                'value': state.q_min,
                'threshold': self.thresholds.q_min,
                'message': f"Safety factor below threshold: {state.q_min:.2f} < {self.thresholds.q_min}"
            })
            
        # Shape error alert
        if state.shape_error > self.thresholds.shape_error:
            alerts.append({
                'type': 'high_shape_error',
                'severity': 'warning',
                'value': state.shape_error,
                'threshold': self.thresholds.shape_error,
                'message': f"Shape error above threshold: {state.shape_error:.2f} cm > {self.thresholds.shape_error} cm"
            })
            
        # Disruption risk alert
        if state.disruption_probability > self.thresholds.disruption_probability:
            severity = 'critical' if state.disruption_probability > 0.5 else 'warning'
            alerts.append({
                'type': 'high_disruption_risk',
                'severity': severity,
                'value': state.disruption_probability,
                'threshold': self.thresholds.disruption_probability,
                'message': f"Disruption risk above threshold: {state.disruption_probability:.3f} > {self.thresholds.disruption_probability}"
            })
            
        # Beta limit alert
        if state.plasma_beta > self.thresholds.beta_limit:
            alerts.append({
                'type': 'high_beta',
                'severity': 'warning',
                'value': state.plasma_beta,
                'threshold': self.thresholds.beta_limit,
                'message': f"Beta above threshold: {state.plasma_beta:.3f} > {self.thresholds.beta_limit}"
            })
            
        # Log alerts
        for alert in alerts:
            alert_entry = {
                'timestamp': log_entry['timestamp'],
                'step': log_entry['step'],
                'episode': log_entry['episode'],
                'alert': alert
            }
            
            with open(self.alert_file, 'a') as f:
                f.write(json.dumps(alert_entry) + '\n')
                
            # Print critical alerts
            if alert['severity'] == 'critical':
                print(f"🚨 CRITICAL ALERT: {alert['message']}")
                
            self.alert_count += 1
            
    def get_statistics(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        if not self.performance_buffer['shape_errors']:
            return {}
            
        return {
            'total_steps': self.step_count,
            'total_episodes': self.episode_count,
            'total_alerts': self.alert_count,
            'recent_performance': {
                'mean_shape_error': np.mean(self.performance_buffer['shape_errors']),
                'min_q_factor': np.min(self.performance_buffer['q_mins']),
                'max_disruption_risk': np.max(self.performance_buffer['disruption_risks']),
                'data_points': len(self.performance_buffer['shape_errors'])
            }
        }
        
    def generate_report(self) -> str:
        """Generate monitoring report."""
        stats = self.get_statistics()
        
        if not stats:
            return "No monitoring data available."
            
        report = f"""
Plasma Monitoring Report
========================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Overall Statistics:
- Total Steps: {stats['total_steps']}
- Total Episodes: {stats['total_episodes']}
- Total Alerts: {stats['total_alerts']}

Recent Performance (last {stats['recent_performance']['data_points']} steps):
- Mean Shape Error: {stats['recent_performance']['mean_shape_error']:.2f} cm
- Minimum Q-factor: {stats['recent_performance']['min_q_factor']:.2f}
- Maximum Disruption Risk: {stats['recent_performance']['max_disruption_risk']:.3f}

Log Files:
- Plasma Log: {self.log_file}
- Alert Log: {self.alert_file}
"""
        return report


class PlasmaRenderer:
    """Real-time plasma visualization renderer."""
    
    def __init__(self, resolution: Tuple[int, int] = (800, 600),
                 style: str = 'plasma'):
        self.resolution = resolution
        self.style = style
        
        # Set up matplotlib style
        if style == 'plasma':
            plt.style.use('dark_background')
            
        self.fig = None
        self.axes = None
        
    def render(self, flux_surfaces: Optional[np.ndarray] = None,
              q_profile: Optional[np.ndarray] = None,
              pressure: Optional[np.ndarray] = None,
              state: Optional[PlasmaState] = None) -> np.ndarray:
        """
        Render plasma state visualization.
        
        Args:
            flux_surfaces: 2D array of flux surface contours
            q_profile: 1D array of safety factor profile
            pressure: 1D array of pressure profile
            state: PlasmaState object with full state information
            
        Returns:
            RGB image array of rendered plasma
        """
        if self.fig is None:
            self._setup_figure()
            
        # Clear previous plots
        for ax in self.axes.flat:
            ax.clear()
            
        if state is not None:
            self._render_from_state(state)
        else:
            self._render_from_arrays(flux_surfaces, q_profile, pressure)
            
        # Convert to RGB array
        self.fig.canvas.draw()
        buf = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        
        return buf
        
    def _setup_figure(self) -> None:
        """Setup matplotlib figure and axes."""
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle('Tokamak Plasma State', fontsize=16, color='white')
        
        # Configure subplots
        titles = ['Flux Surfaces', 'Safety Factor Profile', 
                 'Pressure Profile', 'Temperature Profile']
        
        for ax, title in zip(self.axes.flat, titles):
            ax.set_title(title, color='white')
            ax.tick_params(colors='white')
            
    def _render_from_state(self, state: PlasmaState) -> None:
        """Render visualization from PlasmaState object."""
        # Flux surfaces (simplified circular representation)
        ax1 = self.axes[0, 0]
        theta = np.linspace(0, 2*np.pi, 100)
        for i, psi in enumerate(state.psi_profile[::10]):
            r = 0.5 + psi * 0.4  # Simplified radial coordinate
            x = r * np.cos(theta) * state.elongation
            y = r * np.sin(theta)
            ax1.plot(x, y, alpha=0.7, color=plt.cm.plasma(psi))
        ax1.set_xlim(-1.5, 1.5)
        ax1.set_ylim(-1.2, 1.2)
        ax1.set_aspect('equal')
        ax1.set_xlabel('R (m)', color='white')
        ax1.set_ylabel('Z (m)', color='white')
        
        # Safety factor profile
        ax2 = self.axes[0, 1]
        ax2.plot(state.psi_profile, state.q_profile, 'cyan', linewidth=2)
        ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='q=1')
        ax2.axhline(y=2.0, color='yellow', linestyle='--', alpha=0.7, label='q=2')
        ax2.set_xlabel('ψ (normalized)', color='white')
        ax2.set_ylabel('Safety Factor q', color='white')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Pressure profile
        ax3 = self.axes[1, 0]
        ax3.plot(state.psi_profile, state.pressure_profile, 'orange', linewidth=2)
        ax3.set_xlabel('ψ (normalized)', color='white')
        ax3.set_ylabel('Pressure (normalized)', color='white')
        ax3.grid(True, alpha=0.3)
        
        # Temperature profile
        ax4 = self.axes[1, 1]
        ax4.plot(state.psi_profile, state.temperature_profile, 'red', linewidth=2)
        ax4.set_xlabel('ψ (normalized)', color='white')
        ax4.set_ylabel('Temperature (keV)', color='white')
        ax4.grid(True, alpha=0.3)
        
        # Add state information as text
        info_text = f"""
Shape Error: {state.shape_error:.2f} cm
Q-min: {state.q_min:.2f}
Beta: {state.plasma_beta:.3f}
κ: {state.elongation:.2f}
δ: {state.triangularity:.2f}
"""
        self.fig.text(0.02, 0.98, info_text, fontsize=10, color='white',
                     verticalalignment='top', fontfamily='monospace')
        
    def _render_from_arrays(self, flux_surfaces: Optional[np.ndarray],
                           q_profile: Optional[np.ndarray],
                           pressure: Optional[np.ndarray]) -> None:
        """Render visualization from individual arrays."""
        # Placeholder implementation for array-based rendering
        for ax in self.axes.flat:
            ax.text(0.5, 0.5, 'Data visualization\nrequires PlasmaState',
                   transform=ax.transAxes, ha='center', va='center',
                   color='white', fontsize=12)
            
    def save_frame(self, filename: str) -> None:
        """Save current frame to file."""
        if self.fig is not None:
            self.fig.savefig(filename, dpi=100, bbox_inches='tight',
                           facecolor='black', edgecolor='none')
            
    def close(self) -> None:
        """Close the renderer and free resources."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.axes = None


class TensorBoardLogger:
    """TensorBoard integration for RL training metrics."""
    
    def __init__(self, log_dir: str = "./tensorboard_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.log_dir)
            self.enabled = True
        except ImportError:
            warnings.warn("TensorBoard not available. Install with: pip install tensorboard")
            self.writer = None
            self.enabled = False
            
    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log scalar value."""
        if self.enabled:
            self.writer.add_scalar(tag, value, step)
            
    def log_scalars(self, tag_dict: Dict[str, float], step: int) -> None:
        """Log multiple scalar values."""
        if self.enabled:
            for tag, value in tag_dict.items():
                self.writer.add_scalar(tag, value, step)
                
    def log_plasma_state(self, state: PlasmaState, step: int) -> None:
        """Log plasma state metrics."""
        if not self.enabled:
            return
            
        # Plasma physics metrics
        self.writer.add_scalar('plasma/q_min', state.q_min, step)
        self.writer.add_scalar('plasma/beta', state.plasma_beta, step)
        self.writer.add_scalar('plasma/shape_error', state.shape_error, step)
        self.writer.add_scalar('plasma/elongation', state.elongation, step)
        self.writer.add_scalar('plasma/triangularity', state.triangularity, step)
        self.writer.add_scalar('plasma/disruption_probability', state.disruption_probability, step)
        
        # Profile histograms
        self.writer.add_histogram('profiles/q_profile', state.q_profile, step)
        self.writer.add_histogram('profiles/pressure', state.pressure_profile, step)
        self.writer.add_histogram('profiles/temperature', state.temperature_profile, step)
        self.writer.add_histogram('profiles/density', state.density_profile, step)
        
        # Control currents
        self.writer.add_histogram('control/pf_currents', state.pf_coil_currents, step)
        
    def log_training_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Log RL training metrics."""
        if not self.enabled:
            return
            
        for key, value in metrics.items():
            self.writer.add_scalar(f'training/{key}', value, step)
            
    def log_episode_metrics(self, metrics: Dict[str, float], episode: int) -> None:
        """Log episode-level metrics."""
        if not self.enabled:
            return
            
        for key, value in metrics.items():
            self.writer.add_scalar(f'episode/{key}', value, episode)
            
    def close(self) -> None:
        """Close TensorBoard writer."""
        if self.enabled and self.writer:
            self.writer.close()


def create_monitoring_system(log_dir: str = "./logs",
                           enable_tensorboard: bool = True,
                           enable_alerts: bool = True) -> Dict[str, Any]:
    """
    Factory function to create complete monitoring system.
    
    Returns:
        Dictionary containing monitor, renderer, and logger instances
    """
    monitor = PlasmaMonitor(
        log_dir=f"{log_dir}/plasma",
        enable_alerts=enable_alerts
    )
    
    renderer = PlasmaRenderer()
    
    logger = None
    if enable_tensorboard:
        logger = TensorBoardLogger(f"{log_dir}/tensorboard")
        
    return {
        'monitor': monitor,
        'renderer': renderer,
        'logger': logger
    }