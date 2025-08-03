"""
Business logic algorithms for tokamak optimization and operation management.

This module implements advanced algorithms for plasma shape optimization,
performance analysis, operational scenario planning, and cost-benefit analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from pathlib import Path
from scipy.optimize import minimize, differential_evolution
from scipy.signal import savgol_filter
from .physics import PlasmaState, TokamakConfig
from .safety import SafetyLimits


class OperationalMode(Enum):
    """Operational modes for tokamak operation."""
    STARTUP = "startup"
    FLATTOP = "flattop"
    RAMPDOWN = "rampdown"
    DISCHARGE_CLEANING = "cleaning"
    DISRUPTION_MITIGATION = "mitigation"
    MAINTENANCE = "maintenance"


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for tokamak operation."""
    # Physics performance
    q_factor_stability: float = 0.0
    shape_control_accuracy: float = 0.0
    beta_normalized: float = 0.0
    confinement_quality: float = 0.0
    disruption_avoidance_rate: float = 0.0
    
    # Operational efficiency
    uptime_percentage: float = 0.0
    energy_efficiency: float = 0.0
    target_achievement_rate: float = 0.0
    control_smoothness: float = 0.0
    
    # Economic metrics
    operational_cost_per_shot: float = 0.0
    power_consumption_efficiency: float = 0.0
    maintenance_cost_factor: float = 0.0
    
    # Timestamps and identifiers
    timestamp: float = 0.0
    session_id: str = ""
    

@dataclass
class OptimizationTarget:
    """Target parameters for plasma optimization."""
    desired_beta: float = 0.025
    target_q_min: float = 2.0
    max_shape_error: float = 2.0
    elongation_target: float = 1.7
    triangularity_target: float = 0.4
    
    # Weights for multi-objective optimization
    beta_weight: float = 1.0
    q_stability_weight: float = 2.0
    shape_weight: float = 1.5
    efficiency_weight: float = 1.0
    safety_weight: float = 3.0


class PlasmaOptimizer:
    """Advanced plasma shape and performance optimization algorithms."""
    
    def __init__(self, config: TokamakConfig, safety_limits: SafetyLimits):
        self.config = config
        self.safety_limits = safety_limits
        self.logger = logging.getLogger(__name__)
        
        # Optimization history
        self.optimization_history: List[Dict[str, Any]] = []
        self.best_solutions: List[Tuple[np.ndarray, float]] = []
        
        # Performance tracking
        self.performance_buffer: List[PerformanceMetrics] = []
        self.buffer_size = 1000
        
    def optimize_plasma_shape(self, 
                            current_state: PlasmaState,
                            target: OptimizationTarget,
                            method: str = "differential_evolution") -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Optimize plasma shape and control parameters.
        
        Args:
            current_state: Current plasma state
            target: Optimization targets and weights
            method: Optimization method ('scipy', 'differential_evolution', 'genetic')
            
        Returns:
            Optimal control action and optimization metrics
        """
        def objective_function(action: np.ndarray) -> float:
            """Multi-objective optimization function."""
            # Simulate plasma response (simplified)
            predicted_state = self._simulate_plasma_response(current_state, action)
            
            # Calculate individual objectives
            beta_error = abs(predicted_state.plasma_beta - target.desired_beta)
            q_stability = max(0, target.target_q_min - predicted_state.q_min)
            shape_error = predicted_state.shape_error
            
            # Safety penalties
            safety_penalty = 0.0
            if predicted_state.q_min < self.safety_limits.q_min_threshold:
                safety_penalty += 10.0 * (self.safety_limits.q_min_threshold - predicted_state.q_min) ** 2
            if predicted_state.plasma_beta > self.safety_limits.beta_limit:
                safety_penalty += 5.0 * (predicted_state.plasma_beta - self.safety_limits.beta_limit) ** 2
            if predicted_state.disruption_probability > self.safety_limits.disruption_probability_limit:
                safety_penalty += 20.0 * predicted_state.disruption_probability ** 2
                
            # Control smoothness (penalize large changes)
            control_penalty = 0.1 * np.sum(action ** 2)
            
            # Weighted objective
            total_cost = (target.beta_weight * beta_error +
                         target.q_stability_weight * q_stability +
                         target.shape_weight * shape_error +
                         target.safety_weight * safety_penalty +
                         target.efficiency_weight * control_penalty)
            
            return total_cost
        
        # Define bounds for control actions
        bounds = [
            (-5.0, 5.0),  # PF coil currents
            (-5.0, 5.0),
            (-3.0, 3.0),
            (-3.0, 3.0),
            (-2.0, 2.0),
            (-2.0, 2.0),
            (0.0, 1.0),   # Gas puff rate
            (0.0, 50.0)   # Heating power
        ]
        
        # Run optimization
        if method == "differential_evolution":
            result = differential_evolution(
                objective_function,
                bounds,
                maxiter=50,
                popsize=15,
                atol=1e-3,
                seed=42
            )
        else:
            # Scipy minimize
            x0 = np.zeros(8)  # Initial guess
            result = minimize(
                objective_function,
                x0,
                bounds=bounds,
                method='L-BFGS-B'
            )
        
        # Store optimization results
        optimization_info = {
            'success': result.success,
            'fun': result.fun,
            'nit': getattr(result, 'nit', 0),
            'nfev': result.nfev,
            'method': method,
            'target': target,
            'timestamp': np.datetime64('now').astype(float)
        }
        
        self.optimization_history.append(optimization_info)
        
        if result.success:
            self.best_solutions.append((result.x, result.fun))
            # Keep only best 100 solutions
            self.best_solutions.sort(key=lambda x: x[1])
            self.best_solutions = self.best_solutions[:100]
        
        return result.x, optimization_info
    
    def _simulate_plasma_response(self, state: PlasmaState, action: np.ndarray) -> PlasmaState:
        """Simplified plasma response simulation for optimization."""
        # Create new state by copying current state
        new_state = PlasmaState(self.config)
        new_state.psi_profile = state.psi_profile.copy()
        new_state.q_profile = state.q_profile.copy()
        new_state.pressure_profile = state.pressure_profile.copy()
        new_state.temperature_profile = state.temperature_profile.copy()
        new_state.density_profile = state.density_profile.copy()
        
        # Apply control action effects (simplified linear model)
        pf_effect = np.sum(action[:6]) * 0.01
        new_state.shape_error = max(0.1, state.shape_error + pf_effect)
        
        # Beta response to heating
        heating_effect = action[7] * 0.0001
        new_state.plasma_beta = min(0.1, state.plasma_beta + heating_effect)
        
        # Q-profile response
        new_state.q_min = max(0.5, state.q_min + pf_effect * 0.1)
        
        # Elongation and triangularity
        new_state.elongation = np.clip(state.elongation + action[2] * 0.01, 1.0, 2.5)
        new_state.triangularity = np.clip(state.triangularity + action[3] * 0.01, 0.0, 0.8)
        
        # Update PF coil currents
        new_state.pf_coil_currents = state.pf_coil_currents + action[:6] * 0.1
        
        return new_state
    
    def analyze_operational_efficiency(self, performance_data: List[PerformanceMetrics]) -> Dict[str, float]:
        """Analyze operational efficiency from performance data."""
        if not performance_data:
            return {}
        
        metrics = {
            'mean_q_stability': np.mean([p.q_factor_stability for p in performance_data]),
            'shape_control_rms': np.sqrt(np.mean([p.shape_control_accuracy ** 2 for p in performance_data])),
            'avg_beta_normalized': np.mean([p.beta_normalized for p in performance_data]),
            'disruption_rate': 1.0 - np.mean([p.disruption_avoidance_rate for p in performance_data]),
            'uptime_factor': np.mean([p.uptime_percentage for p in performance_data]),
            'energy_efficiency': np.mean([p.energy_efficiency for p in performance_data]),
            'cost_per_shot': np.mean([p.operational_cost_per_shot for p in performance_data]),
            'control_smoothness': np.mean([p.control_smoothness for p in performance_data])
        }
        
        # Calculate efficiency trends
        if len(performance_data) > 10:
            recent_efficiency = np.mean([p.energy_efficiency for p in performance_data[-10:]])
            historical_efficiency = np.mean([p.energy_efficiency for p in performance_data[:-10]])
            metrics['efficiency_trend'] = recent_efficiency - historical_efficiency
            
        return metrics
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization performance."""
        if not self.optimization_history:
            return {"message": "No optimization data available"}
        
        successful_opts = [opt for opt in self.optimization_history if opt['success']]
        
        summary = {
            'total_optimizations': len(self.optimization_history),
            'successful_optimizations': len(successful_opts),
            'success_rate': len(successful_opts) / len(self.optimization_history),
            'best_objective_value': min([opt['fun'] for opt in successful_opts]) if successful_opts else None,
            'average_iterations': np.mean([opt['nit'] for opt in successful_opts]) if successful_opts else 0,
            'average_function_evaluations': np.mean([opt['nfev'] for opt in successful_opts]) if successful_opts else 0
        }
        
        return summary


class ScenarioPlanner:
    """Operational scenario planning and scheduling algorithms."""
    
    def __init__(self, config: TokamakConfig):
        self.config = config
        self.scenarios: Dict[str, Dict[str, Any]] = {}
        self.schedule: List[Dict[str, Any]] = []
        
    def create_discharge_scenario(self, 
                                scenario_name: str,
                                duration: float,
                                mode: OperationalMode,
                                target_parameters: Dict[str, float]) -> Dict[str, Any]:
        """Create a discharge scenario with time-dependent targets."""
        
        # Create time vector
        dt = 0.1  # 100ms time steps
        time_points = np.arange(0, duration + dt, dt)
        n_points = len(time_points)
        
        scenario = {
            'name': scenario_name,
            'mode': mode,
            'duration': duration,
            'time_points': time_points,
            'targets': {}
        }
        
        # Generate time-dependent target trajectories
        if mode == OperationalMode.STARTUP:
            scenario['targets'] = self._generate_startup_trajectory(time_points, target_parameters)
        elif mode == OperationalMode.FLATTOP:
            scenario['targets'] = self._generate_flattop_trajectory(time_points, target_parameters)
        elif mode == OperationalMode.RAMPDOWN:
            scenario['targets'] = self._generate_rampdown_trajectory(time_points, target_parameters)
        else:
            # Default constant targets
            for param, value in target_parameters.items():
                scenario['targets'][param] = np.full(n_points, value)
        
        self.scenarios[scenario_name] = scenario
        return scenario
    
    def _generate_startup_trajectory(self, time_points: np.ndarray, targets: Dict[str, float]) -> Dict[str, np.ndarray]:
        """Generate startup phase trajectory."""
        n_points = len(time_points)
        trajectories = {}
        
        # Plasma current ramp
        trajectories['plasma_current'] = targets.get('plasma_current', 15e6) * (
            1 - np.exp(-time_points / 2.0)
        )
        
        # Beta ramp (slower than current)
        trajectories['beta'] = targets.get('beta', 0.025) * (
            1 - np.exp(-time_points / 4.0)
        )
        
        # Shape parameters (gradual optimization)
        trajectories['elongation'] = 1.0 + (targets.get('elongation', 1.7) - 1.0) * (
            1 - np.exp(-time_points / 3.0)
        )
        
        trajectories['triangularity'] = targets.get('triangularity', 0.4) * (
            1 - np.exp(-time_points / 3.0)
        )
        
        # Q-factor target
        trajectories['q_min'] = targets.get('q_min', 2.0) * np.ones(n_points)
        
        return trajectories
    
    def _generate_flattop_trajectory(self, time_points: np.ndarray, targets: Dict[str, float]) -> Dict[str, np.ndarray]:
        """Generate flat-top phase trajectory with small variations."""
        n_points = len(time_points)
        trajectories = {}
        
        for param, value in targets.items():
            if param in ['plasma_current', 'beta']:
                # Add small controlled variations for flattop
                variation = 0.02 * value * np.sin(2 * np.pi * time_points / 10.0)
                trajectories[param] = value + variation
            else:
                trajectories[param] = np.full(n_points, value)
        
        return trajectories
    
    def _generate_rampdown_trajectory(self, time_points: np.ndarray, targets: Dict[str, float]) -> Dict[str, np.ndarray]:
        """Generate controlled rampdown trajectory."""
        n_points = len(time_points)
        trajectories = {}
        
        # Exponential decay for current and beta
        decay_time = targets.get('rampdown_time', 5.0)
        
        trajectories['plasma_current'] = targets.get('initial_current', 15e6) * np.exp(
            -time_points / decay_time
        )
        
        trajectories['beta'] = targets.get('initial_beta', 0.025) * np.exp(
            -time_points / (decay_time * 1.5)
        )
        
        # Shape parameters return to safe values
        trajectories['elongation'] = 1.0 + (targets.get('initial_elongation', 1.7) - 1.0) * np.exp(
            -time_points / decay_time
        )
        
        trajectories['triangularity'] = targets.get('initial_triangularity', 0.4) * np.exp(
            -time_points / decay_time
        )
        
        trajectories['q_min'] = targets.get('q_min', 2.0) * np.ones(n_points)
        
        return trajectories
    
    def optimize_schedule(self, 
                         scenarios: List[str], 
                         constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Optimize operational schedule given scenarios and constraints."""
        
        # Simple scheduling algorithm
        schedule = []
        current_time = 0.0
        
        for scenario_name in scenarios:
            if scenario_name not in self.scenarios:
                continue
                
            scenario = self.scenarios[scenario_name]
            
            # Add preparation time
            prep_time = constraints.get('preparation_time', 600.0)  # 10 minutes
            
            schedule_item = {
                'start_time': current_time,
                'scenario': scenario_name,
                'duration': scenario['duration'],
                'mode': scenario['mode'],
                'preparation_time': prep_time
            }
            
            schedule.append(schedule_item)
            current_time += scenario['duration'] + prep_time
        
        self.schedule = schedule
        return schedule
    
    def get_current_targets(self, scenario_name: str, current_time: float) -> Dict[str, float]:
        """Get target parameters for current time in scenario."""
        if scenario_name not in self.scenarios:
            return {}
        
        scenario = self.scenarios[scenario_name]
        time_points = scenario['time_points']
        
        # Find closest time index
        idx = np.argmin(np.abs(time_points - current_time))
        
        targets = {}
        for param, trajectory in scenario['targets'].items():
            targets[param] = trajectory[idx]
        
        return targets


class PerformanceAnalyzer:
    """Advanced performance analysis and reporting."""
    
    def __init__(self, data_dir: str = "./performance_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.performance_data: List[PerformanceMetrics] = []
        self.analysis_cache: Dict[str, Any] = {}
        
    def add_performance_data(self, metrics: PerformanceMetrics) -> None:
        """Add performance metrics data point."""
        self.performance_data.append(metrics)
        
        # Maintain buffer size
        if len(self.performance_data) > 10000:
            self.performance_data = self.performance_data[-5000:]
    
    def calculate_kpis(self, time_window: Optional[float] = None) -> Dict[str, float]:
        """Calculate key performance indicators."""
        if not self.performance_data:
            return {}
        
        data = self.performance_data
        if time_window:
            current_time = max(p.timestamp for p in self.performance_data)
            data = [p for p in self.performance_data if current_time - p.timestamp <= time_window]
        
        if not data:
            return {}
        
        kpis = {
            # Operational KPIs
            'availability': np.mean([p.uptime_percentage for p in data]),
            'reliability': np.mean([p.disruption_avoidance_rate for p in data]),
            'efficiency': np.mean([p.energy_efficiency for p in data]),
            
            # Performance KPIs
            'shape_accuracy': 100.0 - np.mean([p.shape_control_accuracy for p in data]),
            'plasma_performance': np.mean([p.beta_normalized for p in data]),
            'stability': np.mean([p.q_factor_stability for p in data]),
            
            # Economic KPIs
            'cost_efficiency': 1.0 / (1.0 + np.mean([p.operational_cost_per_shot for p in data])),
            'maintenance_efficiency': 1.0 / (1.0 + np.mean([p.maintenance_cost_factor for p in data])),
            
            # Operational smoothness
            'control_quality': np.mean([p.control_smoothness for p in data])
        }
        
        return kpis
    
    def generate_trend_analysis(self, metric: str, window_size: int = 100) -> Dict[str, Any]:
        """Generate trend analysis for specific metric."""
        if len(self.performance_data) < window_size:
            return {"error": "Insufficient data for trend analysis"}
        
        # Extract metric values
        values = []
        timestamps = []
        
        for p in self.performance_data:
            if hasattr(p, metric):
                values.append(getattr(p, metric))
                timestamps.append(p.timestamp)
        
        if len(values) < window_size:
            return {"error": f"Insufficient data for metric {metric}"}
        
        values = np.array(values)
        timestamps = np.array(timestamps)
        
        # Calculate trends
        recent_values = values[-window_size:]
        historical_values = values[:-window_size] if len(values) > window_size else values[:window_size//2]
        
        trend_analysis = {
            'metric': metric,
            'current_value': values[-1],
            'recent_mean': np.mean(recent_values),
            'recent_std': np.std(recent_values),
            'historical_mean': np.mean(historical_values),
            'trend_direction': 'improving' if np.mean(recent_values) > np.mean(historical_values) else 'declining',
            'trend_magnitude': abs(np.mean(recent_values) - np.mean(historical_values)),
            'volatility': np.std(recent_values) / (np.mean(recent_values) + 1e-6),
            'data_points': len(values)
        }
        
        # Apply smoothing filter for trend line
        if len(values) > 20:
            smoothed = savgol_filter(values, min(21, len(values)//4*2+1), 3)
            trend_analysis['smoothed_trend'] = smoothed[-window_size:].tolist()
        
        return trend_analysis
    
    def generate_performance_report(self) -> str:
        """Generate comprehensive performance report."""
        if not self.performance_data:
            return "No performance data available for reporting."
        
        # Calculate KPIs
        kpis = self.calculate_kpis()
        recent_kpis = self.calculate_kpis(time_window=86400)  # Last 24 hours
        
        # Generate report
        report = f"""
TOKAMAK PERFORMANCE REPORT
==========================
Generated: {np.datetime64('now')}
Data Points: {len(self.performance_data)}

OPERATIONAL KPIs:
-----------------
Availability: {kpis.get('availability', 0):.1f}% (24h: {recent_kpis.get('availability', 0):.1f}%)
Reliability: {kpis.get('reliability', 0):.1f}% (24h: {recent_kpis.get('reliability', 0):.1f}%)
Energy Efficiency: {kpis.get('efficiency', 0):.1f}% (24h: {recent_kpis.get('efficiency', 0):.1f}%)

PLASMA PERFORMANCE:
-------------------
Shape Control Accuracy: {kpis.get('shape_accuracy', 0):.1f}% (24h: {recent_kpis.get('shape_accuracy', 0):.1f}%)
Normalized Beta: {kpis.get('plasma_performance', 0):.3f} (24h: {recent_kpis.get('plasma_performance', 0):.3f})
Q-factor Stability: {kpis.get('stability', 0):.1f}% (24h: {recent_kpis.get('stability', 0):.1f}%)

ECONOMIC METRICS:
-----------------
Cost Efficiency: {kpis.get('cost_efficiency', 0):.3f} (24h: {recent_kpis.get('cost_efficiency', 0):.3f})
Maintenance Efficiency: {kpis.get('maintenance_efficiency', 0):.3f} (24h: {recent_kpis.get('maintenance_efficiency', 0):.3f})

CONTROL QUALITY:
----------------
Control Smoothness: {kpis.get('control_quality', 0):.1f}% (24h: {recent_kpis.get('control_quality', 0):.1f}%)
"""
        
        # Add trend analysis for key metrics
        key_metrics = ['energy_efficiency', 'shape_control_accuracy', 'disruption_avoidance_rate']
        
        report += "\nTREND ANALYSIS:\n---------------\n"
        for metric in key_metrics:
            trend = self.generate_trend_analysis(metric)
            if 'error' not in trend:
                direction = "📈" if trend['trend_direction'] == 'improving' else "📉"
                report += f"{metric}: {direction} {trend['trend_direction']} (magnitude: {trend['trend_magnitude']:.3f})\n"
        
        return report
    
    def save_performance_data(self, filename: Optional[str] = None) -> str:
        """Save performance data to file."""
        if not filename:
            timestamp = np.datetime64('now').astype(str).replace(':', '-')
            filename = f"performance_data_{timestamp}.json"
        
        filepath = self.data_dir / filename
        
        # Convert to serializable format
        data = []
        for p in self.performance_data:
            data.append({
                'q_factor_stability': p.q_factor_stability,
                'shape_control_accuracy': p.shape_control_accuracy,
                'beta_normalized': p.beta_normalized,
                'confinement_quality': p.confinement_quality,
                'disruption_avoidance_rate': p.disruption_avoidance_rate,
                'uptime_percentage': p.uptime_percentage,
                'energy_efficiency': p.energy_efficiency,
                'target_achievement_rate': p.target_achievement_rate,
                'control_smoothness': p.control_smoothness,
                'operational_cost_per_shot': p.operational_cost_per_shot,
                'power_consumption_efficiency': p.power_consumption_efficiency,
                'maintenance_cost_factor': p.maintenance_cost_factor,
                'timestamp': p.timestamp,
                'session_id': p.session_id
            })
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        return str(filepath)


def create_business_system(config: TokamakConfig, 
                          safety_limits: SafetyLimits,
                          data_dir: str = "./business_data") -> Dict[str, Any]:
    """
    Factory function to create complete business logic system.
    
    Returns:
        Dictionary containing optimizer, planner, and analyzer instances
    """
    optimizer = PlasmaOptimizer(config, safety_limits)
    planner = ScenarioPlanner(config)
    analyzer = PerformanceAnalyzer(data_dir)
    
    return {
        'optimizer': optimizer,
        'planner': planner,
        'analyzer': analyzer
    }