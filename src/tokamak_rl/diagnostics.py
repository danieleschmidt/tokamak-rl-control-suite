"""
Comprehensive system diagnostics and health checks for tokamak RL control.

This module provides advanced diagnostic capabilities, system health monitoring,
and automated troubleshooting for production tokamak control systems.
"""

import time
import threading
import psutil
import os
import gc
import traceback
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import json
import logging

try:
    import numpy as np
except ImportError:
    import math
    
    class np:
        @staticmethod
        def array(x):
            return list(x) if hasattr(x, '__iter__') else [x]
        
        @staticmethod
        def mean(arr):
            return sum(arr) / len(arr) if arr else 0

logger = logging.getLogger(__name__)


class DiagnosticSeverity(Enum):
    """Severity levels for diagnostic issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class SystemComponent(Enum):
    """System components for diagnostic categorization."""
    ENVIRONMENT = "environment"
    PHYSICS_SOLVER = "physics_solver"
    RL_AGENT = "rl_agent"
    SAFETY_SYSTEM = "safety_system"
    MONITORING = "monitoring"
    HARDWARE = "hardware"
    NETWORK = "network"
    STORAGE = "storage"


@dataclass
class DiagnosticResult:
    """Result of a diagnostic check."""
    component: SystemComponent
    check_name: str
    severity: DiagnosticSeverity
    status: str  # "pass", "fail", "warning"
    message: str
    timestamp: float
    details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    metric_values: Dict[str, float] = field(default_factory=dict)


class PerformanceDiagnostics:
    """Performance monitoring and analysis diagnostics."""
    
    def __init__(self):
        self.performance_history = deque(maxlen=1000)
        self.memory_tracker = deque(maxlen=100)
        self.cpu_tracker = deque(maxlen=100)
        self._lock = threading.RLock()
    
    def check_system_performance(self) -> DiagnosticResult:
        """Comprehensive system performance check."""
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Track metrics
            with self._lock:
                self.cpu_tracker.append(cpu_percent)
                self.memory_tracker.append(memory.percent)
            
            # Analyze performance
            issues = []
            recommendations = []
            severity = DiagnosticSeverity.INFO
            
            # CPU analysis
            if cpu_percent > 90:
                issues.append(f"High CPU usage: {cpu_percent:.1f}%")
                recommendations.append("Consider reducing computational load or scaling resources")
                severity = max(severity, DiagnosticSeverity.WARNING)
            
            # Memory analysis
            if memory.percent > 85:
                issues.append(f"High memory usage: {memory.percent:.1f}%")
                recommendations.append("Monitor for memory leaks, consider increasing available RAM")
                severity = max(severity, DiagnosticSeverity.WARNING)
            
            if memory.percent > 95:
                severity = DiagnosticSeverity.CRITICAL
                recommendations.append("CRITICAL: System may become unstable, immediate action required")
            
            # Disk space analysis
            disk_percent = (disk.used / disk.total) * 100
            if disk_percent > 85:
                issues.append(f"Low disk space: {disk_percent:.1f}% used")
                recommendations.append("Clean up old files or expand storage capacity")
                severity = max(severity, DiagnosticSeverity.WARNING)
            
            # Performance trends
            if len(self.cpu_tracker) > 10:
                recent_cpu_avg = np.mean(list(self.cpu_tracker)[-10:])
                if recent_cpu_avg > 80:
                    issues.append(f"Sustained high CPU usage: {recent_cpu_avg:.1f}% average")
                    recommendations.append("Investigate long-running processes or optimize algorithms")
            
            status = "pass" if not issues else ("warning" if severity != DiagnosticSeverity.CRITICAL else "fail")
            message = "System performance normal" if not issues else "; ".join(issues)
            
            return DiagnosticResult(
                component=SystemComponent.HARDWARE,
                check_name="system_performance",
                severity=severity,
                status=status,
                message=message,
                timestamp=time.time(),
                details={
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_available_gb': memory.available / (1024**3),
                    'disk_percent': disk_percent,
                    'disk_free_gb': disk.free / (1024**3)
                },
                recommendations=recommendations,
                metric_values={
                    'cpu_usage': cpu_percent,
                    'memory_usage': memory.percent,
                    'disk_usage': disk_percent
                }
            )
            
        except Exception as e:
            return DiagnosticResult(
                component=SystemComponent.HARDWARE,
                check_name="system_performance",
                severity=DiagnosticSeverity.ERROR,
                status="fail",
                message=f"Performance check failed: {e}",
                timestamp=time.time(),
                recommendations=["Check system monitoring tools and permissions"]
            )
    
    def check_memory_leaks(self) -> DiagnosticResult:
        """Check for potential memory leaks."""
        try:
            # Force garbage collection
            gc.collect()
            
            # Get current memory info
            process = psutil.Process()
            memory_info = process.memory_info()
            
            # Track memory over time
            with self._lock:
                self.performance_history.append({
                    'timestamp': time.time(),
                    'rss': memory_info.rss,
                    'vms': memory_info.vms
                })
            
            issues = []
            recommendations = []
            severity = DiagnosticSeverity.INFO
            
            # Analyze memory growth trends
            if len(self.performance_history) >= 50:
                recent_data = list(self.performance_history)[-50:]
                early_data = recent_data[:25]
                late_data = recent_data[25:]
                
                early_avg = np.mean([d['rss'] for d in early_data])
                late_avg = np.mean([d['rss'] for d in late_data])
                
                growth_percent = ((late_avg - early_avg) / early_avg) * 100
                
                if growth_percent > 20:
                    issues.append(f"Significant memory growth detected: {growth_percent:.1f}%")
                    recommendations.append("Monitor for memory leaks in RL training or physics calculations")
                    severity = DiagnosticSeverity.WARNING
                
                if growth_percent > 50:
                    severity = DiagnosticSeverity.ERROR
                    recommendations.append("Investigate immediate memory leak sources")
            
            # Check object counts
            object_count = len(gc.get_objects())
            if object_count > 100000:
                issues.append(f"High object count: {object_count}")
                recommendations.append("Check for unreferenced objects or circular references")
                severity = max(severity, DiagnosticSeverity.WARNING)
            
            status = "pass" if not issues else "warning"
            message = "No memory issues detected" if not issues else "; ".join(issues)
            
            return DiagnosticResult(
                component=SystemComponent.ENVIRONMENT,
                check_name="memory_leaks",
                severity=severity,
                status=status,
                message=message,
                timestamp=time.time(),
                details={
                    'rss_mb': memory_info.rss / (1024**2),
                    'vms_mb': memory_info.vms / (1024**2),
                    'object_count': object_count,
                    'gc_collections': gc.get_stats() if hasattr(gc, 'get_stats') else []
                },
                recommendations=recommendations,
                metric_values={
                    'memory_rss_mb': memory_info.rss / (1024**2),
                    'object_count': object_count
                }
            )
            
        except Exception as e:
            return DiagnosticResult(
                component=SystemComponent.ENVIRONMENT,
                check_name="memory_leaks",
                severity=DiagnosticSeverity.ERROR,
                status="fail",
                message=f"Memory leak check failed: {e}",
                timestamp=time.time()
            )


class ComponentDiagnostics:
    """Diagnostics for specific system components."""
    
    def __init__(self):
        self.component_states = {}
        self.error_counts = defaultdict(int)
        self.last_errors = defaultdict(list)
    
    def check_environment_health(self, env) -> DiagnosticResult:
        """Check RL environment health and functionality."""
        try:
            issues = []
            recommendations = []
            severity = DiagnosticSeverity.INFO
            
            # Check environment attributes
            required_attrs = ['observation_space', 'action_space', 'reset', 'step']
            for attr in required_attrs:
                if not hasattr(env, attr):
                    issues.append(f"Missing required attribute: {attr}")
                    severity = DiagnosticSeverity.ERROR
            
            # Test environment reset
            try:
                obs, info = env.reset()
                if obs is None:
                    issues.append("Environment reset returned None observation")
                    severity = max(severity, DiagnosticSeverity.ERROR)
            except Exception as e:
                issues.append(f"Environment reset failed: {e}")
                severity = DiagnosticSeverity.ERROR
                recommendations.append("Check environment initialization parameters")
            
            # Test environment step
            try:
                if hasattr(env, 'action_space') and hasattr(env.action_space, 'sample'):
                    action = env.action_space.sample()
                else:
                    action = [0.0] * 8  # Default action
                
                obs, reward, terminated, truncated, info = env.step(action)
                
                if obs is None:
                    issues.append("Environment step returned None observation")
                    severity = max(severity, DiagnosticSeverity.ERROR)
                
                if not isinstance(reward, (int, float)):
                    issues.append(f"Invalid reward type: {type(reward)}")
                    severity = max(severity, DiagnosticSeverity.WARNING)
                    
            except Exception as e:
                issues.append(f"Environment step failed: {e}")
                severity = DiagnosticSeverity.ERROR
                recommendations.append("Verify action space compatibility and physics solver")
            
            # Check safety integration
            if hasattr(env, 'safety_shield') and env.safety_shield:
                try:
                    # Test safety shield functionality
                    if hasattr(env.safety_shield, 'get_safety_statistics'):
                        stats = env.safety_shield.get_safety_statistics()
                        if stats.get('total_interventions', 0) > 1000:
                            issues.append("High safety intervention count")
                            recommendations.append("Review safety thresholds and agent behavior")
                            severity = max(severity, DiagnosticSeverity.WARNING)
                except Exception as e:
                    issues.append(f"Safety shield check failed: {e}")
                    severity = max(severity, DiagnosticSeverity.WARNING)
            
            status = "pass" if not issues else ("warning" if severity != DiagnosticSeverity.ERROR else "fail")
            message = "Environment functioning normally" if not issues else "; ".join(issues)
            
            return DiagnosticResult(
                component=SystemComponent.ENVIRONMENT,
                check_name="environment_health",
                severity=severity,
                status=status,
                message=message,
                timestamp=time.time(),
                details={
                    'has_safety_shield': hasattr(env, 'safety_shield') and env.safety_shield is not None,
                    'observation_space_shape': getattr(env.observation_space, 'shape', None),
                    'action_space_shape': getattr(env.action_space, 'shape', None)
                },
                recommendations=recommendations
            )
            
        except Exception as e:
            return DiagnosticResult(
                component=SystemComponent.ENVIRONMENT,
                check_name="environment_health",
                severity=DiagnosticSeverity.ERROR,
                status="fail",
                message=f"Environment health check failed: {e}",
                timestamp=time.time(),
                recommendations=["Check environment initialization and dependencies"]
            )
    
    def check_agent_health(self, agent) -> DiagnosticResult:
        """Check RL agent health and training status."""
        try:
            issues = []
            recommendations = []
            severity = DiagnosticSeverity.INFO
            
            # Check agent attributes
            required_methods = ['act', 'learn', 'save', 'load']
            for method in required_methods:
                if not hasattr(agent, method) or not callable(getattr(agent, method)):
                    issues.append(f"Missing or invalid method: {method}")
                    severity = DiagnosticSeverity.ERROR
            
            # Check training progress
            if hasattr(agent, 'training_steps'):
                steps = agent.training_steps
                if steps == 0:
                    issues.append("Agent has not been trained")
                    recommendations.append("Ensure training loop is running")
                    severity = max(severity, DiagnosticSeverity.WARNING)
                elif steps < 1000:
                    issues.append(f"Low training steps: {steps}")
                    recommendations.append("Allow more training time for better performance")
                    severity = max(severity, DiagnosticSeverity.INFO)
            
            # Check replay buffer if available
            if hasattr(agent, 'replay_buffer'):
                buffer_size = len(agent.replay_buffer)
                if buffer_size < 100:
                    issues.append(f"Small replay buffer: {buffer_size}")
                    recommendations.append("Collect more experience before intensive training")
                    severity = max(severity, DiagnosticSeverity.WARNING)
            
            # Test action generation
            try:
                test_obs = [0.0] * 45  # Standard observation size
                action = agent.act(test_obs, deterministic=True)
                
                if action is None:
                    issues.append("Agent returned None action")
                    severity = DiagnosticSeverity.ERROR
                elif not hasattr(action, '__len__') or len(action) != 8:
                    issues.append(f"Invalid action shape: {np.array(action).shape}")
                    severity = DiagnosticSeverity.ERROR
                    
            except Exception as e:
                issues.append(f"Action generation failed: {e}")
                severity = DiagnosticSeverity.ERROR
                recommendations.append("Check agent network initialization and input processing")
            
            # Check for NaN in networks (if accessible)
            try:
                if hasattr(agent, 'actor') and hasattr(agent.actor, 'parameters'):
                    for param in agent.actor.parameters():
                        if hasattr(param, 'data') and hasattr(param.data, 'isnan'):
                            if param.data.isnan().any():
                                issues.append("NaN detected in actor network")
                                severity = DiagnosticSeverity.CRITICAL
                                recommendations.append("Reduce learning rate or check gradient clipping")
                                break
            except Exception:
                pass  # Skip if torch not available or network structure different
            
            status = "pass" if not issues else ("warning" if severity not in [DiagnosticSeverity.ERROR, DiagnosticSeverity.CRITICAL] else "fail")
            message = "Agent functioning normally" if not issues else "; ".join(issues)
            
            return DiagnosticResult(
                component=SystemComponent.RL_AGENT,
                check_name="agent_health",
                severity=severity,
                status=status,
                message=message,
                timestamp=time.time(),
                details={
                    'training_steps': getattr(agent, 'training_steps', 0),
                    'buffer_size': len(getattr(agent, 'replay_buffer', [])),
                    'agent_type': type(agent).__name__
                },
                recommendations=recommendations
            )
            
        except Exception as e:
            return DiagnosticResult(
                component=SystemComponent.RL_AGENT,
                check_name="agent_health",
                severity=DiagnosticSeverity.ERROR,
                status="fail",
                message=f"Agent health check failed: {e}",
                timestamp=time.time(),
                recommendations=["Check agent initialization and network architecture"]
            )
    
    def check_safety_system(self, safety_shield) -> DiagnosticResult:
        """Check safety system health and responsiveness."""
        try:
            issues = []
            recommendations = []
            severity = DiagnosticSeverity.INFO
            
            if safety_shield is None:
                return DiagnosticResult(
                    component=SystemComponent.SAFETY_SYSTEM,
                    check_name="safety_system",
                    severity=DiagnosticSeverity.CRITICAL,
                    status="fail",
                    message="Safety system not initialized",
                    timestamp=time.time(),
                    recommendations=["Initialize safety shield before operation"]
                )
            
            # Check safety statistics
            if hasattr(safety_shield, 'get_safety_statistics'):
                stats = safety_shield.get_safety_statistics()
                
                intervention_rate = stats.get('violation_rate', 0)
                if intervention_rate > 0.5:
                    issues.append(f"High safety intervention rate: {intervention_rate:.2f}")
                    recommendations.append("Review agent behavior and safety thresholds")
                    severity = max(severity, DiagnosticSeverity.WARNING)
                
                if stats.get('emergency_mode_active', False):
                    issues.append("Emergency mode currently active")
                    recommendations.append("Investigate emergency conditions")
                    severity = DiagnosticSeverity.CRITICAL
                
                avg_risk = stats.get('average_risk', 0)
                if avg_risk > 0.1:
                    issues.append(f"High average disruption risk: {avg_risk:.3f}")
                    recommendations.append("Monitor plasma stability and adjust safety margins")
                    severity = max(severity, DiagnosticSeverity.WARNING)
            
            # Test safety shield functionality
            try:
                from .physics import TokamakConfig, PlasmaState
                config = TokamakConfig.from_preset("ITER")
                test_state = PlasmaState(config)
                test_action = [0.1] * 8
                
                safe_action, safety_info = safety_shield.filter_action(test_action, test_state)
                
                if safe_action is None:
                    issues.append("Safety shield returned None action")
                    severity = DiagnosticSeverity.ERROR
                
                if safety_info is None or not isinstance(safety_info, dict):
                    issues.append("Safety shield returned invalid info")
                    severity = max(severity, DiagnosticSeverity.WARNING)
                    
            except Exception as e:
                issues.append(f"Safety shield test failed: {e}")
                severity = DiagnosticSeverity.ERROR
                recommendations.append("Check safety shield dependencies and configuration")
            
            status = "pass" if not issues else ("warning" if severity != DiagnosticSeverity.CRITICAL else "fail")
            message = "Safety system functioning normally" if not issues else "; ".join(issues)
            
            return DiagnosticResult(
                component=SystemComponent.SAFETY_SYSTEM,
                check_name="safety_system",
                severity=severity,
                status=status,
                message=message,
                timestamp=time.time(),
                details={
                    'has_predictor': hasattr(safety_shield, 'predictor'),
                    'adaptive_constraints': getattr(safety_shield, 'adaptive_constraints', False)
                },
                recommendations=recommendations
            )
            
        except Exception as e:
            return DiagnosticResult(
                component=SystemComponent.SAFETY_SYSTEM,
                check_name="safety_system",
                severity=DiagnosticSeverity.ERROR,
                status="fail",
                message=f"Safety system check failed: {e}",
                timestamp=time.time(),
                recommendations=["Verify safety system initialization and dependencies"]
            )


class SystemDiagnostics:
    """Comprehensive system diagnostics coordinator."""
    
    def __init__(self):
        self.performance_diag = PerformanceDiagnostics()
        self.component_diag = ComponentDiagnostics()
        self.diagnostic_history = deque(maxlen=1000)
        self.last_full_check = 0
        self._lock = threading.RLock()
    
    def run_full_diagnostics(self, env=None, agent=None, safety_shield=None) -> Dict[str, DiagnosticResult]:
        """Run comprehensive system diagnostics."""
        with self._lock:
            results = {}
            
            # Performance diagnostics
            results['system_performance'] = self.performance_diag.check_system_performance()
            results['memory_leaks'] = self.performance_diag.check_memory_leaks()
            
            # Component diagnostics
            if env is not None:
                results['environment_health'] = self.component_diag.check_environment_health(env)
            
            if agent is not None:
                results['agent_health'] = self.component_diag.check_agent_health(agent)
            
            if safety_shield is not None:
                results['safety_system'] = self.component_diag.check_safety_system(safety_shield)
            
            # Store results in history
            for result in results.values():
                self.diagnostic_history.append(result)
            
            self.last_full_check = time.time()
            
            return results
    
    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get overall system health summary."""
        with self._lock:
            if not self.diagnostic_history:
                return {
                    'overall_status': 'unknown',
                    'last_check': 0,
                    'critical_issues': 0,
                    'warnings': 0,
                    'message': 'No diagnostics run yet'
                }
            
            # Analyze recent diagnostics (last 10 minutes)
            cutoff_time = time.time() - 600
            recent_results = [
                result for result in self.diagnostic_history
                if result.timestamp >= cutoff_time
            ]
            
            if not recent_results:
                return {
                    'overall_status': 'stale',
                    'last_check': self.last_full_check,
                    'critical_issues': 0,
                    'warnings': 0,
                    'message': 'Diagnostics data is stale'
                }
            
            # Count issues by severity
            critical_count = len([r for r in recent_results if r.severity == DiagnosticSeverity.CRITICAL])
            error_count = len([r for r in recent_results if r.severity == DiagnosticSeverity.ERROR])
            warning_count = len([r for r in recent_results if r.severity == DiagnosticSeverity.WARNING])
            
            # Determine overall status
            if critical_count > 0:
                overall_status = 'critical'
                message = f"{critical_count} critical issue(s) detected"
            elif error_count > 0:
                overall_status = 'error'
                message = f"{error_count} error(s) detected"
            elif warning_count > 0:
                overall_status = 'warning'
                message = f"{warning_count} warning(s) detected"
            else:
                overall_status = 'healthy'
                message = 'All systems operating normally'
            
            # Get component status breakdown
            component_status = {}
            for component in SystemComponent:
                component_results = [r for r in recent_results if r.component == component]
                if component_results:
                    latest = max(component_results, key=lambda x: x.timestamp)
                    component_status[component.value] = {
                        'status': latest.status,
                        'severity': latest.severity.value,
                        'message': latest.message,
                        'last_check': latest.timestamp
                    }
            
            return {
                'overall_status': overall_status,
                'last_check': self.last_full_check,
                'critical_issues': critical_count,
                'errors': error_count,
                'warnings': warning_count,
                'message': message,
                'component_status': component_status,
                'total_checks': len(recent_results)
            }
    
    def get_recommendations(self, max_recommendations: int = 10) -> List[str]:
        """Get prioritized recommendations based on recent diagnostics."""
        with self._lock:
            cutoff_time = time.time() - 3600  # Last hour
            recent_results = [
                result for result in self.diagnostic_history
                if result.timestamp >= cutoff_time and result.recommendations
            ]
            
            # Collect and prioritize recommendations
            recommendation_scores = defaultdict(float)
            
            for result in recent_results:
                severity_weight = {
                    DiagnosticSeverity.CRITICAL: 4.0,
                    DiagnosticSeverity.ERROR: 3.0,
                    DiagnosticSeverity.WARNING: 2.0,
                    DiagnosticSeverity.INFO: 1.0
                }.get(result.severity, 1.0)
                
                for rec in result.recommendations:
                    recommendation_scores[rec] += severity_weight
            
            # Sort by score and return top recommendations
            sorted_recs = sorted(
                recommendation_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            return [rec for rec, score in sorted_recs[:max_recommendations]]


def create_diagnostic_system() -> SystemDiagnostics:
    """Factory function to create diagnostic system."""
    return SystemDiagnostics()


# Global diagnostic instance
_global_diagnostics = None


def get_global_diagnostics() -> SystemDiagnostics:
    """Get or create global diagnostic instance."""
    global _global_diagnostics
    if _global_diagnostics is None:
        _global_diagnostics = create_diagnostic_system()
    return _global_diagnostics