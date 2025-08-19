"""
Enhanced monitoring and observability system for tokamak RL control.

This module provides comprehensive monitoring, logging, alerting, and diagnostics
for production-ready tokamak plasma control systems.
"""

try:
    import numpy as np
except ImportError:
    import math
    import random as rand
    
    class np:
        @staticmethod
        def array(x):
            return list(x) if hasattr(x, '__iter__') else [x]
        
        @staticmethod
        def mean(arr):
            return sum(arr) / len(arr) if arr else 0
        
        @staticmethod
        def std(arr):
            if not arr:
                return 0
            mean_val = sum(arr) / len(arr)
            variance = sum((x - mean_val) ** 2 for x in arr) / len(arr)
            return math.sqrt(variance)
        
        @staticmethod
        def percentile(arr, q):
            if not arr:
                return 0
            sorted_arr = sorted(arr)
            idx = int(len(sorted_arr) * q / 100)
            return sorted_arr[min(idx, len(sorted_arr) - 1)]

import time
import logging
import threading
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import json
from datetime import datetime, timedelta

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('tokamak_rl_system.log', mode='a')
    ]
)

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class MetricType(Enum):
    """Types of metrics for monitoring."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class Alert:
    """System alert with context and severity."""
    timestamp: float
    severity: AlertSeverity
    component: str
    message: str
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    threshold: Optional[float] = None
    context: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    resolved: bool = False


@dataclass
class MetricPoint:
    """Time-series metric data point."""
    timestamp: float
    value: float
    tags: Dict[str, str] = field(default_factory=dict)


class MetricCollector:
    """Advanced metric collection and aggregation."""
    
    def __init__(self, retention_hours: int = 24, max_points_per_metric: int = 10000):
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_points_per_metric))
        self.metric_types: Dict[str, MetricType] = {}
        self.retention_seconds = retention_hours * 3600
        self.aggregation_cache: Dict[str, Dict[str, float]] = {}
        self.cache_ttl = 60  # Cache aggregations for 60 seconds
        self.last_cache_update: Dict[str, float] = {}
        self._lock = threading.RLock()
        
    def record_metric(self, name: str, value: float, metric_type: MetricType = MetricType.GAUGE,
                     tags: Dict[str, str] = None) -> None:
        """Record a metric value with timestamp and tags."""
        with self._lock:
            point = MetricPoint(
                timestamp=time.time(),
                value=value,
                tags=tags or {}
            )
            
            self.metrics[name].append(point)
            self.metric_types[name] = metric_type
            
            # Invalidate cache for this metric
            if name in self.aggregation_cache:
                del self.aggregation_cache[name]
                del self.last_cache_update[name]
    
    def get_metric_summary(self, name: str, time_window_seconds: int = 3600) -> Dict[str, float]:
        """Get statistical summary of metric over time window."""
        with self._lock:
            # Check cache first
            cache_key = f"{name}_{time_window_seconds}"
            current_time = time.time()
            
            if (cache_key in self.aggregation_cache and 
                current_time - self.last_cache_update.get(cache_key, 0) < self.cache_ttl):
                return self.aggregation_cache[cache_key]
            
            # Calculate fresh aggregations
            cutoff_time = current_time - time_window_seconds
            recent_points = [
                point.value for point in self.metrics[name]
                if point.timestamp >= cutoff_time
            ]
            
            if not recent_points:
                summary = {'count': 0, 'mean': 0, 'min': 0, 'max': 0, 'std': 0}
            else:
                summary = {
                    'count': len(recent_points),
                    'mean': np.mean(recent_points),
                    'min': min(recent_points),
                    'max': max(recent_points),
                    'std': np.std(recent_points),
                    'p50': np.percentile(recent_points, 50),
                    'p90': np.percentile(recent_points, 90),
                    'p95': np.percentile(recent_points, 95),
                    'p99': np.percentile(recent_points, 99)
                }
            
            # Cache the result
            self.aggregation_cache[cache_key] = summary
            self.last_cache_update[cache_key] = current_time
            
            return summary
    
    def get_recent_values(self, name: str, count: int = 100) -> List[Tuple[float, float]]:
        """Get recent timestamp-value pairs for a metric."""
        with self._lock:
            recent_points = list(self.metrics[name])[-count:]
            return [(point.timestamp, point.value) for point in recent_points]
    
    def cleanup_old_metrics(self) -> None:
        """Remove old metric points beyond retention period."""
        with self._lock:
            cutoff_time = time.time() - self.retention_seconds
            
            for metric_name, points in self.metrics.items():
                # Remove old points
                while points and points[0].timestamp < cutoff_time:
                    points.popleft()
            
            # Clear stale cache entries
            self.aggregation_cache.clear()
            self.last_cache_update.clear()


class AlertManager:
    """Advanced alerting system with thresholds and escalation."""
    
    def __init__(self, max_alerts: int = 1000):
        self.alerts: deque = deque(maxlen=max_alerts)
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        self.alert_callbacks: Dict[AlertSeverity, List[Callable]] = defaultdict(list)
        self.suppression_rules: Dict[str, float] = {}  # metric_name -> cooldown seconds
        self.last_alert_times: Dict[str, float] = {}
        self._lock = threading.RLock()
    
    def add_threshold_rule(self, metric_name: str, threshold: float, 
                          severity: AlertSeverity, condition: str = "greater",
                          cooldown_seconds: int = 300) -> None:
        """Add a threshold-based alerting rule."""
        with self._lock:
            self.alert_rules[metric_name] = {
                'threshold': threshold,
                'severity': severity,
                'condition': condition,  # 'greater', 'less', 'equal'
                'cooldown': cooldown_seconds
            }
            self.suppression_rules[metric_name] = cooldown_seconds
    
    def check_thresholds(self, metric_name: str, value: float) -> Optional[Alert]:
        """Check if metric value triggers any alert rules."""
        with self._lock:
            if metric_name not in self.alert_rules:
                return None
            
            rule = self.alert_rules[metric_name]
            threshold = rule['threshold']
            condition = rule['condition']
            severity = rule['severity']
            
            # Check if condition is met
            triggered = False
            if condition == "greater" and value > threshold:
                triggered = True
            elif condition == "less" and value < threshold:
                triggered = True
            elif condition == "equal" and abs(value - threshold) < 1e-6:
                triggered = True
            
            if not triggered:
                return None
            
            # Check cooldown period
            current_time = time.time()
            last_alert = self.last_alert_times.get(metric_name, 0)
            cooldown = self.suppression_rules.get(metric_name, 300)
            
            if current_time - last_alert < cooldown:
                return None  # Still in cooldown period
            
            # Create alert
            alert = Alert(
                timestamp=current_time,
                severity=severity,
                component="monitoring",
                message=f"Metric {metric_name} ({value:.3f}) {condition} threshold ({threshold:.3f})",
                metric_name=metric_name,
                metric_value=value,
                threshold=threshold
            )
            
            self.alerts.append(alert)
            self.last_alert_times[metric_name] = current_time
            
            # Trigger callbacks
            self._trigger_alert_callbacks(alert)
            
            logger.warning(f"ALERT [{severity.value.upper()}]: {alert.message}")
            
            return alert
    
    def create_custom_alert(self, severity: AlertSeverity, component: str, 
                          message: str, context: Dict[str, Any] = None) -> Alert:
        """Create a custom alert."""
        with self._lock:
            alert = Alert(
                timestamp=time.time(),
                severity=severity,
                component=component,
                message=message,
                context=context or {}
            )
            
            self.alerts.append(alert)
            self._trigger_alert_callbacks(alert)
            
            logger.warning(f"CUSTOM ALERT [{severity.value.upper()}]: {message}")
            
            return alert
    
    def register_alert_callback(self, severity: AlertSeverity, callback: Callable) -> None:
        """Register callback function for specific alert severity."""
        with self._lock:
            self.alert_callbacks[severity].append(callback)
    
    def get_active_alerts(self, severity_filter: Optional[AlertSeverity] = None,
                         time_window_hours: int = 24) -> List[Alert]:
        """Get active (unresolved) alerts within time window."""
        with self._lock:
            cutoff_time = time.time() - (time_window_hours * 3600)
            
            active_alerts = [
                alert for alert in self.alerts
                if (not alert.resolved and 
                    alert.timestamp >= cutoff_time and
                    (severity_filter is None or alert.severity == severity_filter))
            ]
            
            return sorted(active_alerts, key=lambda a: a.timestamp, reverse=True)
    
    def acknowledge_alert(self, alert_index: int) -> bool:
        """Acknowledge an alert by index."""
        with self._lock:
            if 0 <= alert_index < len(self.alerts):
                self.alerts[alert_index].acknowledged = True
                return True
            return False
    
    def resolve_alert(self, alert_index: int) -> bool:
        """Resolve an alert by index."""
        with self._lock:
            if 0 <= alert_index < len(self.alerts):
                self.alerts[alert_index].resolved = True
                return True
            return False
    
    def _trigger_alert_callbacks(self, alert: Alert) -> None:
        """Trigger registered callbacks for alert severity."""
        callbacks = self.alert_callbacks.get(alert.severity, [])
        for callback in callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")


class SystemHealthMonitor:
    """Comprehensive system health monitoring and diagnostics."""
    
    def __init__(self, check_interval_seconds: int = 30):
        self.check_interval = check_interval_seconds
        self.health_checks: Dict[str, Callable[[], Dict[str, Any]]] = {}
        self.health_status: Dict[str, Dict[str, Any]] = {}
        self.metric_collector = MetricCollector()
        self.alert_manager = AlertManager()
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        
        # Register default health checks
        self._register_default_health_checks()
        
        # Set up default alert rules
        self._setup_default_alert_rules()
    
    def register_health_check(self, name: str, check_function: Callable[[], Dict[str, Any]]) -> None:
        """Register a custom health check function."""
        with self._lock:
            self.health_checks[name] = check_function
    
    def start_monitoring(self) -> None:
        """Start background health monitoring."""
        with self._lock:
            if self.monitoring_active:
                return
            
            self.monitoring_active = True
            self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitor_thread.start()
            logger.info("System health monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop background health monitoring."""
        with self._lock:
            self.monitoring_active = False
            if self.monitor_thread:
                self.monitor_thread.join(timeout=5)
            logger.info("System health monitoring stopped")
    
    def run_health_check(self, check_name: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """Run health checks and return results."""
        with self._lock:
            if check_name:
                # Run specific check
                if check_name in self.health_checks:
                    result = self._run_single_check(check_name)
                    return {check_name: result}
                else:
                    return {}
            else:
                # Run all checks
                results = {}
                for name in self.health_checks:
                    results[name] = self._run_single_check(name)
                return results
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status summary."""
        with self._lock:
            # Get recent health check results
            health_summary = {}
            overall_healthy = True
            
            for check_name, status in self.health_status.items():
                is_healthy = status.get('healthy', False)
                health_summary[check_name] = {
                    'healthy': is_healthy,
                    'last_check': status.get('timestamp', 0),
                    'message': status.get('message', '')
                }
                overall_healthy = overall_healthy and is_healthy
            
            # Get active alerts summary
            active_alerts = self.alert_manager.get_active_alerts(time_window_hours=1)
            alert_summary = {
                'total_active': len(active_alerts),
                'by_severity': {}
            }
            
            for severity in AlertSeverity:
                count = len([a for a in active_alerts if a.severity == severity])
                alert_summary['by_severity'][severity.value] = count
            
            # Get key metrics summary
            key_metrics = ['plasma_stability', 'control_performance', 'safety_status']
            metrics_summary = {}
            for metric in key_metrics:
                summary = self.metric_collector.get_metric_summary(metric, time_window_seconds=300)
                metrics_summary[metric] = summary
            
            return {
                'overall_healthy': overall_healthy,
                'timestamp': time.time(),
                'health_checks': health_summary,
                'alerts': alert_summary,
                'key_metrics': metrics_summary,
                'monitoring_active': self.monitoring_active
            }
    
    def _register_default_health_checks(self) -> None:
        """Register default system health checks."""
        
        def memory_check() -> Dict[str, Any]:
            """Check system memory usage."""
            try:
                import psutil
                memory = psutil.virtual_memory()
                return {
                    'healthy': memory.percent < 90,
                    'memory_percent': memory.percent,
                    'available_gb': memory.available / (1024**3),
                    'message': f"Memory usage: {memory.percent:.1f}%"
                }
            except ImportError:
                # Fallback when psutil not available
                return {
                    'healthy': True,
                    'message': 'Memory check not available (psutil not installed)'
                }
        
        def disk_check() -> Dict[str, Any]:
            """Check disk space usage."""
            try:
                import psutil
                disk = psutil.disk_usage('/')
                percent_used = (disk.used / disk.total) * 100
                return {
                    'healthy': percent_used < 85,
                    'disk_percent': percent_used,
                    'free_gb': disk.free / (1024**3),
                    'message': f"Disk usage: {percent_used:.1f}%"
                }
            except ImportError:
                return {
                    'healthy': True,
                    'message': 'Disk check not available (psutil not installed)'
                }
        
        def thread_check() -> Dict[str, Any]:
            """Check active thread count."""
            try:
                import threading
                thread_count = threading.active_count()
                return {
                    'healthy': thread_count < 100,
                    'active_threads': thread_count,
                    'message': f"Active threads: {thread_count}"
                }
            except Exception:
                return {
                    'healthy': True,
                    'message': 'Thread check failed'
                }
        
        self.health_checks['memory'] = memory_check
        self.health_checks['disk'] = disk_check
        self.health_checks['threads'] = thread_check
    
    def _setup_default_alert_rules(self) -> None:
        """Set up default alerting rules for key metrics."""
        # Plasma safety alerts
        self.alert_manager.add_threshold_rule(
            'q_min', 1.5, AlertSeverity.CRITICAL, 'less'
        )
        self.alert_manager.add_threshold_rule(
            'disruption_probability', 0.1, AlertSeverity.WARNING, 'greater'
        )
        self.alert_manager.add_threshold_rule(
            'disruption_probability', 0.3, AlertSeverity.CRITICAL, 'greater'
        )
        
        # Control performance alerts
        self.alert_manager.add_threshold_rule(
            'shape_error', 5.0, AlertSeverity.WARNING, 'greater'
        )
        self.alert_manager.add_threshold_rule(
            'shape_error', 10.0, AlertSeverity.CRITICAL, 'greater'
        )
        
        # System health alerts
        self.alert_manager.add_threshold_rule(
            'control_latency_ms', 50.0, AlertSeverity.WARNING, 'greater'
        )
        self.alert_manager.add_threshold_rule(
            'control_latency_ms', 100.0, AlertSeverity.CRITICAL, 'greater'
        )
    
    def _run_single_check(self, check_name: str) -> Dict[str, Any]:
        """Run a single health check and record results."""
        try:
            check_function = self.health_checks[check_name]
            result = check_function()
            result['timestamp'] = time.time()
            result['check_name'] = check_name
            
            # Store result
            self.health_status[check_name] = result
            
            # Record metrics
            if 'healthy' in result:
                self.metric_collector.record_metric(
                    f'health_check_{check_name}',
                    1.0 if result['healthy'] else 0.0,
                    MetricType.GAUGE
                )
            
            # Check for alerts
            if not result.get('healthy', True):
                self.alert_manager.create_custom_alert(
                    AlertSeverity.WARNING,
                    f'health_check_{check_name}',
                    f"Health check failed: {result.get('message', 'Unknown issue')}",
                    result
                )
            
            return result
            
        except Exception as e:
            error_result = {
                'healthy': False,
                'error': str(e),
                'message': f'Health check {check_name} failed: {e}',
                'timestamp': time.time(),
                'check_name': check_name
            }
            
            self.health_status[check_name] = error_result
            logger.error(f"Health check {check_name} failed: {e}")
            
            return error_result
    
    def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while self.monitoring_active:
            try:
                # Run all health checks
                self.run_health_check()
                
                # Clean up old metrics
                self.metric_collector.cleanup_old_metrics()
                
                # Sleep until next check
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(self.check_interval)


class PerformanceTracker:
    """Track system and algorithm performance metrics."""
    
    def __init__(self, monitor: SystemHealthMonitor):
        self.monitor = monitor
        self.timing_context = {}
        
    def start_timer(self, operation_name: str) -> str:
        """Start timing an operation."""
        timer_id = f"{operation_name}_{time.time()}"
        self.timing_context[timer_id] = time.time()
        return timer_id
    
    def end_timer(self, timer_id: str, operation_name: str) -> float:
        """End timing and record metric."""
        if timer_id not in self.timing_context:
            return 0.0
        
        duration = time.time() - self.timing_context[timer_id]
        del self.timing_context[timer_id]
        
        # Record timing metric
        self.monitor.metric_collector.record_metric(
            f'{operation_name}_duration_ms',
            duration * 1000,
            MetricType.TIMER
        )
        
        return duration
    
    def time_operation(self, operation_name: str):
        """Decorator to time function execution."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                timer_id = self.start_timer(operation_name)
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    self.end_timer(timer_id, operation_name)
            return wrapper
        return decorator


def create_monitoring_system(check_interval: int = 30,
                           enable_auto_start: bool = True) -> SystemHealthMonitor:
    """Factory function to create a complete monitoring system."""
    monitor = SystemHealthMonitor(check_interval)
    
    if enable_auto_start:
        monitor.start_monitoring()
    
    return monitor


class PlasmaMonitor:
    """Specialized monitoring for plasma physics parameters."""
    
    def __init__(self, monitor: SystemHealthMonitor):
        self.monitor = monitor
        self.plasma_metrics = [
            'q_min', 'plasma_beta', 'shape_error', 'elongation', 'triangularity',
            'disruption_probability', 'plasma_current', 'density_profile_peak'
        ]
        
        # Register plasma-specific health checks
        self._register_plasma_health_checks()
    
    def log_plasma_state(self, plasma_state, control_action=None) -> None:
        """Log comprehensive plasma state metrics."""
        timestamp = time.time()
        
        # Record core plasma metrics
        self.monitor.metric_collector.record_metric('q_min', plasma_state.q_min)
        self.monitor.metric_collector.record_metric('plasma_beta', plasma_state.plasma_beta)
        self.monitor.metric_collector.record_metric('shape_error', plasma_state.shape_error)
        self.monitor.metric_collector.record_metric('elongation', plasma_state.elongation)
        self.monitor.metric_collector.record_metric('triangularity', plasma_state.triangularity)
        self.monitor.metric_collector.record_metric('disruption_probability', plasma_state.disruption_probability)
        
        # Check thresholds and trigger alerts
        for metric_name in ['q_min', 'plasma_beta', 'shape_error', 'disruption_probability']:
            if hasattr(plasma_state, metric_name):
                value = getattr(plasma_state, metric_name)
                self.monitor.alert_manager.check_thresholds(metric_name, value)
        
        # Log control action if provided
        if control_action is not None:
            control_effort = np.sum(np.array(control_action[:6])**2)  # PF coil effort
            self.monitor.metric_collector.record_metric('control_effort', control_effort)
    
    def _register_plasma_health_checks(self) -> None:
        """Register plasma-specific health checks."""
        
        def plasma_stability_check() -> Dict[str, Any]:
            """Check overall plasma stability."""
            try:
                q_min_summary = self.monitor.metric_collector.get_metric_summary('q_min', 300)
                beta_summary = self.monitor.metric_collector.get_metric_summary('plasma_beta', 300)
                disruption_summary = self.monitor.metric_collector.get_metric_summary('disruption_probability', 300)
                
                # Check stability criteria
                q_stable = q_min_summary.get('mean', 0) > 1.5
                beta_stable = beta_summary.get('max', 0) < 0.04
                disruption_low = disruption_summary.get('max', 0) < 0.1
                
                overall_stable = q_stable and beta_stable and disruption_low
                
                return {
                    'healthy': overall_stable,
                    'q_min_mean': q_min_summary.get('mean', 0),
                    'beta_max': beta_summary.get('max', 0),
                    'disruption_max': disruption_summary.get('max', 0),
                    'message': 'Plasma stable' if overall_stable else 'Plasma instability detected'
                }
            except Exception as e:
                return {
                    'healthy': False,
                    'message': f'Plasma stability check failed: {e}'
                }
        
        self.monitor.register_health_check('plasma_stability', plasma_stability_check)


# Global monitoring instance
_global_monitor = None


def get_global_monitor() -> SystemHealthMonitor:
    """Get or create global monitoring instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = create_monitoring_system()
    return _global_monitor