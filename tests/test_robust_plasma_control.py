"""
Tests for Robust Plasma Control System
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from tokamak_rl.robust_plasma_control import (
    SystemState,
    AlertLevel,
    SystemMetrics,
    CircuitBreaker,
    HealthMonitor,
    RobustPlasmaController,
    create_robust_plasma_control_system
)

class TestSystemMetrics:
    def test_system_metrics_creation(self):
        metrics = SystemMetrics(
            timestamp=time.time(),
            system_state="operational",
            shape_error=0.1,
            control_effort=0.5,
            safety_violations=0,
            response_time_ms=50.0,
            memory_usage_mb=100.0,
            cpu_usage_percent=25.0,
            temperature=295.0,
            magnetic_field_strength=5.3,
            plasma_current=15.0
        )
        
        assert metrics.shape_error == 0.1
        assert metrics.system_state == "operational"
        assert metrics.safety_violations == 0

class TestCircuitBreaker:
    def test_circuit_breaker_initialization(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=30.0)
        assert cb.failure_threshold == 3
        assert cb.recovery_timeout == 30.0
        assert cb.state == "CLOSED"
        assert cb.failure_count == 0

    def test_circuit_breaker_normal_operation(self):
        cb = CircuitBreaker()
        
        with cb.protected_call():
            pass  # Normal operation
        
        assert cb.state == "CLOSED"
        assert cb.failure_count == 0

    def test_circuit_breaker_failure_handling(self):
        cb = CircuitBreaker(failure_threshold=2)
        
        # First failure
        with pytest.raises(ValueError):
            with cb.protected_call():
                raise ValueError("Test error")
        
        assert cb.failure_count == 1
        assert cb.state == "CLOSED"
        
        # Second failure - should open circuit
        with pytest.raises(ValueError):
            with cb.protected_call():
                raise ValueError("Test error")
        
        assert cb.failure_count == 2
        assert cb.state == "OPEN"

    def test_circuit_breaker_open_state(self):
        cb = CircuitBreaker(failure_threshold=1)
        
        # Trigger failure to open circuit
        with pytest.raises(ValueError):
            with cb.protected_call():
                raise ValueError("Test error")
        
        # Should reject calls while open
        with pytest.raises(Exception, match="Circuit breaker is OPEN"):
            with cb.protected_call():
                pass

    def test_circuit_breaker_recovery(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)
        
        # Trigger failure
        with pytest.raises(ValueError):
            with cb.protected_call():
                raise ValueError("Test error")
        
        assert cb.state == "OPEN"
        
        # Wait for recovery timeout
        time.sleep(0.2)
        
        # Should transition to HALF_OPEN and allow successful call
        with cb.protected_call():
            pass
        
        assert cb.state == "CLOSED"
        assert cb.failure_count == 0

class TestHealthMonitor:
    def test_health_monitor_initialization(self):
        monitor = HealthMonitor(check_interval=1.0)
        assert monitor.check_interval == 1.0
        assert len(monitor.metrics_history) == 0
        assert len(monitor.alerts) == 0
        assert not monitor.is_monitoring

    def test_add_health_check(self):
        monitor = HealthMonitor()
        
        def dummy_check():
            return True
        
        monitor.add_health_check("test_check", dummy_check)
        assert len(monitor.health_checks) == 1
        assert monitor.health_checks[0][0] == "test_check"

    def test_record_metrics(self):
        monitor = HealthMonitor()
        
        metrics = SystemMetrics(
            timestamp=time.time(),
            system_state="operational",
            shape_error=0.1,
            control_effort=0.5,
            safety_violations=0,
            response_time_ms=50.0,
            memory_usage_mb=100.0,
            cpu_usage_percent=25.0,
            temperature=295.0,
            magnetic_field_strength=5.3,
            plasma_current=15.0
        )
        
        monitor.record_metrics(metrics)
        assert len(monitor.metrics_history) == 1
        assert monitor.metrics_history[0] == metrics

    def test_get_system_health_no_data(self):
        monitor = HealthMonitor()
        health = monitor.get_system_health()
        
        assert health["status"] == "NO_DATA"
        assert "alerts" in health

    def test_get_system_health_with_data(self):
        monitor = HealthMonitor()
        
        # Add some metrics
        for i in range(5):
            metrics = SystemMetrics(
                timestamp=time.time() - (4-i),
                system_state="operational",
                shape_error=0.1 + i*0.01,
                control_effort=0.5,
                safety_violations=0,
                response_time_ms=50.0,
                memory_usage_mb=100.0,
                cpu_usage_percent=25.0,
                temperature=295.0,
                magnetic_field_strength=5.3,
                plasma_current=15.0
            )
            monitor.record_metrics(metrics)
        
        health = monitor.get_system_health()
        
        assert health["status"] == "HEALTHY"
        assert health["avg_shape_error"] > 0
        assert health["avg_response_time"] == 50.0
        assert health["recent_alerts"] == 0

    def test_threshold_checking(self):
        monitor = HealthMonitor()
        monitor.thresholds['shape_error_critical'] = 0.05
        
        # Create metrics that exceed threshold
        metrics = SystemMetrics(
            timestamp=time.time(),
            system_state="operational",
            shape_error=0.1,  # Above critical threshold
            control_effort=0.5,
            safety_violations=0,
            response_time_ms=50.0,
            memory_usage_mb=100.0,
            cpu_usage_percent=25.0,
            temperature=295.0,
            magnetic_field_strength=5.3,
            plasma_current=15.0
        )
        
        monitor.record_metrics(metrics)
        monitor._check_thresholds(metrics)
        
        # Should have created an alert
        assert len(monitor.alerts) > 0
        assert monitor.alerts[-1]['level'] == 'critical'

    def test_monitoring_thread(self):
        monitor = HealthMonitor(check_interval=0.1)
        
        # Add a health check that fails
        def failing_check():
            return False
        
        monitor.add_health_check("failing_test", failing_check)
        
        # Start monitoring briefly
        monitor.start_monitoring()
        time.sleep(0.3)  # Let it run a few checks
        monitor.stop_monitoring()
        
        # Should have created alerts for failing check
        assert len(monitor.alerts) > 0

class TestRobustPlasmaController:
    def test_controller_initialization(self):
        controller = RobustPlasmaController()
        
        assert controller.state == SystemState.OPERATIONAL
        assert controller.circuit_breaker is not None
        assert controller.health_monitor is not None
        assert controller.control_metrics['total_steps'] == 0

    def test_controller_with_config(self):
        config = {
            'failure_threshold': 10,
            'recovery_timeout': 120.0,
            'health_check_interval': 10.0
        }
        
        controller = RobustPlasmaController(config)
        assert controller.config == config
        assert controller.circuit_breaker.failure_threshold == 10
        assert controller.circuit_breaker.recovery_timeout == 120.0

    def test_successful_control_step(self):
        controller = RobustPlasmaController()
        
        observation = [1.5, 0.3] + [1.8] * 8 + [1e19] * 35  # 45-dim observation
        result = controller.control_step_robust(observation)
        
        assert result['success'] is True
        assert 'action' in result
        assert len(result['action']) == 8
        assert 'shape_error' in result
        assert 'control_effort' in result
        assert controller.control_metrics['total_steps'] == 1
        assert controller.control_metrics['successful_steps'] == 1

    def test_invalid_observation_handling(self):
        controller = RobustPlasmaController()
        
        # Test with insufficient observation dimension
        invalid_observation = [1.0, 2.0]  # Only 2 dimensions
        result = controller.control_step_robust(invalid_observation)
        
        # Should handle error gracefully
        assert result['success'] is False
        assert 'fallback_reason' in result
        assert controller.control_metrics['failed_steps'] == 1

    def test_safety_checks(self):
        controller = RobustPlasmaController()
        
        observation = [15.0, 15.0] + [1.8] * 43  # Out of bounds observation
        action = [2.0] * 8  # Out of bounds action
        
        safety_violation = controller._perform_safety_checks(observation, action)
        assert safety_violation is True

    def test_fallback_control(self):
        controller = RobustPlasmaController()
        
        # Set some previous action
        controller.last_action = [0.5, -0.3, 0.8, 0.2, -0.1, 0.6, 0.4, 0.7]
        
        observation = [1.5, 0.3] * 10  # Some observation
        error = ValueError("Test error")
        
        result = controller._execute_fallback_control(observation, error)
        
        assert result['success'] is False
        assert result['shape_error'] == 999.0  # Fallback indicator
        assert 'fallback_reason' in result
        assert len(result['action']) == 8

    def test_emergency_procedures(self):
        controller = RobustPlasmaController()
        
        # Test plasma disruption procedure
        result = controller.trigger_emergency_procedure('plasma_disruption')
        assert result['success'] is True
        assert result['action'] == 'emergency_shutdown'
        assert controller.state == SystemState.EMERGENCY

    def test_unknown_emergency_procedure(self):
        controller = RobustPlasmaController()
        
        result = controller.trigger_emergency_procedure('unknown_procedure')
        assert result['success'] is False
        assert result['action'] == 'unknown_procedure'

    def test_system_status(self):
        controller = RobustPlasmaController()
        
        # Execute a few control steps
        observation = [1.5, 0.3] * 20 + [1.8] * 5
        for _ in range(3):
            controller.control_step_robust(observation)
        
        status = controller.get_system_status()
        
        assert 'system_state' in status
        assert 'uptime_seconds' in status
        assert 'control_metrics' in status
        assert 'health_status' in status
        assert status['control_metrics']['total_steps'] == 3

    def test_diagnostics_export(self):
        controller = RobustPlasmaController()
        
        # Execute some steps to generate data
        observation = [1.5, 0.3] * 20 + [1.8] * 5
        for _ in range(2):
            controller.control_step_robust(observation)
        
        # Export diagnostics
        filename = controller.export_diagnostics("test_diagnostics.json")
        
        if filename:
            assert filename.endswith(".json")
            # Clean up
            try:
                os.remove(filename)
            except:
                pass

    def test_monitoring_integration(self):
        controller = RobustPlasmaController()
        controller.start_monitoring()
        
        # Execute some control steps
        observation = [1.5, 0.3] * 20 + [1.8] * 5
        for _ in range(5):
            controller.control_step_robust(observation)
        
        # Let monitoring run briefly
        time.sleep(0.1)
        
        controller.stop_monitoring()
        
        # Should have recorded metrics
        assert len(controller.health_monitor.metrics_history) > 0

class TestFactoryFunction:
    def test_create_default_system(self):
        controller = create_robust_plasma_control_system()
        
        assert isinstance(controller, RobustPlasmaController)
        assert controller.health_monitor.is_monitoring  # Should start monitoring by default
        
        controller.stop_monitoring()  # Cleanup

    def test_create_custom_system(self):
        config = {
            'failure_threshold': 5,
            'enable_monitoring': False
        }
        
        controller = create_robust_plasma_control_system(config)
        
        assert controller.circuit_breaker.failure_threshold == 5
        assert not controller.health_monitor.is_monitoring  # Monitoring disabled

class TestIntegrationScenarios:
    def test_complete_failure_recovery_scenario(self):
        """Test complete failure and recovery scenario"""
        config = {
            'failure_threshold': 2,
            'recovery_timeout': 0.1
        }
        
        controller = create_robust_plasma_control_system(config)
        
        try:
            # Normal operation
            normal_obs = [1.5, 0.3] + [1.8] * 43
            result = controller.control_step_robust(normal_obs)
            assert result['success'] is True
            
            # Trigger failures to open circuit breaker
            invalid_obs = [999.0] * 2  # Invalid observation
            
            for _ in range(3):  # Trigger multiple failures
                result = controller.control_step_robust(invalid_obs)
            
            assert controller.circuit_breaker.state == "OPEN"
            
            # Wait for recovery
            time.sleep(0.2)
            
            # Should recover with normal operation
            result = controller.control_step_robust(normal_obs)
            assert result['success'] is True
            assert controller.circuit_breaker.state == "CLOSED"
            
        finally:
            controller.stop_monitoring()

    def test_high_load_stress_test(self):
        """Test system under high load"""
        controller = create_robust_plasma_control_system()
        
        try:
            successful_steps = 0
            total_steps = 50
            
            for i in range(total_steps):
                observation = [1.5 + 0.1*i, 0.3] + [1.8] * 43
                result = controller.control_step_robust(observation)
                
                if result['success']:
                    successful_steps += 1
            
            success_rate = successful_steps / total_steps
            assert success_rate > 0.8  # Should have >80% success rate
            
            status = controller.get_system_status()
            assert status['control_metrics']['total_steps'] == total_steps
            
        finally:
            controller.stop_monitoring()

    def test_concurrent_monitoring_and_control(self):
        """Test concurrent monitoring and control operations"""
        controller = create_robust_plasma_control_system()
        
        try:
            # Start control in separate thread
            def control_loop():
                for _ in range(10):
                    observation = [1.5, 0.3] + [1.8] * 43
                    controller.control_step_robust(observation)
                    time.sleep(0.01)
            
            control_thread = threading.Thread(target=control_loop)
            control_thread.start()
            
            # Monitor system while control is running
            time.sleep(0.2)
            
            control_thread.join()
            
            # Verify both operations completed successfully
            status = controller.get_system_status()
            assert status['control_metrics']['total_steps'] == 10
            assert len(controller.health_monitor.metrics_history) > 0
            
        finally:
            controller.stop_monitoring()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])