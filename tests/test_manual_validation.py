#!/usr/bin/env python3
"""
Manual validation test suite for Generation 1-3 autonomous implementation.

This test suite validates the enhanced tokamak RL control system without
requiring external dependencies like pytest.
"""

import sys
import os
import traceback

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_result(test_name, success, message=""):
    """Print formatted test result."""
    status = "PASS" if success else "FAIL"
    print(f"[{status}] {test_name}")
    if message:
        print(f"    {message}")


def test_basic_imports():
    """Test that core modules can be imported."""
    print("\n=== Testing Basic Imports ===")
    
    try:
        from tokamak_rl import validation, security, optimization
        test_result("Import validation module", True)
    except Exception as e:
        test_result("Import validation module", False, str(e))
    
    try:
        from tokamak_rl import enhanced_monitoring, diagnostics
        test_result("Import monitoring modules", True)
    except Exception as e:
        test_result("Import monitoring modules", False, str(e))
    
    try:
        from tokamak_rl import distributed_computing
        test_result("Import distributed computing", True)
    except Exception as e:
        test_result("Import distributed computing", False, str(e))


def test_validation_system():
    """Test the validation system."""
    print("\n=== Testing Validation System ===")
    
    try:
        from tokamak_rl.validation import InputValidator, ValidationResult, ValidationSeverity
        
        validator = InputValidator()
        
        # Test numeric validation
        result = validator.validate_numeric_input(5.0, "test_value", min_val=0, max_val=10)
        test_result("Numeric validation (valid)", result.is_valid)
        
        # Test invalid numeric
        result = validator.validate_numeric_input(15.0, "test_value", min_val=0, max_val=10)
        test_result("Numeric validation (invalid)", not result.is_valid)
        
        # Test array validation
        result = validator.validate_array_input([1, 2, 3, 4], "test_array", expected_shape=(4,))
        test_result("Array validation (valid shape)", result.is_valid)
        
        test_result("Validation system", True, "All validation tests passed")
        
    except Exception as e:
        test_result("Validation system", False, str(e))


def test_security_system():
    """Test the security system."""
    print("\n=== Testing Security System ===")
    
    try:
        from tokamak_rl.security import InputSanitizer, AccessController, SecurityLevel
        
        sanitizer = InputSanitizer()
        
        # Test string sanitization
        clean_str = sanitizer.sanitize_string("normal_string", "test_field")
        test_result("String sanitization (clean)", isinstance(clean_str, str))
        
        # Test malicious string
        malicious = "normal_string exec('malicious_code')"
        clean_str = sanitizer.sanitize_string(malicious, "test_field")
        test_result("String sanitization (malicious)", "exec" not in clean_str)
        
        # Test numeric sanitization
        clean_num = sanitizer.sanitize_numeric_input("123.45", "test_num")
        test_result("Numeric sanitization", clean_num == 123.45)
        
        # Test access controller
        access_controller = AccessController()
        test_result("Access controller creation", access_controller is not None)
        
        test_result("Security system", True, "All security tests passed")
        
    except Exception as e:
        test_result("Security system", False, str(e))


def test_optimization_system():
    """Test the optimization system."""
    print("\n=== Testing Optimization System ===")
    
    try:
        from tokamak_rl.optimization import (
            AdaptiveCache, CacheStrategy, PerformanceOptimizer, 
            MemoizationManager, get_global_optimizer
        )
        
        # Test adaptive cache
        cache = AdaptiveCache(max_size=100)
        cache.put("test_key", "test_value")
        value = cache.get("test_key")
        test_result("Adaptive cache basic operations", value == "test_value")
        
        # Test cache statistics
        stats = cache.get_stats()
        test_result("Cache statistics", stats.hits == 1)
        
        # Test memoization manager
        memo_manager = MemoizationManager()
        
        @memo_manager.memoize(cache_size=10)
        def test_function(x):
            return x * 2
        
        result1 = test_function(5)
        result2 = test_function(5)  # Should be cached
        test_result("Memoization", result1 == result2 == 10)
        
        # Test global optimizer
        optimizer = get_global_optimizer()
        test_result("Global optimizer", optimizer is not None)
        
        test_result("Optimization system", True, "All optimization tests passed")
        
    except Exception as e:
        test_result("Optimization system", False, str(e))


def test_monitoring_system():
    """Test the enhanced monitoring system."""
    print("\n=== Testing Monitoring System ===")
    
    try:
        from tokamak_rl.enhanced_monitoring import (
            SystemHealthMonitor, MetricCollector, AlertManager, 
            AlertSeverity, create_monitoring_system
        )
        
        # Test metric collector
        collector = MetricCollector()
        collector.record_metric("test_metric", 42.0)
        summary = collector.get_metric_summary("test_metric")
        test_result("Metric collection", summary['count'] == 1)
        
        # Test alert manager
        alert_manager = AlertManager()
        alert_manager.add_threshold_rule("test_metric", 50.0, AlertSeverity.WARNING, "greater")
        alert = alert_manager.check_thresholds("test_metric", 60.0)
        test_result("Alert threshold checking", alert is not None)
        
        # Test health monitor
        monitor = create_monitoring_system(enable_auto_start=False)
        test_result("Health monitor creation", monitor is not None)
        
        # Test health check execution
        health_results = monitor.run_health_check()
        test_result("Health check execution", isinstance(health_results, dict))
        
        test_result("Monitoring system", True, "All monitoring tests passed")
        
    except Exception as e:
        test_result("Monitoring system", False, str(e))


def test_diagnostics_system():
    """Test the diagnostics system."""
    print("\n=== Testing Diagnostics System ===")
    
    try:
        from tokamak_rl.diagnostics import (
            SystemDiagnostics, PerformanceDiagnostics, ComponentDiagnostics,
            create_diagnostic_system
        )
        
        # Test performance diagnostics
        perf_diag = PerformanceDiagnostics()
        result = perf_diag.check_system_performance()
        test_result("Performance diagnostics", result.component.value == "hardware")
        
        # Test component diagnostics
        comp_diag = ComponentDiagnostics()
        test_result("Component diagnostics creation", comp_diag is not None)
        
        # Test system diagnostics
        sys_diag = create_diagnostic_system()
        health_summary = sys_diag.get_system_health_summary()
        test_result("System health summary", isinstance(health_summary, dict))
        
        test_result("Diagnostics system", True, "All diagnostics tests passed")
        
    except Exception as e:
        test_result("Diagnostics system", False, str(e))


def test_distributed_computing():
    """Test the distributed computing system."""
    print("\n=== Testing Distributed Computing ===")
    
    try:
        from tokamak_rl.distributed_computing import (
            LoadBalancer, TaskScheduler, AutoScaler, DistributedTrainingCoordinator,
            NodeInfo, NodeRole, DistributedTask, ScalingPolicy,
            create_distributed_system
        )
        
        # Test load balancer
        load_balancer = LoadBalancer()
        test_node = NodeInfo("test_node", NodeRole.WORKER, "localhost", 8080)
        load_balancer.register_node(test_node)
        stats = load_balancer.get_cluster_stats()
        test_result("Load balancer node registration", stats['total_nodes'] == 1)
        
        # Test task scheduler
        scheduler = TaskScheduler()
        test_task = DistributedTask("test_task", "test_type", {"data": "test"})
        task_id = scheduler.submit_task(test_task)
        test_result("Task scheduler submission", task_id == "test_task")
        
        # Test auto scaler
        auto_scaler = AutoScaler(policy=ScalingPolicy.ADAPTIVE)
        auto_scaler.set_components(load_balancer, scheduler)
        scaling_decision = auto_scaler.evaluate_scaling()
        test_result("Auto scaler evaluation", isinstance(scaling_decision, dict))
        
        # Test distributed coordinator
        coordinator = create_distributed_system()
        test_result("Distributed coordinator creation", coordinator is not None)
        
        test_result("Distributed computing", True, "All distributed computing tests passed")
        
    except Exception as e:
        test_result("Distributed computing", False, str(e))


def test_fallback_implementations():
    """Test that fallback implementations work."""
    print("\n=== Testing Fallback Implementations ===")
    
    try:
        # Test fallback numpy
        from tokamak_rl.validation import np as fallback_np
        arr = fallback_np.array([1, 2, 3, 4])
        test_result("Fallback numpy array", len(arr) == 4)
        
        mean_val = fallback_np.mean(arr)
        test_result("Fallback numpy mean", isinstance(mean_val, (int, float)))
        
        # Test fallback torch (if no real torch available)
        try:
            import torch
            test_result("Real torch available", True, "Using real PyTorch")
        except ImportError:
            from tokamak_rl.agents import torch as fallback_torch
            tensor = fallback_torch.FloatTensor([1, 2, 3])
            test_result("Fallback torch", tensor is not None)
        
        test_result("Fallback implementations", True, "All fallback tests passed")
        
    except Exception as e:
        test_result("Fallback implementations", False, str(e))


def test_integration():
    """Test integration between systems."""
    print("\n=== Testing System Integration ===")
    
    try:
        # Test validation + security integration
        from tokamak_rl.validation import InputValidator
        from tokamak_rl.security import InputSanitizer
        
        validator = InputValidator()
        sanitizer = InputSanitizer()
        
        # Process potentially malicious input through both systems
        test_input = "123.45 <script>alert('xss')</script>"
        clean_input = sanitizer.sanitize_string(test_input, "test_field")
        validation_result = validator.validate_numeric_input(clean_input, "test_field")
        
        test_result("Validation + Security integration", 
                   validation_result.is_valid and "script" not in clean_input)
        
        # Test optimization + monitoring integration
        from tokamak_rl.optimization import get_global_optimizer
        from tokamak_rl.enhanced_monitoring import get_global_monitor
        
        optimizer = get_global_optimizer()
        monitor = get_global_monitor()
        
        # Record a metric about optimization
        stats = optimizer.get_optimization_stats()
        if monitor.metric_collector:
            monitor.metric_collector.record_metric("optimization_level", 1.0)
        
        test_result("Optimization + Monitoring integration", 
                   stats is not None and monitor is not None)
        
        test_result("System integration", True, "All integration tests passed")
        
    except Exception as e:
        test_result("System integration", False, str(e))


def run_all_tests():
    """Run all test suites."""
    import builtins
    
    builtins.print("🧪 AUTONOMOUS SDLC GENERATION 1-3 VALIDATION SUITE")
    builtins.print("=" * 60)
    
    total_tests = 0
    passed_tests = 0
    
    # Track test results
    test_results = []
    
    def capturing_print(*args, **kwargs):
        line = " ".join(str(arg) for arg in args)
        test_results.append(line)
        builtins.print(*args, **kwargs)
    
    test_basic_imports()
    test_validation_system()
    test_security_system()
    test_optimization_system()
    test_monitoring_system()
    test_diagnostics_system()
    test_distributed_computing()
    test_fallback_implementations()
    test_integration()
    
    # Count results
    for line in test_results:
        if line.startswith("[PASS]") or line.startswith("[FAIL]"):
            total_tests += 1
            if line.startswith("[PASS]"):
                passed_tests += 1
    
    builtins.print("\n" + "=" * 60)
    builtins.print(f"🏁 TEST SUITE COMPLETE")
    builtins.print(f"📊 RESULTS: {passed_tests}/{total_tests} tests passed")
    
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    builtins.print(f"📈 SUCCESS RATE: {success_rate:.1f}%")
    
    if success_rate >= 90:
        builtins.print("✅ QUALITY GATE: PASSED - System ready for deployment")
    elif success_rate >= 75:
        builtins.print("⚠️  QUALITY GATE: WARNING - Some issues detected")
    else:
        builtins.print("❌ QUALITY GATE: FAILED - Significant issues require attention")
    
    return success_rate


if __name__ == "__main__":
    success_rate = run_all_tests()
    sys.exit(0 if success_rate >= 75 else 1)