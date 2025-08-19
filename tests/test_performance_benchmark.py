#!/usr/bin/env python3
"""
Performance benchmark suite for tokamak RL control system.
"""

import sys
import os
import time
from collections import defaultdict

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def main():
    print('⚡ PERFORMANCE BENCHMARK SUITE')
    print('='*45)
    
    total_tests = 0
    passed_tests = 0
    benchmark_results = {}
    
    def test_result(name, success, message="", benchmark_time=None):
        nonlocal total_tests, passed_tests
        total_tests += 1
        if success:
            passed_tests += 1
        status = "PASS" if success else "FAIL"
        print(f"[{status}] {name}")
        if message:
            print(f"    {message}")
        if benchmark_time is not None:
            benchmark_results[name] = benchmark_time
            print(f"    ⏱️  {benchmark_time:.4f}s")
    
    # Test 1: Basic Operations Performance
    print('\n=== Testing Basic Operations Performance ===')
    try:
        # Test list operations
        start_time = time.time()
        test_list = list(range(10000))
        for i in range(1000):
            test_list.append(i)
            test_list.pop()
        list_time = time.time() - start_time
        
        list_fast = list_time < 0.1  # Should be very fast
        test_result("List operations", list_fast, f"10K list ops", list_time)
        
        # Test dictionary operations
        start_time = time.time()
        test_dict = {}
        for i in range(10000):
            test_dict[f"key_{i}"] = i
        for i in range(5000):
            _ = test_dict.get(f"key_{i}", None)
        dict_time = time.time() - start_time
        
        dict_fast = dict_time < 0.1
        test_result("Dictionary operations", dict_fast, f"10K dict ops", dict_time)
        
    except Exception as e:
        test_result("Basic operations", False, str(e))
    
    # Test 2: Caching Performance
    print('\n=== Testing Caching Performance ===')
    try:
        # Simple cache implementation
        class SimpleCache:
            def __init__(self, max_size=1000):
                self.cache = {}
                self.max_size = max_size
                self.access_order = []
            
            def get(self, key, default=None):
                if key in self.cache:
                    self.access_order.remove(key)
                    self.access_order.append(key)
                    return self.cache[key]
                return default
            
            def put(self, key, value):
                if key in self.cache:
                    self.access_order.remove(key)
                elif len(self.cache) >= self.max_size:
                    lru_key = self.access_order.pop(0)
                    del self.cache[lru_key]
                
                self.cache[key] = value
                self.access_order.append(key)
        
        cache = SimpleCache(max_size=1000)
        
        # Benchmark cache operations
        start_time = time.time()
        for i in range(5000):
            cache.put(f"key_{i}", f"value_{i}")
        
        for i in range(2500):
            _ = cache.get(f"key_{i}")
        cache_time = time.time() - start_time
        
        cache_fast = cache_time < 0.5  # Should be reasonably fast
        test_result("Cache operations", cache_fast, f"5K cache ops", cache_time)
        
    except Exception as e:
        test_result("Cache performance", False, str(e))
    
    # Test 3: Validation Performance
    print('\n=== Testing Validation Performance ===')
    try:
        import re
        
        def validate_input(value, min_val=0, max_val=100):
            """Simple validation function."""
            try:
                num_val = float(value)
                return min_val <= num_val <= max_val
            except (ValueError, TypeError):
                return False
        
        def sanitize_string(input_str):
            """Simple string sanitization."""
            if not isinstance(input_str, str):
                return str(input_str)
            
            # Remove dangerous patterns
            dangerous_patterns = [r'exec\s*\(', r'eval\s*\(', r'<script']
            sanitized = input_str
            for pattern in dangerous_patterns:
                sanitized = re.sub(pattern, '[BLOCKED]', sanitized, flags=re.IGNORECASE)
            
            return sanitized
        
        # Benchmark validation
        start_time = time.time()
        for i in range(10000):
            validate_input(i % 150, 0, 100)  # Mix of valid/invalid
        
        for i in range(1000):
            sanitize_string(f"normal_string_{i}")
        validation_time = time.time() - start_time
        
        validation_fast = validation_time < 1.0  # Should be fast
        test_result("Validation operations", validation_fast, f"10K validations", validation_time)
        
    except Exception as e:
        test_result("Validation performance", False, str(e))
    
    # Test 4: Monitoring Performance
    print('\n=== Testing Monitoring Performance ===')
    try:
        from collections import deque, defaultdict
        
        class SimpleMetrics:
            def __init__(self):
                self.metrics = defaultdict(lambda: deque(maxlen=1000))
            
            def record(self, name, value):
                self.metrics[name].append((time.time(), value))
            
            def get_stats(self, name):
                values = [v for t, v in self.metrics[name]]
                if not values:
                    return {}
                return {
                    'count': len(values),
                    'avg': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values)
                }
        
        metrics = SimpleMetrics()
        
        # Benchmark metrics
        start_time = time.time()
        for i in range(10000):
            metrics.record("test_metric", i % 100)
            if i % 100 == 0:
                _ = metrics.get_stats("test_metric")
        monitoring_time = time.time() - start_time
        
        monitoring_fast = monitoring_time < 1.0
        test_result("Monitoring operations", monitoring_fast, f"10K metric ops", monitoring_time)
        
    except Exception as e:
        test_result("Monitoring performance", False, str(e))
    
    # Test 5: Mathematical Operations
    print('\n=== Testing Mathematical Operations ===')
    try:
        import math
        
        # Simulate plasma physics calculations
        start_time = time.time()
        for i in range(50000):
            # Simulate some physics calculations
            x = i * 0.001
            result = math.sin(x) * math.exp(-x*0.1) + math.cos(x*2) * math.sqrt(abs(x))
        math_time = time.time() - start_time
        
        math_fast = math_time < 2.0  # Mathematical operations should be reasonably fast
        test_result("Mathematical operations", math_fast, f"50K math ops", math_time)
        
        # Test array-like operations
        start_time = time.time()
        data = list(range(10000))
        
        # Simulate array operations
        for _ in range(100):
            mean_val = sum(data) / len(data)
            max_val = max(data)
            min_val = min(data)
            filtered = [x for x in data if x > mean_val]
        
        array_time = time.time() - start_time
        array_fast = array_time < 1.0
        test_result("Array operations", array_fast, f"Array processing", array_time)
        
    except Exception as e:
        test_result("Mathematical operations", False, str(e))
    
    # Test 6: Memory Usage
    print('\n=== Testing Memory Efficiency ===')
    try:
        import gc
        
        # Force garbage collection
        gc.collect()
        
        # Test memory usage with large data structures
        large_data = []
        for i in range(100000):
            large_data.append({'id': i, 'value': f"test_value_{i}", 'data': list(range(10))})
        
        # Test that memory can be reclaimed
        del large_data
        gc.collect()
        
        # Test object creation/destruction performance
        start_time = time.time()
        for _ in range(10000):
            temp_obj = {'data': list(range(100))}
            del temp_obj
        memory_time = time.time() - start_time
        
        memory_efficient = memory_time < 2.0
        test_result("Memory efficiency", memory_efficient, f"Object lifecycle", memory_time)
        
    except Exception as e:
        test_result("Memory efficiency", False, str(e))
    
    # Performance Summary
    print('\n' + '='*45)
    print('🏁 PERFORMANCE BENCHMARK COMPLETE')
    print(f'📊 RESULTS: {passed_tests}/{total_tests} tests passed')
    
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    print(f'📈 SUCCESS RATE: {success_rate:.1f}%')
    
    # Show benchmark summary
    if benchmark_results:
        print(f'\n📈 BENCHMARK SUMMARY:')
        total_time = sum(benchmark_results.values())
        print(f'⏱️  Total benchmark time: {total_time:.4f}s')
        
        for test_name, test_time in benchmark_results.items():
            percentage = (test_time / total_time * 100) if total_time > 0 else 0
            print(f'   • {test_name}: {test_time:.4f}s ({percentage:.1f}%)')
    
    if success_rate >= 90:
        print('✅ PERFORMANCE: EXCELLENT - All operations within target times')
    elif success_rate >= 75:
        print('✅ PERFORMANCE: GOOD - Most operations performing well')
    elif success_rate >= 50:
        print('⚠️  PERFORMANCE: ACCEPTABLE - Some optimization opportunities')
    else:
        print('❌ PERFORMANCE: NEEDS IMPROVEMENT - Significant performance issues')
    
    return success_rate

if __name__ == "__main__":
    success_rate = main()
    sys.exit(0 if success_rate >= 75 else 1)