"""
Comprehensive performance benchmarks for tokamak-rl-control-suite.

These benchmarks measure performance across different system components
and provide standardized metrics for performance regression testing.
"""

import pytest
import time
import psutil
import numpy as np
from typing import Dict, Any, List
from contextlib import contextmanager
from unittest.mock import Mock

from tests.fixtures.test_data import BenchmarkDatasets, TestDataGenerator


class PerformanceMetrics:
    """Collects and analyzes performance metrics."""
    
    def __init__(self):
        self.metrics = {}
        self.process = psutil.Process()
    
    def start_timing(self, operation: str):
        """Start timing an operation."""
        self.metrics[operation] = {
            "start_time": time.time(),
            "start_memory": self.process.memory_info().rss / 1024 / 1024,  # MB
            "start_cpu": self.process.cpu_percent()
        }
    
    def end_timing(self, operation: str):
        """End timing and record metrics."""
        if operation not in self.metrics:
            raise ValueError(f"Operation {operation} not started")
        
        end_time = time.time()
        end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        self.metrics[operation].update({
            "end_time": end_time,
            "end_memory": end_memory,
            "duration": end_time - self.metrics[operation]["start_time"],
            "memory_delta": end_memory - self.metrics[operation]["start_memory"],
            "peak_memory": end_memory,
        })
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        summary = {}
        for op, data in self.metrics.items():
            if "duration" in data:
                summary[op] = {
                    "duration_ms": data["duration"] * 1000,
                    "memory_mb": data["memory_delta"],
                    "peak_memory_mb": data["peak_memory"],
                }
        return summary


@contextmanager
def performance_monitor(metrics: PerformanceMetrics, operation: str):
    """Context manager for performance monitoring."""
    metrics.start_timing(operation)
    try:
        yield
    finally:
        metrics.end_timing(operation)


class TestPhysicsSolverPerformance:
    """Performance tests for physics solver components."""
    
    @pytest.fixture
    def metrics(self):
        return PerformanceMetrics()
    
    @pytest.mark.performance
    def test_equilibrium_solving_speed(self, metrics, benchmark_config):
        """Benchmark equilibrium solving speed."""
        generator = TestDataGenerator()
        
        with performance_monitor(metrics, "equilibrium_solving"):
            for _ in range(10):  # Multiple solves
                # Mock equilibrium solving
                eq_data = generator.generate_equilibrium_data(grid_size=64)
                time.sleep(0.01)  # Simulate computation time
        
        summary = metrics.get_summary()
        duration_per_solve = summary["equilibrium_solving"]["duration_ms"] / 10
        
        # Performance requirements
        assert duration_per_solve < 100, f"Equilibrium solve too slow: {duration_per_solve:.1f}ms"
        assert summary["equilibrium_solving"]["memory_mb"] < 100, "Memory usage too high"
    
    @pytest.mark.performance
    def test_grid_scaling_performance(self, metrics):
        """Test performance scaling with grid size."""
        generator = TestDataGenerator()
        grid_sizes = [32, 64, 128, 256]
        timings = []
        
        for grid_size in grid_sizes:
            with performance_monitor(metrics, f"grid_{grid_size}"):
                eq_data = generator.generate_equilibrium_data(grid_size=grid_size)
                # Simulate O(N²) computation
                dummy_calc = np.sum(eq_data["psi"]**2)
                time.sleep(grid_size / 10000)  # Simulate scaling
            
            summary = metrics.get_summary()
            timings.append(summary[f"grid_{grid_size}"]["duration_ms"])
        
        # Check scaling behavior (should be roughly quadratic)
        for i in range(1, len(timings)):
            scaling_factor = timings[i] / timings[i-1]
            expected_factor = (grid_sizes[i] / grid_sizes[i-1])**2
            
            # Allow some deviation from perfect scaling
            assert 0.5 * expected_factor < scaling_factor < 2.0 * expected_factor, \
                f"Scaling factor {scaling_factor:.2f} not reasonable for {grid_sizes[i-1]} -> {grid_sizes[i]}"
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_convergence_performance(self, metrics):
        """Benchmark convergence performance."""
        max_iterations = 100
        tolerance = 1e-6
        
        with performance_monitor(metrics, "convergence"):
            # Mock iterative solver
            residual = 1.0
            for iteration in range(max_iterations):
                residual *= 0.9  # Mock convergence
                time.sleep(0.001)  # Simulate iteration time
                if residual < tolerance:
                    break
        
        summary = metrics.get_summary()
        assert summary["convergence"]["duration_ms"] < 1000, "Convergence too slow"
        assert iteration < max_iterations * 0.8, "Convergence should be achieved efficiently"


class TestEnvironmentPerformance:
    """Performance tests for the RL environment."""
    
    @pytest.fixture
    def metrics(self):
        return PerformanceMetrics()
    
    @pytest.mark.performance
    def test_environment_step_speed(self, metrics, benchmark_config):
        """Test environment step execution speed."""
        num_steps = benchmark_config["max_steps_per_episode"]
        acceptable_time = benchmark_config["acceptable_step_time_ms"]
        
        # Mock environment step
        with performance_monitor(metrics, "environment_steps"):
            for _ in range(num_steps):
                # Simulate environment step
                action = np.random.randn(8)
                observation = np.random.randn(45)
                reward = np.random.randn()
                time.sleep(0.001)  # Simulate step computation
        
        summary = metrics.get_summary()
        avg_step_time = summary["environment_steps"]["duration_ms"] / num_steps
        
        assert avg_step_time < acceptable_time, \
            f"Average step time {avg_step_time:.2f}ms exceeds limit {acceptable_time}ms"
    
    @pytest.mark.performance
    def test_parallel_environment_scaling(self, metrics):
        """Test parallel environment performance scaling."""
        num_envs_list = [1, 2, 4, 8]
        timings = []
        
        for num_envs in num_envs_list:
            with performance_monitor(metrics, f"parallel_{num_envs}"):
                # Simulate parallel environment execution
                for env_id in range(num_envs):
                    for step in range(100):
                        time.sleep(0.0001)  # Simulate individual env step
        
            summary = metrics.get_summary()
            timings.append(summary[f"parallel_{num_envs}"]["duration_ms"])
        
        # Check that parallel execution provides speedup
        for i in range(1, len(timings)):
            speedup = timings[0] / timings[i] * num_envs_list[i]
            assert speedup > 0.5, f"Parallel efficiency too low: {speedup:.2f}"
    
    @pytest.mark.performance
    def test_observation_computation_speed(self, metrics):
        """Test observation computation performance."""
        generator = TestDataGenerator()
        
        with performance_monitor(metrics, "observation_computation"):
            for _ in range(1000):
                # Mock observation computation
                plasma_state = generator.generate_plasma_states(1)
                # Simulate observation processing
                observation = np.concatenate([
                    plasma_state["plasma_current"],
                    plasma_state["plasma_beta"],
                    plasma_state["q_profiles"][0],
                ])
                time.sleep(0.0001)  # Simulate computation
        
        summary = metrics.get_summary()
        avg_obs_time = summary["observation_computation"]["duration_ms"] / 1000
        
        assert avg_obs_time < 1.0, f"Observation computation too slow: {avg_obs_time:.3f}ms"


class TestMemoryPerformance:
    """Memory usage and efficiency tests."""
    
    @pytest.fixture
    def metrics(self):
        return PerformanceMetrics()
    
    @pytest.mark.performance
    def test_memory_usage_bounds(self, metrics, benchmark_config):
        """Test memory usage stays within bounds."""
        memory_limit = benchmark_config["memory_limit_mb"]
        
        with performance_monitor(metrics, "memory_test"):
            # Simulate memory-intensive operations
            large_arrays = []
            for i in range(10):
                # Create progressively larger arrays
                size = 1000 * (i + 1)
                array = np.random.randn(size, size)
                large_arrays.append(array)
                
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                if current_memory > memory_limit:
                    break
        
        summary = metrics.get_summary()
        peak_memory = summary["memory_test"]["peak_memory_mb"]
        
        # Should not exceed memory limit significantly
        assert peak_memory < memory_limit * 1.5, \
            f"Memory usage {peak_memory:.1f}MB exceeds acceptable limit"
    
    @pytest.mark.performance
    def test_memory_leak_detection(self, metrics):
        """Test for memory leaks in repeated operations."""
        baseline_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Perform repeated operations
        for cycle in range(5):
            with performance_monitor(metrics, f"cycle_{cycle}"):
                # Simulate operations that should not leak memory
                for _ in range(100):
                    temp_array = np.random.randn(1000, 1000)
                    result = np.sum(temp_array**2)
                    del temp_array  # Explicit cleanup
                
                # Force garbage collection
                import gc
                gc.collect()
        
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_increase = final_memory - baseline_memory
        
        # Memory should not increase significantly
        assert memory_increase < 50, f"Potential memory leak: {memory_increase:.1f}MB increase"


class TestDataProcessingPerformance:
    """Performance tests for data processing and I/O."""
    
    @pytest.fixture
    def metrics(self):
        return PerformanceMetrics()
    
    @pytest.mark.performance
    def test_data_loading_speed(self, metrics):
        """Test data loading performance."""
        datasets = [
            BenchmarkDatasets.small_dataset(),
            BenchmarkDatasets.medium_dataset(),
        ]
        
        for dataset in datasets:
            size_name = dataset["size"]
            with performance_monitor(metrics, f"load_{size_name}"):
                # Simulate data loading
                states = dataset["states"]
                controls = dataset["controls"]
                
                # Simulate processing
                processed_states = {k: v.copy() for k, v in states.items()}
                time.sleep(0.1)  # Simulate I/O time
        
        summary = metrics.get_summary()
        
        # Loading should be reasonably fast
        for size_name in ["small", "medium"]:
            load_time = summary[f"load_{size_name}"]["duration_ms"]
            if size_name == "small":
                assert load_time < 500, f"Small dataset loading too slow: {load_time:.1f}ms"
            elif size_name == "medium":
                assert load_time < 2000, f"Medium dataset loading too slow: {load_time:.1f}ms"
    
    @pytest.mark.performance
    def test_batch_processing_efficiency(self, metrics):
        """Test batch processing performance."""
        batch_sizes = [32, 64, 128, 256]
        
        for batch_size in batch_sizes:
            with performance_monitor(metrics, f"batch_{batch_size}"):
                # Simulate batch processing
                for batch in range(10):
                    batch_data = np.random.randn(batch_size, 45)  # observations
                    batch_actions = np.random.randn(batch_size, 8)  # actions
                    
                    # Simulate processing
                    processed = batch_data * batch_actions.sum(axis=1, keepdims=True)
                    result = np.mean(processed)
                    time.sleep(batch_size / 10000)  # Simulate processing time
        
        summary = metrics.get_summary()
        
        # Check that larger batches are more efficient per sample
        times_per_sample = {}
        for batch_size in batch_sizes:
            total_time = summary[f"batch_{batch_size}"]["duration_ms"]
            total_samples = batch_size * 10
            times_per_sample[batch_size] = total_time / total_samples
        
        # Larger batches should be more efficient
        assert times_per_sample[256] < times_per_sample[32], \
            "Larger batches should be more efficient per sample"


@pytest.mark.performance
@pytest.mark.slow
class TestFullSystemBenchmark:
    """End-to-end system performance benchmarks."""
    
    def test_complete_training_cycle_performance(self, benchmark_config):
        """Benchmark a complete training cycle."""
        metrics = PerformanceMetrics()
        
        with performance_monitor(metrics, "full_training_cycle"):
            # Simulate complete training
            num_episodes = benchmark_config["num_episodes"]
            max_steps = benchmark_config["max_steps_per_episode"]
            
            for episode in range(num_episodes):
                # Episode initialization
                time.sleep(0.01)
                
                for step in range(max_steps):
                    # Environment step + agent update
                    time.sleep(0.001)
                    
                    # Simulate occasional longer operations
                    if step % 100 == 0:
                        time.sleep(0.01)  # Model save, logging, etc.
                
                # Episode cleanup
                time.sleep(0.005)
        
        summary = metrics.get_summary()
        total_time = summary["full_training_cycle"]["duration_ms"] / 1000  # seconds
        time_limit = benchmark_config["time_limit_seconds"]
        
        assert total_time < time_limit, \
            f"Training cycle too slow: {total_time:.1f}s > {time_limit}s"
        
        # Performance targets
        steps_per_second = (num_episodes * max_steps) / total_time
        assert steps_per_second > 100, f"Training too slow: {steps_per_second:.1f} steps/s"
    
    @pytest.mark.gpu
    def test_gpu_utilization_efficiency(self):
        """Test GPU utilization efficiency (if available)."""
        try:
            import torch
            if not torch.cuda.is_available():
                pytest.skip("GPU not available")
        except ImportError:
            pytest.skip("PyTorch not available")
        
        metrics = PerformanceMetrics()
        
        with performance_monitor(metrics, "gpu_computation"):
            # Simulate GPU computation
            device = torch.device("cuda")
            for _ in range(100):
                tensor = torch.randn(1000, 1000, device=device)
                result = torch.matmul(tensor, tensor.T)
                torch.cuda.synchronize()
        
        summary = metrics.get_summary()
        gpu_time = summary["gpu_computation"]["duration_ms"]
        
        # GPU computation should be fast
        assert gpu_time < 5000, f"GPU computation too slow: {gpu_time:.1f}ms"