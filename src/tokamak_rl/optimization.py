"""
Performance optimization and caching system for tokamak RL control.

This module provides advanced caching, memoization, parallel processing,
and performance optimization for scalable tokamak control systems.
"""

import time
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import wraps, lru_cache
from typing import Dict, Any, List, Optional, Callable, Tuple, Union
import hashlib
import pickle
import gc
import weakref
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, OrderedDict
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
        
        @staticmethod
        def concatenate(arrays):
            result = []
            for arr in arrays:
                if hasattr(arr, '__iter__'):
                    result.extend(arr)
                else:
                    result.append(arr)
            return result

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Cache eviction strategies."""
    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"
    TTL = "ttl"
    ADAPTIVE = "adaptive"


class OptimizationLevel(Enum):
    """Performance optimization levels."""
    BASIC = "basic"
    STANDARD = "standard"
    AGGRESSIVE = "aggressive"
    MAXIMUM = "maximum"


@dataclass
class CacheStats:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    memory_usage: int = 0
    avg_access_time: float = 0.0
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    @property
    def miss_rate(self) -> float:
        return 1.0 - self.hit_rate


class AdaptiveCache:
    """Adaptive high-performance cache with multiple strategies."""
    
    def __init__(self, max_size: int = 1000, strategy: CacheStrategy = CacheStrategy.ADAPTIVE,
                 ttl_seconds: Optional[float] = None, memory_limit_mb: Optional[int] = None):
        self.max_size = max_size
        self.strategy = strategy
        self.ttl_seconds = ttl_seconds
        self.memory_limit = memory_limit_mb * 1024 * 1024 if memory_limit_mb else None
        
        self._cache = OrderedDict()
        self._access_times = {}
        self._access_counts = defaultdict(int)
        self._creation_times = {}
        self._stats = CacheStats()
        self._lock = threading.RLock()
        
        # Adaptive strategy parameters
        self._recent_hit_rates = []
        self._strategy_performance = defaultdict(list)
        self._current_adaptive_strategy = CacheStrategy.LRU
    
    def get(self, key: str, default=None):
        """Get value from cache."""
        with self._lock:
            start_time = time.time()
            
            if key in self._cache:
                # Cache hit
                value = self._cache[key]
                self._update_access_stats(key)
                self._stats.hits += 1
                
                # Move to end for LRU
                if self.strategy in [CacheStrategy.LRU, CacheStrategy.ADAPTIVE]:
                    self._cache.move_to_end(key)
                
                access_time = time.time() - start_time
                self._update_avg_access_time(access_time)
                
                return value
            else:
                # Cache miss
                self._stats.misses += 1
                access_time = time.time() - start_time
                self._update_avg_access_time(access_time)
                return default
    
    def put(self, key: str, value: Any, ttl_override: Optional[float] = None) -> None:
        """Put value in cache."""
        with self._lock:
            current_time = time.time()
            
            # Check if we need to evict
            if len(self._cache) >= self.max_size and key not in self._cache:
                self._evict()
            
            # Store value
            self._cache[key] = value
            self._access_times[key] = current_time
            self._access_counts[key] = 1
            self._creation_times[key] = current_time
            
            # Check memory limit
            if self.memory_limit:
                self._check_memory_limit()
            
            # Update stats
            self._update_memory_usage()
    
    def invalidate(self, key: str) -> bool:
        """Invalidate specific cache entry."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._access_times.pop(key, None)
                self._access_counts.pop(key, None)
                self._creation_times.pop(key, None)
                self._update_memory_usage()
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
            self._access_counts.clear()
            self._creation_times.clear()
            self._stats.memory_usage = 0
    
    def get_stats(self) -> CacheStats:
        """Get cache performance statistics."""
        with self._lock:
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                memory_usage=self._stats.memory_usage,
                avg_access_time=self._stats.avg_access_time
            )
    
    def _evict(self) -> None:
        """Evict entries based on strategy."""
        if not self._cache:
            return
        
        strategy = self._get_effective_strategy()
        
        if strategy == CacheStrategy.LRU:
            # Remove least recently used (first item)
            key = next(iter(self._cache))
        elif strategy == CacheStrategy.LFU:
            # Remove least frequently used
            key = min(self._access_counts.keys(), key=lambda k: self._access_counts[k])
        elif strategy == CacheStrategy.FIFO:
            # Remove oldest entry
            key = min(self._creation_times.keys(), key=lambda k: self._creation_times[k])
        elif strategy == CacheStrategy.TTL:
            # Remove expired entries first
            current_time = time.time()
            expired_keys = [
                k for k, t in self._creation_times.items()
                if self.ttl_seconds and (current_time - t) > self.ttl_seconds
            ]
            if expired_keys:
                key = expired_keys[0]
            else:
                key = next(iter(self._cache))  # Fallback to FIFO
        else:
            # Default to LRU
            key = next(iter(self._cache))
        
        # Remove the selected key
        self._remove_key(key)
        self._stats.evictions += 1
    
    def _remove_key(self, key: str) -> None:
        """Remove key and associated metadata."""
        self._cache.pop(key, None)
        self._access_times.pop(key, None)
        self._access_counts.pop(key, None)
        self._creation_times.pop(key, None)
    
    def _update_access_stats(self, key: str) -> None:
        """Update access statistics for key."""
        current_time = time.time()
        self._access_times[key] = current_time
        self._access_counts[key] += 1
    
    def _update_avg_access_time(self, access_time: float) -> None:
        """Update average access time with exponential smoothing."""
        alpha = 0.1  # Smoothing factor
        if self._stats.avg_access_time == 0:
            self._stats.avg_access_time = access_time
        else:
            self._stats.avg_access_time = (alpha * access_time + 
                                         (1 - alpha) * self._stats.avg_access_time)
    
    def _update_memory_usage(self) -> None:
        """Update memory usage estimate."""
        try:
            # Rough estimate of memory usage
            total_size = 0
            for key, value in self._cache.items():
                total_size += len(key.encode('utf-8'))
                try:
                    total_size += len(pickle.dumps(value))
                except (pickle.PicklingError, TypeError):
                    total_size += 1000  # Rough estimate for unpicklable objects
            
            self._stats.memory_usage = total_size
        except Exception:
            # Fallback if memory estimation fails
            self._stats.memory_usage = len(self._cache) * 1000
    
    def _check_memory_limit(self) -> None:
        """Check and enforce memory limit."""
        if not self.memory_limit:
            return
        
        self._update_memory_usage()
        
        while (self._stats.memory_usage > self.memory_limit and 
               len(self._cache) > 1):
            self._evict()
            self._update_memory_usage()
    
    def _get_effective_strategy(self) -> CacheStrategy:
        """Get effective strategy for adaptive caching."""
        if self.strategy != CacheStrategy.ADAPTIVE:
            return self.strategy
        
        # Update hit rate tracking
        current_hit_rate = self.get_stats().hit_rate
        self._recent_hit_rates.append(current_hit_rate)
        
        # Keep only recent history
        if len(self._recent_hit_rates) > 100:
            self._recent_hit_rates.pop(0)
        
        # Switch strategy if performance is poor
        if len(self._recent_hit_rates) >= 50:
            recent_avg = np.mean(self._recent_hit_rates[-25:])
            if recent_avg < 0.5 and len(self._recent_hit_rates) > 75:
                # Try different strategy
                strategies = [CacheStrategy.LRU, CacheStrategy.LFU, CacheStrategy.FIFO]
                current_idx = strategies.index(self._current_adaptive_strategy) if self._current_adaptive_strategy in strategies else 0
                self._current_adaptive_strategy = strategies[(current_idx + 1) % len(strategies)]
                logger.info(f"Adaptive cache switching to {self._current_adaptive_strategy.value}")
        
        return self._current_adaptive_strategy


class MemoizationManager:
    """Advanced memoization with cache management."""
    
    def __init__(self, default_cache_size: int = 500, 
                 default_ttl: Optional[float] = None):
        self.caches: Dict[str, AdaptiveCache] = {}
        self.default_cache_size = default_cache_size
        self.default_ttl = default_ttl
        self._lock = threading.RLock()
    
    def memoize(self, cache_size: Optional[int] = None, 
               ttl: Optional[float] = None,
               cache_key_func: Optional[Callable] = None,
               strategy: CacheStrategy = CacheStrategy.LRU):
        """Decorator for function memoization."""
        def decorator(func):
            func_name = f"{func.__module__}.{func.__qualname__}"
            
            with self._lock:
                if func_name not in self.caches:
                    self.caches[func_name] = AdaptiveCache(
                        max_size=cache_size or self.default_cache_size,
                        strategy=strategy,
                        ttl_seconds=ttl or self.default_ttl
                    )
            
            cache = self.caches[func_name]
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                if cache_key_func:
                    key = cache_key_func(*args, **kwargs)
                else:
                    key = self._generate_cache_key(func_name, args, kwargs)
                
                # Try to get from cache
                result = cache.get(key)
                if result is not None:
                    return result
                
                # Compute result and cache it
                result = func(*args, **kwargs)
                cache.put(key, result)
                
                return result
            
            # Add cache management methods to wrapper
            wrapper.cache_stats = lambda: cache.get_stats()
            wrapper.cache_clear = lambda: cache.clear()
            wrapper.cache_invalidate = lambda key: cache.invalidate(key)
            
            return wrapper
        return decorator
    
    def _generate_cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key for function call."""
        try:
            # Create a hashable representation
            key_data = (func_name, args, tuple(sorted(kwargs.items())))
            key_str = pickle.dumps(key_data)
            return hashlib.sha256(key_str).hexdigest()
        except (pickle.PicklingError, TypeError):
            # Fallback for unpicklable arguments
            key_parts = [func_name, str(args), str(sorted(kwargs.items()))]
            key_str = '|'.join(key_parts)
            return hashlib.sha256(key_str.encode()).hexdigest()
    
    def get_all_stats(self) -> Dict[str, CacheStats]:
        """Get statistics for all caches."""
        with self._lock:
            return {name: cache.get_stats() for name, cache in self.caches.items()}
    
    def clear_all_caches(self) -> None:
        """Clear all memoization caches."""
        with self._lock:
            for cache in self.caches.values():
                cache.clear()


class ParallelProcessingManager:
    """Manager for parallel and concurrent processing."""
    
    def __init__(self, max_workers: Optional[int] = None, 
                 use_processes: bool = False):
        self.max_workers = max_workers or min(32, (mp.cpu_count() or 1) + 4)
        self.use_processes = use_processes
        self._executor = None
        self._lock = threading.RLock()
        
        # Performance tracking
        self.task_stats = defaultdict(list)
        self.active_tasks = set()
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
    
    def start(self) -> None:
        """Start the executor."""
        with self._lock:
            if self._executor is None:
                if self.use_processes:
                    self._executor = ProcessPoolExecutor(max_workers=self.max_workers)
                else:
                    self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
    
    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the executor."""
        with self._lock:
            if self._executor:
                self._executor.shutdown(wait=wait)
                self._executor = None
    
    def submit_task(self, func: Callable, *args, task_name: str = None, **kwargs):
        """Submit a task for parallel execution."""
        if self._executor is None:
            self.start()
        
        task_id = f"{task_name or func.__name__}_{time.time()}"
        start_time = time.time()
        
        future = self._executor.submit(func, *args, **kwargs)
        
        # Track task
        self.active_tasks.add(task_id)
        
        def track_completion(fut):
            self.active_tasks.discard(task_id)
            duration = time.time() - start_time
            self.task_stats[task_name or func.__name__].append(duration)
            
            # Keep only recent stats
            if len(self.task_stats[task_name or func.__name__]) > 100:
                self.task_stats[task_name or func.__name__].pop(0)
        
        future.add_done_callback(track_completion)
        
        return future
    
    def map_parallel(self, func: Callable, iterable, 
                    chunk_size: Optional[int] = None, 
                    task_name: str = None) -> List[Any]:
        """Map function over iterable in parallel."""
        if self._executor is None:
            self.start()
        
        start_time = time.time()
        
        # Submit all tasks
        futures = []
        for i, item in enumerate(iterable):
            future = self.submit_task(func, item, task_name=f"{task_name or 'map'}_{i}")
            futures.append(future)
        
        # Collect results
        results = []
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Parallel task failed: {e}")
                results.append(None)
        
        duration = time.time() - start_time
        logger.info(f"Parallel map of {len(futures)} tasks completed in {duration:.2f}s")
        
        return results
    
    def get_performance_stats(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics for parallel tasks."""
        stats = {}
        for task_name, durations in self.task_stats.items():
            if durations:
                stats[task_name] = {
                    'count': len(durations),
                    'avg_duration': np.mean(durations),
                    'min_duration': min(durations),
                    'max_duration': max(durations),
                    'total_duration': sum(durations)
                }
        return stats


class ResourcePool:
    """Generic resource pool for efficient resource management."""
    
    def __init__(self, factory: Callable[[], Any], 
                 max_size: int = 10, min_size: int = 2,
                 max_idle_time: float = 300.0):
        self.factory = factory
        self.max_size = max_size
        self.min_size = min_size
        self.max_idle_time = max_idle_time
        
        self._pool = []
        self._in_use = set()
        self._creation_times = {}
        self._last_used = {}
        self._lock = threading.RLock()
        
        # Pre-populate with minimum resources
        self._ensure_min_size()
        
        # Start cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()
    
    def acquire(self, timeout: Optional[float] = None) -> Any:
        """Acquire a resource from the pool."""
        start_time = time.time()
        
        while True:
            with self._lock:
                # Try to get an existing resource
                if self._pool:
                    resource = self._pool.pop()
                    self._in_use.add(id(resource))
                    self._last_used[id(resource)] = time.time()
                    return resource
                
                # Create new resource if under limit
                if len(self._in_use) < self.max_size:
                    resource = self._create_resource()
                    self._in_use.add(id(resource))
                    self._last_used[id(resource)] = time.time()
                    return resource
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError("Timeout waiting for resource")
            
            # Wait briefly before retrying
            time.sleep(0.01)
    
    def release(self, resource: Any) -> None:
        """Release a resource back to the pool."""
        with self._lock:
            resource_id = id(resource)
            
            if resource_id in self._in_use:
                self._in_use.remove(resource_id)
                self._pool.append(resource)
                self._last_used[resource_id] = time.time()
    
    def get_stats(self) -> Dict[str, int]:
        """Get pool statistics."""
        with self._lock:
            return {
                'pool_size': len(self._pool),
                'in_use': len(self._in_use),
                'total_created': len(self._creation_times),
                'max_size': self.max_size
            }
    
    def _create_resource(self) -> Any:
        """Create a new resource."""
        resource = self.factory()
        self._creation_times[id(resource)] = time.time()
        return resource
    
    def _ensure_min_size(self) -> None:
        """Ensure pool has minimum number of resources."""
        with self._lock:
            current_total = len(self._pool) + len(self._in_use)
            needed = self.min_size - current_total
            
            for _ in range(needed):
                if len(self._pool) + len(self._in_use) < self.max_size:
                    resource = self._create_resource()
                    self._pool.append(resource)
    
    def _cleanup_loop(self) -> None:
        """Background cleanup of idle resources."""
        while True:
            try:
                time.sleep(60)  # Check every minute
                self._cleanup_idle_resources()
            except Exception as e:
                logger.error(f"Resource pool cleanup error: {e}")
    
    def _cleanup_idle_resources(self) -> None:
        """Remove idle resources beyond minimum."""
        with self._lock:
            current_time = time.time()
            
            # Remove old unused resources
            resources_to_remove = []
            for i, resource in enumerate(self._pool):
                resource_id = id(resource)
                last_used = self._last_used.get(resource_id, 0)
                
                if (current_time - last_used) > self.max_idle_time:
                    if len(self._pool) - len(resources_to_remove) > self.min_size:
                        resources_to_remove.append(i)
            
            # Remove from back to front to maintain indices
            for i in reversed(resources_to_remove):
                resource = self._pool.pop(i)
                resource_id = id(resource)
                self._creation_times.pop(resource_id, None)
                self._last_used.pop(resource_id, None)


class PerformanceOptimizer:
    """Central performance optimization coordinator."""
    
    def __init__(self, optimization_level: OptimizationLevel = OptimizationLevel.STANDARD):
        self.optimization_level = optimization_level
        self.memoization_manager = MemoizationManager()
        self.parallel_manager = ParallelProcessingManager()
        self.resource_pools: Dict[str, ResourcePool] = {}
        
        # Performance monitoring
        self.performance_metrics = defaultdict(list)
        self._lock = threading.RLock()
        
        # Apply optimizations based on level
        self._apply_optimizations()
    
    def _apply_optimizations(self) -> None:
        """Apply optimizations based on optimization level."""
        if self.optimization_level == OptimizationLevel.BASIC:
            # Basic caching only
            self.memoization_manager.default_cache_size = 100
        
        elif self.optimization_level == OptimizationLevel.STANDARD:
            # Standard caching and some parallelization
            self.memoization_manager.default_cache_size = 500
            self.parallel_manager.max_workers = min(8, mp.cpu_count())
        
        elif self.optimization_level == OptimizationLevel.AGGRESSIVE:
            # Aggressive caching and parallelization
            self.memoization_manager.default_cache_size = 1000
            self.parallel_manager.max_workers = mp.cpu_count() * 2
            
            # Enable adaptive caching
            for cache in self.memoization_manager.caches.values():
                cache.strategy = CacheStrategy.ADAPTIVE
        
        elif self.optimization_level == OptimizationLevel.MAXIMUM:
            # Maximum performance optimizations
            self.memoization_manager.default_cache_size = 2000
            self.parallel_manager.max_workers = mp.cpu_count() * 2
            
            # Use process-based parallelism for CPU-intensive tasks
            self.parallel_manager.use_processes = True
            
            # Aggressive garbage collection
            gc.set_threshold(700, 10, 10)
    
    def create_resource_pool(self, name: str, factory: Callable[[], Any],
                           max_size: int = 10, min_size: int = 2) -> ResourcePool:
        """Create a named resource pool."""
        with self._lock:
            pool = ResourcePool(factory, max_size, min_size)
            self.resource_pools[name] = pool
            return pool
    
    def get_resource_pool(self, name: str) -> Optional[ResourcePool]:
        """Get existing resource pool by name."""
        return self.resource_pools.get(name)
    
    def optimize_function(self, cache_size: int = None, 
                         use_parallel: bool = False,
                         cache_strategy: CacheStrategy = CacheStrategy.LRU):
        """Decorator to optimize function performance."""
        def decorator(func):
            # Apply memoization
            memoized_func = self.memoization_manager.memoize(
                cache_size=cache_size,
                strategy=cache_strategy
            )(func)
            
            if use_parallel:
                # Add parallel execution capability
                original_func = memoized_func
                
                @wraps(original_func)
                def parallel_wrapper(*args, **kwargs):
                    # For single calls, use original function
                    if not hasattr(args[0], '__iter__') or isinstance(args[0], str):
                        return original_func(*args, **kwargs)
                    
                    # For iterable first argument, use parallel map
                    iterable = args[0]
                    remaining_args = args[1:]
                    
                    def single_call(item):
                        return original_func(item, *remaining_args, **kwargs)
                    
                    return self.parallel_manager.map_parallel(single_call, iterable)
                
                return parallel_wrapper
            
            return memoized_func
        return decorator
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        with self._lock:
            stats = {
                'optimization_level': self.optimization_level.value,
                'memoization_stats': self.memoization_manager.get_all_stats(),
                'parallel_stats': self.parallel_manager.get_performance_stats(),
                'resource_pool_stats': {
                    name: pool.get_stats() 
                    for name, pool in self.resource_pools.items()
                },
                'active_parallel_tasks': len(self.parallel_manager.active_tasks)
            }
            
            return stats


# Global optimizer instance
_global_optimizer = None


def get_global_optimizer() -> PerformanceOptimizer:
    """Get or create global performance optimizer."""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = PerformanceOptimizer()
    return _global_optimizer


# Convenience decorators using global optimizer
def cached(cache_size: int = 500, strategy: CacheStrategy = CacheStrategy.LRU):
    """Convenience decorator for caching using global optimizer."""
    return get_global_optimizer().memoization_manager.memoize(
        cache_size=cache_size, strategy=strategy
    )


def optimized(cache_size: int = None, use_parallel: bool = False,
              cache_strategy: CacheStrategy = CacheStrategy.LRU):
    """Convenience decorator for full optimization using global optimizer."""
    return get_global_optimizer().optimize_function(
        cache_size=cache_size,
        use_parallel=use_parallel,
        cache_strategy=cache_strategy
    )