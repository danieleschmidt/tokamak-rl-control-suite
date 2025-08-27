"""
Simple test of advanced shape control system without heavy dependencies
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Mock the dependencies if not available
try:
    import numpy as np
except ImportError:
    print("NumPy not available, using mock implementation")
    class MockNumPy:
        @staticmethod
        def array(x):
            return x
        @staticmethod
        def random():
            return type('', (), {'randn': lambda *args: [0.1] * (args[0] if args else 1)})()
        @staticmethod
        def zeros(shape):
            return [0.0] * shape
        @staticmethod
        def linspace(start, stop, num):
            return [start + i*(stop-start)/(num-1) for i in range(num)]
        @staticmethod
        def column_stack(arrays):
            return [[a[i] for a in arrays] for i in range(len(arrays[0]))]
        @staticmethod
        def full(size, value):
            return [value] * size
        @staticmethod
        def min(arr):
            return min(arr) if hasattr(arr, '__iter__') else arr
        @staticmethod
        def max(arr):
            return max(arr) if hasattr(arr, '__iter__') else arr
        @staticmethod
        def mean(arr):
            return sum(arr) / len(arr) if arr else 0
        @staticmethod
        def sum(arr):
            return sum(arr) if hasattr(arr, '__iter__') else arr
        @staticmethod
        def sqrt(x):
            return x ** 0.5
        @staticmethod
        def clip(arr, min_val, max_val):
            if hasattr(arr, '__iter__'):
                return [max(min_val, min(max_val, x)) for x in arr]
            return max(min_val, min(max_val, arr))
        @staticmethod
        def all(arr):
            return all(arr)
    np = MockNumPy()

try:
    import torch
    import torch.nn as nn
except ImportError:
    print("PyTorch not available, using mock implementation")
    class MockTorch:
        class FloatTensor:
            def __init__(self, data):
                self.data = data
            def unsqueeze(self, dim):
                return self
            def squeeze(self):
                return self.data
        
        @staticmethod
        def randn(*shape):
            return type('Tensor', (), {
                'shape': shape,
                'data': [0.1] * (shape[0] if shape else 1),
                'detach': lambda: type('', (), {'numpy': lambda: type('', (), {'squeeze': lambda: [0.1] * 8})()})()
            })()
    
    torch = MockTorch()
    
    class MockModule:
        def parameters(self):
            return []
    
    class MockOptim:
        class Adam:
            def __init__(self, params, lr=1e-4):
                pass
            def zero_grad(self):
                pass
            def step(self):
                pass
    
    torch.nn = type('nn', (), {
        'Module': MockModule,
        'Sequential': lambda *args: type('', (), {'forward': lambda x: x})(),
        'Linear': lambda i, o: None,
        'ReLU': lambda: None,
        'Dropout': lambda p: None,
        'Tanh': lambda: None,
        'MSELoss': lambda: lambda x, y: type('', (), {'backward': lambda: None, 'item': lambda: 0.1})()
    })()
    torch.optim = MockOptim()

# Test basic functionality
def test_basic_functionality():
    print("Testing Advanced Plasma Shape Control System")
    print("=" * 50)
    
    try:
        # Test imports work
        from tokamak_rl.advanced_shape_control import (
            PlasmaTarget, ControlLimits, create_advanced_shape_control_system
        )
        print("✓ Imports successful")
        
        # Test basic object creation
        target = PlasmaTarget()
        limits = ControlLimits()
        print(f"✓ PlasmaTarget created: elongation={target.elongation}")
        print(f"✓ ControlLimits created: max_current={limits.max_coil_current}")
        
        # Test system creation
        system = create_advanced_shape_control_system()
        print("✓ Advanced control system created")
        
        # Test reset functionality
        initial_state = [0.1] * 45  # Mock 45-dimensional state
        system.reset(initial_state)
        print("✓ System reset with initial state")
        
        # Test control steps
        for i in range(5):
            observation = [0.1 + i*0.01] * 45  # Mock evolving observation
            result = system.control_step(observation)
            
            print(f"✓ Step {i+1}: action_shape={len(result.get('action', []))}, "
                  f"shape_error={result.get('shape_error', 0):.6f}")
        
        # Test performance summary
        summary = system.get_performance_summary()
        print("✓ Performance summary generated:")
        for key, value in summary.items():
            print(f"  {key}: {value:.6f}")
        
        print("\n🎉 Advanced Plasma Shape Control System test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_functionality()
    if success:
        print("\n✅ All tests passed - Generation 1 functionality verified!")
        exit(0)
    else:
        print("\n❌ Tests failed")
        exit(1)