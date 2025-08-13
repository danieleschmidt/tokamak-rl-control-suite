#!/usr/bin/env python3
"""
Generation 1: MAKE IT WORK - Basic Functionality Test

This script validates core tokamak-rl functionality with basic implementations.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_imports():
    """Test that core modules can be imported."""
    print("🔧 Testing basic imports...")
    
    try:
        import tokamak_rl
        print(f"✓ tokamak_rl imported successfully (version: {tokamak_rl.__version__})")
        
        from tokamak_rl.physics import TokamakConfig, GradShafranovSolver
        print("✓ Physics components imported")
        
        # Test basic physics configuration
        config = TokamakConfig(
            major_radius=1.65,
            minor_radius=0.65,
            toroidal_field=5.3,
            plasma_current=2.0
        )
        print(f"✓ TokamakConfig created - major_radius: {config.major_radius}m")
        
        solver = GradShafranovSolver(config)
        print("✓ GradShafranovSolver created")
        
        return True
        
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_basic_physics():
    """Test basic physics calculations."""
    print("\n🧮 Testing basic physics calculations...")
    
    try:
        from tokamak_rl.physics import TokamakConfig, GradShafranovSolver
        
        config = TokamakConfig(
            major_radius=1.65,  # ITER-like
            minor_radius=0.65,
            toroidal_field=5.3,
            plasma_current=2.0
        )
        
        solver = GradShafranovSolver(config)
        
        # Test basic equilibrium calculation with proper PlasmaState
        from tokamak_rl.physics import PlasmaState
        import numpy as np
        
        # Create initial plasma state
        initial_state = PlasmaState(config)
        pf_currents = np.array([1.0, 1.2, 0.8, 1.1, 0.9, 1.0])  # PF coil currents
        
        new_state = solver.solve_equilibrium(initial_state, pf_currents)
        print(f"✓ Equilibrium solved - current: {new_state.plasma_current:.2f} MA")
        
        # Test safety factor calculation  
        q_profile = new_state.q_profile
        print(f"✓ Safety factor calculated - q_min: {min(q_profile):.2f}, q_edge: {q_profile[-1]:.2f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Physics test failed: {e}")
        return False

def test_basic_environment():
    """Test basic environment functionality with simplified setup."""
    print("\n🌍 Testing basic environment...")
    
    try:
        # Import with dependency handling
        from tokamak_rl.environment import TokamakEnv
        from tokamak_rl.physics import TokamakConfig
        
        # Create minimal configuration
        config = TokamakConfig(
            major_radius=1.65,
            minor_radius=0.65,
            toroidal_field=5.3,
            plasma_current=2.0
        )
        
        # Create environment with minimal dependencies
        env_config = {
            'tokamak_config': config,
            'control_frequency': 10,  # Reduced frequency
            'safety_factor': 1.2,
            'enable_safety': False,   # Disable safety systems for basic test
            'enable_monitoring': False  # Disable monitoring for basic test
        }
        
        # Patch environment creation to bypass safety systems
        from tokamak_rl import environment
        original_create_safety = getattr(environment, 'create_safety_system', None)
        
        def mock_safety_system(config):
            class MockSafety:
                def filter_action(self, action, state): return action
                def check_disruption_risk(self, state): return 0.0
                def emergency_shutdown(self): pass
            return MockSafety()
        
        environment.create_safety_system = mock_safety_system
        
        try:
            env = TokamakEnv(env_config)
            print("✓ TokamakEnv created with mock safety")
            
            # Test reset
            obs, info = env.reset()
            print(f"✓ Environment reset - obs length: {len(obs)}")
            
            # Test action space
            action = [0.1, -0.05, 0.2, -0.1, 0.15, 0.0, 0.5, 0.3]  # Manual action
            if hasattr(env.action_space, 'shape') and len(action) == env.action_space.shape[0]:
                obs, reward, done, truncated, info = env.step(action)
                print(f"✓ Environment step - reward: {reward:.3f}, done: {done}")
                
                return True
            else:
                shape = getattr(env.action_space, 'shape', [None])
                expected = shape[0] if shape and shape[0] is not None else "unknown"
                print(f"✗ Action space mismatch: expected {expected}, got {len(action)}")
                return False
                
        finally:
            # Restore original function if it existed
            if original_create_safety:
                environment.create_safety_system = original_create_safety
        
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        return False

def test_basic_monitoring():
    """Test basic monitoring functionality."""
    print("\n📊 Testing basic monitoring...")
    
    try:
        from tokamak_rl.monitoring import PlasmaMonitor, AlertThresholds
        
        thresholds = AlertThresholds()
        thresholds.q_min = 1.5
        thresholds.shape_error = 5.0
        
        monitor = PlasmaMonitor(
            log_dir="./test_logs",
            alert_thresholds=thresholds
        )
        
        # Test basic logging with PlasmaState
        from tokamak_rl.physics import PlasmaState, TokamakConfig
        
        config = TokamakConfig(
            major_radius=1.65,
            minor_radius=0.65,
            toroidal_field=5.3,
            plasma_current=2.0
        )
        
        test_state = PlasmaState(config)
        test_state.q_profile = [2.1, 1.8, 1.6, 1.9]  # Set test values
        
        test_action = [0.1, -0.05, 0.2, -0.1, 0.15, 0.0, 0.5, 0.3]
        
        monitor.log_step(test_state, test_action, -0.5, {'step': 1})
        print("✓ Monitoring step logged successfully")
        
        print("✓ Alert system initialized")
        
        return True
        
    except Exception as e:
        print(f"✗ Monitoring test failed: {e}")
        return False

def run_generation1_tests():
    """Run all Generation 1 basic functionality tests."""
    print("=" * 60)
    print("🚀 GENERATION 1: MAKE IT WORK - Basic Functionality Tests")
    print("=" * 60)
    
    tests = [
        test_basic_imports,
        test_basic_physics,
        test_basic_environment,
        test_basic_monitoring
    ]
    
    results = []
    for test in tests:
        results.append(test())
        
    print("\n" + "=" * 60)
    print("📊 GENERATION 1 TEST RESULTS")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 ALL GENERATION 1 TESTS PASSED - Core functionality working!")
        return True
    else:
        print("⚠️ Some tests failed - needs fixing before Generation 2")
        return False

if __name__ == "__main__":
    success = run_generation1_tests()
    sys.exit(0 if success else 1)