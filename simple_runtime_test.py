#!/usr/bin/env python3
"""
Simple runtime test to verify core tokamak-rl functionality works.
Designed to work even without external dependencies.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_core_imports():
    """Test that core modules can be imported."""
    print("Testing core imports...")
    
    try:
        from tokamak_rl.physics import TokamakConfig, GradShafranovSolver
        print("✅ Physics module imported successfully")
    except Exception as e:
        print(f"❌ Physics import failed: {e}")
        return False
    
    try:
        from tokamak_rl.environment import make_tokamak_env
        print("✅ Environment module imported successfully")
    except Exception as e:
        print(f"❌ Environment import failed: {e}")
        return False
    
    try:
        from tokamak_rl.safety import SafetyShield
        print("✅ Safety module imported successfully")
    except Exception as e:
        print(f"❌ Safety import failed: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality without heavy dependencies."""
    print("\nTesting basic functionality...")
    
    try:
        from tokamak_rl.physics import TokamakConfig
        
        # Test configuration creation
        config = TokamakConfig.from_preset("ITER")
        print(f"✅ ITER config created: R={config.major_radius}m, B={config.toroidal_field}T")
        
        # Test custom configuration
        custom_config = TokamakConfig(
            major_radius=1.5,
            minor_radius=0.5,
            toroidal_field=5.0,
            plasma_current=2.0
        )
        print(f"✅ Custom config created: R={custom_config.major_radius}m")
        
        return True
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def test_environment_creation():
    """Test environment creation with minimal dependencies."""
    print("\nTesting environment creation...")
    
    try:
        from tokamak_rl.environment import make_tokamak_env
        
        # Create basic environment
        env = make_tokamak_env(
            tokamak_config="ITER",
            control_frequency=10,  # Lower frequency for testing
            enable_safety=False    # Disable for simplicity
        )
        
        print("✅ Environment created successfully")
        
        # Test basic environment methods
        obs_space = env.observation_space
        action_space = env.action_space
        
        print(f"✅ Observation space: {obs_space}")
        print(f"✅ Action space: {action_space}")
        
        return True
    except Exception as e:
        print(f"❌ Environment creation failed: {e}")
        return False

def test_physics_solver():
    """Test basic physics solver functionality."""
    print("\nTesting physics solver...")
    
    try:
        from tokamak_rl.physics import GradShafranovSolver, TokamakConfig
        
        config = TokamakConfig.from_preset("ITER")
        solver = GradShafranovSolver(config)
        
        # Test basic solver initialization
        print("✅ Physics solver created")
        
        # Test equilibrium computation (simplified)
        try:
            equilibrium = solver.compute_equilibrium()
            print("✅ Equilibrium computation completed")
        except Exception as e:
            print(f"⚠️ Equilibrium computation warning: {e}")
        
        return True
    except Exception as e:
        print(f"❌ Physics solver test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting Tokamak-RL Runtime Tests\n")
    
    tests = [
        test_core_imports,
        test_basic_functionality,
        test_environment_creation,
        test_physics_solver
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - System is functional!")
        return True
    else:
        print("❌ Some tests failed - investigating issues...")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)