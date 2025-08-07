#!/usr/bin/env python3
"""
Simple verification that core code structure is correct without dependencies.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_module_structure():
    """Test that all modules can be imported and have expected structure."""
    print("🔍 Testing module structure...")
    
    # Test physics module
    try:
        from tokamak_rl import physics
        print("✅ Physics module imports successfully")
        
        # Check key classes exist
        assert hasattr(physics, 'TokamakConfig')
        assert hasattr(physics, 'PlasmaState') 
        assert hasattr(physics, 'GradShafranovSolver')
        print("✅ Physics classes found")
        
        # Test preset configs
        presets = ['ITER', 'SPARC', 'NSTX', 'DIII-D']
        for preset in presets:
            try:
                config = physics.TokamakConfig.from_preset(preset)
                assert config.major_radius > 0
                assert config.plasma_current > 0
                print(f"✅ {preset} config: R={config.major_radius}m, Ip={config.plasma_current}MA")
            except Exception as e:
                print(f"❌ {preset} config failed: {e}")
        
    except ImportError as e:
        print(f"❌ Physics module import failed: {e}")
        return False
    
    # Test environment module
    try:
        from tokamak_rl import environment
        print("✅ Environment module imports successfully")
        
        assert hasattr(environment, 'TokamakEnv')
        assert hasattr(environment, 'make_tokamak_env')
        print("✅ Environment classes found")
        
    except ImportError as e:
        print(f"❌ Environment module import failed: {e}")
        return False
    
    # Test agents module
    try:
        from tokamak_rl import agents
        print("✅ Agents module imports successfully")
        
        assert hasattr(agents, 'BaseAgent')
        assert hasattr(agents, 'SACAgent')
        assert hasattr(agents, 'DreamerAgent')
        print("✅ Agent classes found")
        
    except ImportError as e:
        print(f"❌ Agents module import failed: {e}")
        return False
    
    # Test safety module
    try:
        from tokamak_rl import safety
        print("✅ Safety module imports successfully")
        
        assert hasattr(safety, 'SafetyShield')
        assert hasattr(safety, 'DisruptionPredictor')
        assert hasattr(safety, 'SafetyLimits')
        print("✅ Safety classes found")
        
    except ImportError as e:
        print(f"❌ Safety module import failed: {e}")
        return False
    
    return True

def test_class_structure():
    """Test that classes have expected methods and attributes."""
    print("\n🏗️ Testing class structure...")
    
    try:
        from tokamak_rl.physics import TokamakConfig, PlasmaState
        
        # Test TokamakConfig
        config = TokamakConfig(
            major_radius=2.0,
            minor_radius=0.5,
            toroidal_field=3.0,
            plasma_current=1.0
        )
        
        assert config.major_radius == 2.0
        assert config.minor_radius == 0.5
        print("✅ TokamakConfig constructor works")
        
        # Test PlasmaState basic functionality
        state = PlasmaState(config)
        assert hasattr(state, 'reset')
        assert hasattr(state, 'get_observation')
        assert hasattr(state, 'compute_safety_metrics')
        print("✅ PlasmaState has required methods")
        
    except Exception as e:
        print(f"❌ Class structure test failed: {e}")
        return False
    
    return True

def test_package_metadata():
    """Test package metadata and configuration."""
    print("\n📦 Testing package metadata...")
    
    try:
        from tokamak_rl import __version__, __author__
        print(f"✅ Package version: {__version__}")
        print(f"✅ Package author: {__author__}")
        
        # Check pyproject.toml exists
        if os.path.exists('pyproject.toml'):
            print("✅ pyproject.toml found")
        else:
            print("❌ pyproject.toml missing")
            return False
            
    except ImportError as e:
        print(f"❌ Package metadata test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("🧪 Simple Verification Test for Tokamak RL Control Suite")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 3
    
    if test_module_structure():
        tests_passed += 1
        
    if test_class_structure():
        tests_passed += 1
        
    if test_package_metadata():
        tests_passed += 1
    
    print("=" * 60)
    print(f"📊 Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🎉 ALL VERIFICATION TESTS PASSED!")
        print("🏗️ Code structure is correct and ready for enhancement")
    else:
        print("⚠️ Some verification tests failed")
        print("🔧 Code structure needs fixes before proceeding")
        sys.exit(1)