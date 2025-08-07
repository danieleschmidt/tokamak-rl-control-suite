#!/usr/bin/env python3
"""
Basic functionality test for tokamak-rl-control-suite.
Tests core components without requiring full installation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from tokamak_rl.physics import TokamakConfig, PlasmaState, GradShafranovSolver
from tokamak_rl.safety import SafetyLimits, SafetyShield, DisruptionPredictor


def test_tokamak_config():
    """Test tokamak configuration creation."""
    print("Testing TokamakConfig...")
    
    # Test preset creation
    iter_config = TokamakConfig.from_preset("ITER")
    assert iter_config.major_radius == 6.2
    assert iter_config.plasma_current == 15.0
    
    sparc_config = TokamakConfig.from_preset("SPARC") 
    assert sparc_config.major_radius == 1.85
    assert sparc_config.plasma_current == 8.7
    
    print("✅ TokamakConfig tests passed")


def test_plasma_state():
    """Test plasma state functionality."""
    print("Testing PlasmaState...")
    
    config = TokamakConfig.from_preset("ITER")
    state = PlasmaState(config)
    
    # Test initial state
    assert state.plasma_current == config.plasma_current
    assert state.elongation == config.elongation
    assert len(state.q_profile) == 101
    assert len(state.density_profile) == 101
    
    # Test observation vector
    obs = state.get_observation()
    assert len(obs) == 45  # Specified observation dimension
    assert obs.dtype == np.float32
    
    # Test safety metrics
    metrics = state.compute_safety_metrics()
    assert 'q_min' in metrics
    assert 'beta_limit_fraction' in metrics
    assert 'disruption_probability' in metrics
    
    print("✅ PlasmaState tests passed")


def test_physics_solver():
    """Test Grad-Shafranov solver."""
    print("Testing GradShafranovSolver...")
    
    config = TokamakConfig.from_preset("SPARC")
    solver = GradShafranovSolver(config)
    state = PlasmaState(config)
    
    # Test grid setup
    assert solver.nr == 65
    assert solver.nz == 65
    assert solver.R.shape == (65, 65)
    
    # Test equilibrium solving
    pf_currents = np.array([1.0, -0.5, 0.8, -0.3, 0.6, -0.4])
    new_state = solver.solve_equilibrium(state, pf_currents)
    
    assert np.array_equal(new_state.pf_coil_currents, pf_currents)
    assert new_state.shape_error >= 0
    assert 0 <= new_state.disruption_probability <= 1
    
    print("✅ GradShafranovSolver tests passed")


def test_safety_systems():
    """Test safety shield and disruption predictor."""
    print("Testing Safety Systems...")
    
    # Test safety limits
    limits = SafetyLimits()
    assert limits.q_min_threshold == 1.5
    assert limits.beta_limit == 0.04
    
    # Test disruption predictor
    predictor = DisruptionPredictor()
    config = TokamakConfig.from_preset("ITER")
    state = PlasmaState(config)
    
    # Test prediction (should be low initially)
    risk = predictor.predict_disruption(state)
    assert 0 <= risk <= 1
    
    # Test safety shield
    shield = SafetyShield(limits, predictor)
    
    # Test normal action
    normal_action = np.array([0.1, -0.1, 0.2, -0.2, 0.1, -0.1, 0.3, 0.5])
    safe_action, info = shield.filter_action(normal_action, state)
    
    assert len(safe_action) == len(normal_action)
    assert 'action_modified' in info
    assert 'violations' in info
    assert 'disruption_risk' in info
    
    # Test extreme action that should be modified
    extreme_action = np.array([15.0, -15.0, 20.0, -20.0, 10.0, -10.0, 2.0, 2.0])
    safe_action2, info2 = shield.filter_action(extreme_action, state)
    
    assert info2['action_modified'] == True
    assert len(info2['violations']) > 0
    
    print("✅ Safety Systems tests passed")


def test_integration():
    """Test integrated system functionality."""
    print("Testing System Integration...")
    
    # Create complete system
    config = TokamakConfig.from_preset("DIII-D")
    solver = GradShafranovSolver(config)
    state = PlasmaState(config)
    shield = SafetyShield()
    
    # Simulate control loop
    for step in range(5):
        # Generate random control action
        action = np.random.uniform(-1, 1, 8)
        
        # Apply safety filtering
        safe_action, safety_info = shield.filter_action(action, state)
        
        # Execute physics step
        pf_currents = safe_action[:6] * 2.0  # Scale to MA
        new_state = solver.solve_equilibrium(state, pf_currents)
        
        # Verify state is physically reasonable
        assert new_state.q_min > 0.5
        assert 0 <= new_state.plasma_beta <= 0.15
        assert new_state.shape_error >= 0
        
        state = new_state
        
    print("✅ System Integration tests passed")


if __name__ == "__main__":
    print("🧪 Running Basic Functionality Tests for Tokamak RL Control Suite")
    print("=" * 70)
    
    try:
        test_tokamak_config()
        test_plasma_state()
        test_physics_solver()
        test_safety_systems()
        test_integration()
        
        print("=" * 70)
        print("🎉 ALL TESTS PASSED! Basic functionality is working correctly.")
        print("🚀 Ready to proceed with enhanced features.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)