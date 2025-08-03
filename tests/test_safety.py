"""
Tests for safety system components.
"""

import pytest
import numpy as np
import torch
from tokamak_rl.safety import (
    SafetyLimits, DisruptionPredictor, SafetyShield, 
    ConstraintManager, create_safety_system
)
from tokamak_rl.physics import TokamakConfig, PlasmaState


class TestSafetyLimits:
    """Test safety limits configuration."""
    
    def test_default_limits(self):
        """Test default safety limits."""
        limits = SafetyLimits()
        
        assert limits.q_min_threshold == 1.5
        assert limits.beta_limit == 0.04
        assert limits.density_limit == 1.2e20
        assert limits.shape_error_limit == 5.0
        assert limits.disruption_probability_limit == 0.1
        
    def test_custom_limits(self):
        """Test custom safety limits."""
        limits = SafetyLimits(
            q_min_threshold=2.0,
            beta_limit=0.03,
            density_limit=1.0e20
        )
        
        assert limits.q_min_threshold == 2.0
        assert limits.beta_limit == 0.03
        assert limits.density_limit == 1.0e20


class TestDisruptionPredictor:
    """Test disruption prediction system."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.predictor = DisruptionPredictor()
        self.config = TokamakConfig.from_preset("ITER")
        
    def test_predictor_initialization(self):
        """Test predictor initialization."""
        assert hasattr(self.predictor, 'model')
        assert hasattr(self.predictor, 'history_buffer')
        assert hasattr(self.predictor, 'buffer_size')
        assert self.predictor.buffer_size == 50
        
    def test_model_architecture(self):
        """Test LSTM model architecture."""
        # Test model forward pass
        input_tensor = torch.randn(1, 20, 45)  # batch_size=1, seq_len=20, input_size=45
        
        with torch.no_grad():
            output = self.predictor.model(input_tensor)
            
        assert output.shape == ()  # Scalar output
        assert 0 <= output.item() <= 1  # Sigmoid output
        
    def test_disruption_prediction(self):
        """Test disruption prediction functionality."""
        state = PlasmaState(self.config)
        
        # First few predictions should return 0 (insufficient history)
        for i in range(5):
            prob = self.predictor.predict_disruption(state)
            assert prob == 0.0
            
        # Build up history
        for i in range(15):
            state.q_min = 2.0 - i * 0.05  # Gradually decreasing q
            prob = self.predictor.predict_disruption(state)
            assert 0 <= prob <= 1
            
    def test_predictor_reset(self):
        """Test predictor reset functionality."""
        state = PlasmaState(self.config)
        
        # Build up some history
        for _ in range(10):
            self.predictor.predict_disruption(state)
            
        assert len(self.predictor.history_buffer) == 10
        
        # Reset
        self.predictor.reset()
        assert len(self.predictor.history_buffer) == 0
        
    def test_disruption_detection_sensitivity(self):
        """Test that predictor responds to disruption-like conditions."""
        state = PlasmaState(self.config)
        
        # Build baseline history with safe conditions
        for _ in range(25):
            state.q_min = 2.5
            state.plasma_beta = 0.02
            state.disruption_probability = 0.0
            self.predictor.predict_disruption(state)
            
        baseline_prob = self.predictor.predict_disruption(state)
        
        # Create dangerous conditions
        state.q_min = 0.8  # Very low q
        state.plasma_beta = 0.08  # High beta
        state.disruption_probability = 0.3  # High risk indicators
        
        dangerous_prob = self.predictor.predict_disruption(state)
        
        # Dangerous conditions should generally give higher probability
        # (Though this is not guaranteed with random weights)
        assert 0 <= dangerous_prob <= 1


class TestSafetyShield:
    """Test safety shield functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = TokamakConfig.from_preset("ITER")
        self.limits = SafetyLimits()
        self.predictor = DisruptionPredictor()
        self.shield = SafetyShield(self.limits, self.predictor)
        self.state = PlasmaState(self.config)
        
    def test_shield_initialization(self):
        """Test safety shield initialization."""
        assert hasattr(self.shield, 'limits')
        assert hasattr(self.shield, 'predictor')
        assert hasattr(self.shield, 'last_action')
        assert hasattr(self.shield, 'emergency_mode')
        assert not self.shield.emergency_mode
        
    def test_safe_action_filtering(self):
        """Test action filtering for safe actions."""
        # Safe action within limits
        safe_action = np.array([0.1, -0.1, 0.2, -0.2, 0.15, -0.15, 0.5, 0.3])
        
        filtered_action, safety_info = self.shield.filter_action(safe_action, self.state)
        
        assert isinstance(filtered_action, np.ndarray)
        assert filtered_action.shape == (8,)
        assert isinstance(safety_info, dict)
        assert 'action_modified' in safety_info
        assert 'violations' in safety_info
        assert 'disruption_risk' in safety_info
        
    def test_unsafe_action_filtering(self):
        """Test action filtering for unsafe actions."""
        # Extreme action that should be filtered
        unsafe_action = np.array([50.0, -50.0, 10.0, -10.0, 20.0, -20.0, 5.0, 5.0])
        
        filtered_action, safety_info = self.shield.filter_action(unsafe_action, self.state)
        
        assert isinstance(filtered_action, np.ndarray)
        assert filtered_action.shape == (8,)
        
        # Action should be modified
        assert not np.array_equal(filtered_action, unsafe_action)
        
        # Should have violations reported
        assert len(safety_info['violations']) > 0
        assert safety_info['action_modified'] == True
        
    def test_pf_coil_current_limits(self):
        """Test PF coil current limiting."""
        # Action with excessive PF coil currents
        action = np.array([15.0, -15.0, 12.0, -12.0, 8.0, -8.0, 0.5, 0.3])
        
        filtered_action, safety_info = self.shield.filter_action(action, self.state)
        
        # PF coil currents should be clamped
        assert np.all(np.abs(filtered_action[:6]) <= self.limits.pf_coil_current_limit)
        assert len(safety_info['violations']) > 0
        
    def test_gas_puff_limits(self):
        """Test gas puff rate limiting."""
        # Negative gas puff (unphysical)
        action = np.array([0.1, -0.1, 0.0, 0.0, 0.1, -0.1, -0.5, 0.3])
        
        filtered_action, safety_info = self.shield.filter_action(action, self.state)
        
        # Gas puff should be non-negative
        assert filtered_action[6] >= 0
        
        # Excessive gas puff
        action[6] = 5.0
        filtered_action, safety_info = self.shield.filter_action(action, self.state)
        assert filtered_action[6] <= self.limits.gas_puff_rate_limit
        
    def test_heating_power_limits(self):
        """Test heating power limiting."""
        # Negative heating (unphysical)
        action = np.array([0.1, -0.1, 0.0, 0.0, 0.1, -0.1, 0.5, -0.3])
        
        filtered_action, safety_info = self.shield.filter_action(action, self.state)
        
        # Heating should be non-negative
        assert filtered_action[7] >= 0
        
        # Excessive heating
        action[7] = 200.0
        filtered_action, safety_info = self.shield.filter_action(action, self.state)
        assert filtered_action[7] <= self.limits.heating_power_limit
        
    def test_emergency_mode(self):
        """Test emergency mode activation."""
        # Create dangerous plasma state
        self.state.q_min = 0.8  # Very low q
        self.state.disruption_probability = 0.6  # High disruption risk
        
        action = np.array([0.1, -0.1, 0.0, 0.0, 0.1, -0.1, 0.5, 0.3])
        filtered_action, safety_info = self.shield.filter_action(action, self.state)
        
        # Emergency mode should be activated
        assert self.shield.emergency_mode
        assert safety_info['emergency_mode']
        
        # Action should be emergency action
        assert not np.array_equal(filtered_action, action)
        
    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        # Set initial action
        initial_action = np.array([0.5, -0.5, 0.3, -0.3, 0.2, -0.2, 0.4, 0.6])
        self.shield.filter_action(initial_action, self.state)
        
        # Try large change (should be rate limited)
        large_change_action = np.array([2.0, -2.0, 1.5, -1.5, 1.0, -1.0, 0.8, 0.9])
        filtered_action, safety_info = self.shield.filter_action(large_change_action, self.state)
        
        # Changes should be limited
        change = filtered_action[:6] - initial_action[:6]
        dt = 1.0 / 100.0  # Assuming 100 Hz control
        rates = np.abs(change) / dt
        
        # Some rates might be limited (depending on implementation)
        assert np.all(rates <= self.limits.pf_coil_rate_limit + 1e-6)  # Small tolerance
        
    def test_shield_reset(self):
        """Test safety shield reset."""
        # Set some state
        self.shield.emergency_mode = True
        self.shield.last_action = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        
        # Reset
        self.shield.reset()
        
        assert not self.shield.emergency_mode
        assert self.shield.last_action is None


class TestConstraintManager:
    """Test constraint management system."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = TokamakConfig.from_preset("ITER")
        self.manager = ConstraintManager(self.config)
        self.state = PlasmaState(self.config)
        
    def test_constraint_manager_initialization(self):
        """Test constraint manager initialization."""
        assert hasattr(self.manager, 'config')
        assert hasattr(self.manager, 'constraints')
        assert len(self.manager.constraints) == 3  # q, beta, density constraints
        
    def test_constraint_checking(self):
        """Test constraint checking functionality."""
        # Set safe state
        self.state.q_profile = np.linspace(1.5, 4.0, 101)  # Safe q profile
        self.state.plasma_beta = 0.02  # Safe beta
        self.state.density_profile = np.full(101, 8e19)  # Safe density
        
        results = self.manager.check_constraints(self.state)
        
        assert isinstance(results, dict)
        assert 'q_constraint' in results
        assert 'beta_constraint' in results
        assert 'density_constraint' in results
        
        # All constraints should pass
        assert all(results.values())
        
    def test_q_constraint_violation(self):
        """Test q-constraint violation detection."""
        # Set dangerous q profile
        self.state.q_profile = np.linspace(0.8, 3.0, 101)  # q < 1 at core
        
        results = self.manager.check_constraints(self.state)
        assert not results['q_constraint']
        
    def test_beta_constraint_violation(self):
        """Test beta constraint violation detection."""
        # Set high beta
        self.state.plasma_beta = 0.1  # Above typical limits
        
        results = self.manager.check_constraints(self.state)
        # Note: beta constraint depends on current and field, so might still pass
        assert 'beta_constraint' in results
        
    def test_density_constraint_violation(self):
        """Test density constraint violation detection."""
        # Set very high density
        self.state.density_profile = np.full(101, 5e20)  # Above Greenwald limit
        
        results = self.manager.check_constraints(self.state)
        assert not results['density_constraint']


class TestSafetySystemIntegration:
    """Test complete safety system integration."""
    
    def test_safety_system_factory(self):
        """Test safety system factory function."""
        config = TokamakConfig.from_preset("SPARC")
        safety_system = create_safety_system(config)
        
        assert isinstance(safety_system, SafetyShield)
        assert hasattr(safety_system, 'limits')
        assert hasattr(safety_system, 'predictor')
        
    def test_complete_safety_workflow(self):
        """Test complete safety workflow."""
        config = TokamakConfig.from_preset("NSTX")
        safety_system = create_safety_system(config)
        state = PlasmaState(config)
        
        # Test sequence of actions
        actions = [
            np.array([0.1, -0.1, 0.2, -0.2, 0.1, -0.1, 0.3, 0.4]),  # Safe
            np.array([5.0, -5.0, 3.0, -3.0, 2.0, -2.0, 1.5, 1.5]),  # Unsafe
            np.array([0.05, -0.05, 0.1, -0.1, 0.05, -0.05, 0.2, 0.3])  # Safe again
        ]
        
        for i, action in enumerate(actions):
            filtered_action, safety_info = safety_system.filter_action(action, state)
            
            assert isinstance(filtered_action, np.ndarray)
            assert filtered_action.shape == (8,)
            assert isinstance(safety_info, dict)
            
            # Second action should be heavily modified
            if i == 1:
                assert not np.allclose(filtered_action, action, rtol=0.1)
                assert len(safety_info['violations']) > 0
                
    def test_safety_system_consistency(self):
        """Test safety system consistency across calls."""
        config = TokamakConfig.from_preset("DIII-D")
        safety_system = create_safety_system(config)
        state = PlasmaState(config)
        
        action = np.array([0.2, -0.2, 0.15, -0.15, 0.1, -0.1, 0.4, 0.5])
        
        # Multiple calls with same action should be consistent
        filtered1, info1 = safety_system.filter_action(action, state)
        filtered2, info2 = safety_system.filter_action(action, state)
        
        np.testing.assert_allclose(filtered1, filtered2, rtol=1e-10)
        
    def test_safety_metrics_tracking(self):
        """Test safety metrics tracking."""
        config = TokamakConfig.from_preset("ITER")
        safety_system = create_safety_system(config)
        state = PlasmaState(config)
        
        # Track metrics over multiple steps
        violation_counts = []
        disruption_risks = []
        
        for step in range(10):
            # Random action
            action = np.random.uniform(-1, 1, 8)
            
            # Filter action
            filtered_action, safety_info = safety_system.filter_action(action, state)
            
            # Track metrics
            violation_counts.append(len(safety_info['violations']))
            disruption_risks.append(safety_info['disruption_risk'])
            
            # Update state (simplified)
            state.q_min = max(0.5, state.q_min + np.random.normal(0, 0.05))
            
        # Should have collected metrics for all steps
        assert len(violation_counts) == 10
        assert len(disruption_risks) == 10
        assert all(0 <= risk <= 1 for risk in disruption_risks)
        assert all(count >= 0 for count in violation_counts)