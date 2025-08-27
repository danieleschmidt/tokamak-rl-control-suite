"""
Tests for Advanced Plasma Shape Control System
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch

from tokamak_rl.advanced_shape_control import (
    PlasmaTarget,
    ControlLimits,
    ShapeTrajectoryPlanner,
    AdaptiveShapeController,
    RealTimeShapeOptimizer,
    PlasmaShapeControlSuite,
    create_advanced_shape_control_system,
    EnhancedTokamakWrapper
)

class TestPlasmaTarget:
    def test_default_initialization(self):
        target = PlasmaTarget()
        assert target.elongation == 1.8
        assert target.triangularity == 0.4
        assert target.major_radius == 6.2
        assert target.safety_factor_min == 1.5

    def test_custom_initialization(self):
        target = PlasmaTarget(elongation=2.0, triangularity=0.5)
        assert target.elongation == 2.0
        assert target.triangularity == 0.5

class TestControlLimits:
    def test_default_initialization(self):
        limits = ControlLimits()
        assert limits.max_coil_current == 50000.0
        assert limits.min_q_factor == 1.2

class TestShapeTrajectoryPlanner:
    def setup_method(self):
        self.target = PlasmaTarget()
        self.limits = ControlLimits()
        self.planner = ShapeTrajectoryPlanner(self.target, self.limits)

    def test_initialization(self):
        assert self.planner.target == self.target
        assert self.planner.limits == self.limits
        assert self.planner.trajectory_length == 100

    def test_plan_trajectory(self):
        current_state = np.array([1.0, 0.2, 1.8])
        trajectory = self.planner.plan_trajectory(current_state, time_horizon=5.0)
        
        assert trajectory.shape == (100, 3)
        assert trajectory[0, 0] == pytest.approx(1.0, rel=1e-2)
        assert trajectory[-1, 0] == pytest.approx(1.8, rel=1e-2)

    def test_smooth_trajectory(self):
        time_points = np.linspace(0, 10, 50)
        trajectory = self.planner._create_smooth_trajectory(1.0, 2.0, time_points)
        
        assert len(trajectory) == 50
        assert trajectory[0] == pytest.approx(1.0, rel=1e-3)
        assert trajectory[-1] == pytest.approx(2.0, rel=1e-3)

class TestAdaptiveShapeController:
    def setup_method(self):
        self.controller = AdaptiveShapeController(input_dim=45, output_dim=8)

    def test_initialization(self):
        assert isinstance(self.controller, torch.nn.Module)

    def test_forward_pass(self):
        batch_size = 4
        input_dim = 45
        
        state = torch.randn(batch_size, input_dim)
        actions, shape_pred = self.controller(state)
        
        assert actions.shape == (batch_size, 8)
        assert shape_pred.shape == (batch_size, 3)
        assert torch.all(actions >= -1.0) and torch.all(actions <= 1.0)

    def test_single_input(self):
        state = torch.randn(1, 45)
        actions, shape_pred = self.controller(state)
        
        assert actions.shape == (1, 8)
        assert shape_pred.shape == (1, 3)

class TestRealTimeShapeOptimizer:
    def setup_method(self):
        self.controller = AdaptiveShapeController()
        self.planner = ShapeTrajectoryPlanner(PlasmaTarget(), ControlLimits())
        self.optimizer = RealTimeShapeOptimizer(self.controller, self.planner)

    def test_initialization(self):
        assert self.optimizer.controller == self.controller
        assert self.optimizer.planner == self.planner
        assert len(self.optimizer.loss_history) == 0

    def test_optimize_step(self):
        state = np.random.randn(45)
        target_shape = np.array([1.8, 0.4, 1.5])
        
        action = self.optimizer.optimize_step(state, target_shape)
        
        assert action.shape == (8,)
        assert len(self.optimizer.loss_history) == 1
        assert np.all(action >= -1.0) and np.all(action <= 1.0)

class TestPlasmaShapeControlSuite:
    def setup_method(self):
        self.suite = PlasmaShapeControlSuite()

    def test_initialization(self):
        assert isinstance(self.suite.target, PlasmaTarget)
        assert isinstance(self.suite.limits, ControlLimits)
        assert self.suite.current_state is None
        assert self.suite.step_count == 0

    def test_reset(self):
        initial_state = np.random.randn(45)
        self.suite.reset(initial_state)
        
        assert np.array_equal(self.suite.current_state, initial_state)
        assert self.suite.step_count == 0
        assert self.suite.trajectory is not None
        assert self.suite.trajectory.shape[0] == 100

    def test_control_step_without_reset(self):
        observation = np.random.randn(45)
        result = self.suite.control_step(observation)
        
        # Should automatically reset on first call
        assert 'action' in result
        assert 'target_shape' in result
        assert 'shape_error' in result
        assert result['action'].shape == (8,)

    def test_control_step_with_reset(self):
        initial_state = np.random.randn(45)
        observation = np.random.randn(45)
        
        self.suite.reset(initial_state)
        result = self.suite.control_step(observation)
        
        assert 'action' in result
        assert 'shape_error' in result
        assert 'control_effort' in result
        assert 'safety_violation' in result
        assert 'trajectory_progress' in result
        
        assert isinstance(result['safety_violation'], (bool, np.bool_))
        assert 0 <= result['trajectory_progress'] <= 1

    def test_multiple_control_steps(self):
        initial_state = np.random.randn(45)
        self.suite.reset(initial_state)
        
        for i in range(10):
            observation = np.random.randn(45)
            result = self.suite.control_step(observation)
            
            assert len(self.suite.metrics['shape_errors']) == i + 1
            assert self.suite.step_count == i + 1

    def test_apply_safety_constraints(self):
        action = np.array([2.0, -3.0, 0.5, 1.5, -0.8, 0.3, 1.2, -1.8])
        state = np.random.randn(45)
        
        constrained_action = self.suite._apply_safety_constraints(action, state)
        
        assert np.all(constrained_action >= -1.0)
        assert np.all(constrained_action <= 1.0)

    def test_compute_shape_error(self):
        state = np.array([1.5, 0.3] + [0.0] * 43)
        target = np.array([1.8, 0.4, 1.5])
        
        error = self.suite._compute_shape_error(state, target)
        assert error >= 0.0
        assert isinstance(error, float)

    def test_check_safety_violations(self):
        # Create state with potential violations
        state = np.zeros(45)
        state[10:20] = 1.0  # Safety factor profile - safe
        state[35:45] = 1e19  # Density profile - safe
        
        violation = self.suite._check_safety_violations(state)
        assert isinstance(violation, (bool, np.bool_))

    def test_get_performance_summary_empty(self):
        summary = self.suite.get_performance_summary()
        assert summary == {}

    def test_get_performance_summary_with_data(self):
        initial_state = np.random.randn(45)
        self.suite.reset(initial_state)
        
        # Run a few steps
        for _ in range(5):
            observation = np.random.randn(45)
            self.suite.control_step(observation)
        
        summary = self.suite.get_performance_summary()
        
        expected_keys = {
            'avg_shape_error', 'max_shape_error', 'avg_control_effort',
            'safety_violation_rate', 'final_optimization_loss', 'trajectory_completion'
        }
        assert set(summary.keys()) == expected_keys
        
        for value in summary.values():
            assert isinstance(value, (int, float))
            assert not np.isnan(value)

class TestFactoryFunction:
    def test_create_default_system(self):
        system = create_advanced_shape_control_system()
        
        assert isinstance(system, PlasmaShapeControlSuite)
        assert system.target.elongation == 1.8

    def test_create_custom_system(self):
        config = {
            'target_elongation': 2.0,
            'target_triangularity': 0.5,
            'min_q_factor': 1.4
        }
        
        system = create_advanced_shape_control_system(config)
        
        assert system.target.elongation == 2.0
        assert system.target.triangularity == 0.5
        assert system.limits.min_q_factor == 1.4

class TestEnhancedTokamakWrapper:
    def setup_method(self):
        self.mock_env = Mock()
        self.mock_env.step.return_value = (
            np.random.randn(45),  # obs
            0.5,                  # reward
            False,               # terminated
            False,               # truncated
            {}                   # info
        )
        self.mock_env.reset.return_value = (np.random.randn(45), {})
        
        self.wrapper = EnhancedTokamakWrapper(self.mock_env)

    def test_initialization(self):
        assert hasattr(self.wrapper, 'shape_controller')
        assert isinstance(self.wrapper.shape_controller, PlasmaShapeControlSuite)
        assert self.wrapper.use_advanced_control is True

    def test_reset(self):
        obs, info = self.wrapper.reset()
        
        assert obs.shape == (45,)
        assert hasattr(self.wrapper, '_last_obs')
        self.mock_env.reset.assert_called_once()

    def test_step_without_last_obs(self):
        action = np.random.randn(8)
        obs, reward, term, trunc, info = self.wrapper.step(action)
        
        assert obs.shape == (45,)
        assert isinstance(reward, (int, float))
        assert isinstance(term, (bool, np.bool_))
        self.mock_env.step.assert_called_once()

    def test_step_with_advanced_control(self):
        # First reset to set _last_obs
        self.wrapper.reset()
        
        action = np.random.randn(8)
        obs, reward, term, trunc, info = self.wrapper.step(action)
        
        assert 'advanced_control' in info
        assert isinstance(info['advanced_control'], dict)

    def test_step_disabled_advanced_control(self):
        self.wrapper.use_advanced_control = False
        self.wrapper.reset()
        
        action = np.random.randn(8)
        obs, reward, term, trunc, info = self.wrapper.step(action)
        
        assert 'advanced_control' not in info

class TestIntegration:
    def test_full_system_integration(self):
        """Test complete system integration"""
        system = create_advanced_shape_control_system()
        initial_state = np.random.randn(45)
        
        # Reset and run multiple steps
        system.reset(initial_state)
        
        results = []
        for i in range(10):
            obs = initial_state + 0.1 * i * np.random.randn(45)
            result = system.control_step(obs)
            results.append(result)
        
        # Verify all steps completed successfully
        assert len(results) == 10
        assert all('action' in r for r in results)
        
        # Verify metrics were collected
        summary = system.get_performance_summary()
        assert len(summary) > 0
        assert summary['trajectory_completion'] > 0

    def test_neural_network_learning(self):
        """Test that neural network parameters change during optimization"""
        controller = AdaptiveShapeController()
        planner = ShapeTrajectoryPlanner(PlasmaTarget(), ControlLimits())
        optimizer = RealTimeShapeOptimizer(controller, planner)
        
        # Get initial parameters
        initial_params = [p.clone() for p in controller.parameters()]
        
        # Run optimization steps
        for _ in range(5):
            state = np.random.randn(45)
            target = np.array([1.8, 0.4, 1.5])
            optimizer.optimize_step(state, target)
        
        # Check that parameters changed
        final_params = list(controller.parameters())
        
        params_changed = any(
            not torch.equal(init_p, final_p) 
            for init_p, final_p in zip(initial_params, final_params)
        )
        
        assert params_changed, "Neural network parameters should change during optimization"

if __name__ == "__main__":
    pytest.main([__file__])