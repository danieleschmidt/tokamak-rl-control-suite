"""
Pytest configuration and shared fixtures for tokamak-rl-control-suite tests.
"""

import pytest
import numpy as np
from typing import Dict, Any
from unittest.mock import Mock

@pytest.fixture
def basic_tokamak_config() -> Dict[str, Any]:
    """Basic tokamak configuration for testing."""
    return {
        "name": "ITER",
        "major_radius": 6.2,  # meters
        "minor_radius": 2.0,  # meters
        "magnetic_field": 5.3,  # Tesla
        "plasma_current": 15.0,  # MA
        "num_pf_coils": 6,
        "control_frequency": 100,  # Hz
        "safety_factor": 1.2,
    }


@pytest.fixture
def mock_plasma_state() -> Dict[str, np.ndarray]:
    """Mock plasma state for testing."""
    return {
        "plasma_current": np.array([15.0]),  # MA
        "plasma_beta": np.array([0.025]),  # normalized
        "q_profile": np.linspace(1.0, 4.0, 10),  # safety factor
        "shape_parameters": np.array([1.85, 0.33, 0.0, 0.1, 0.05, 0.02]),
        "magnetic_field": np.zeros(12),  # PF coil currents
        "density_profile": np.ones(10) * 8e19,  # m^-3
        "temperature_profile": np.array([20.0, 15.0, 10.0, 5.0, 2.0]),  # keV
        "error_signals": np.array([0.5]),  # cm
    }


@pytest.fixture
def mock_physics_solver():
    """Mock physics solver for testing."""
    solver = Mock()
    solver.solve_equilibrium.return_value = {
        "psi": np.random.randn(64, 64),  # flux surfaces
        "pressure": np.linspace(1e5, 0, 32),  # Pa
        "q_profile": np.linspace(1.0, 4.0, 32),
        "beta": 0.025,
        "converged": True,
    }
    solver.compute_shape_error.return_value = 1.2  # cm
    solver.check_stability.return_value = True
    return solver


@pytest.fixture  
def mock_safety_system():
    """Mock safety system for testing."""
    safety = Mock()
    safety.check_disruption_risk.return_value = False
    safety.filter_action.return_value = np.array([0.1, -0.2, 0.0, 0.3, -0.1, 0.2, 0.1, 0.0])
    safety.get_safety_constraints.return_value = {
        "q_min": 1.5,
        "density_limit": 1.2e20,
        "beta_limit": 0.04,
        "current_limit": 17.0,
    }
    return safety


@pytest.fixture
def sample_observation_space():
    """Sample observation space configuration."""
    return {
        "plasma_current": 1,
        "plasma_beta": 1, 
        "q_profile": 10,
        "shape_parameters": 6,
        "magnetic_field": 12,
        "density_profile": 10,
        "temperature_profile": 5,
        "error_signals": 1,
    }


@pytest.fixture
def sample_action_space():
    """Sample action space configuration.""" 
    return {
        "PF_coil_currents": 6,  # -1 to 1
        "gas_puff_rate": 1,     # 0 to 1
        "auxiliary_heating": 1   # 0 to 1
    }


@pytest.fixture
def physics_test_tolerances():
    """Physics validation tolerances for testing."""
    return {
        "equilibrium_residual": 1e-6,
        "shape_error": 0.1,  # cm
        "q_profile_error": 0.05,
        "pressure_profile_error": 0.02,
        "energy_conservation": 1e-4,
    }


@pytest.fixture(scope="session")
def benchmark_config():
    """Configuration for performance benchmarks."""
    return {
        "num_episodes": 10,
        "max_steps_per_episode": 1000,
        "time_limit_seconds": 60,
        "memory_limit_mb": 512,
        "acceptable_step_time_ms": 50,
    }


@pytest.fixture
def integration_test_scenarios():
    """Test scenarios for integration testing."""
    return [
        {
            "name": "startup_scenario",
            "initial_current": 0.0,
            "target_current": 15.0,
            "duration": 100,  # time steps
            "expected_final_error": 2.0,  # cm
        },
        {
            "name": "shape_control_scenario", 
            "target_elongation": 1.85,
            "target_triangularity": 0.33,
            "duration": 200,
            "expected_convergence": True,
        },
        {
            "name": "disruption_avoidance_scenario",
            "q_min_threshold": 1.5,
            "density_ramp_rate": 1e19,  # m^-3/s  
            "expected_disruption": False,
        },
    ]


# Pytest markers for test categorization
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "physics: marks tests as physics validation tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance benchmarks"
    )
    config.addinivalue_line(
        "markers", "safety: marks tests as safety-critical tests"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running"
    )


# Skip expensive tests in regular CI
def pytest_collection_modifyitems(config, items):
    """Modify test collection based on markers."""
    if config.getoption("--quick"):
        skip_slow = pytest.mark.skip(reason="--quick specified, skipping slow tests")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)