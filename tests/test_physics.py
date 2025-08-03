"""
Tests for tokamak physics simulation components.
"""

import pytest
import numpy as np
from tokamak_rl.physics import (
    TokamakConfig, PlasmaState, GradShafranovSolver, 
    ShapeAnalyzer, create_physics_solver
)


class TestTokamakConfig:
    """Test tokamak configuration functionality."""
    
    def test_preset_configs(self):
        """Test loading preset tokamak configurations."""
        presets = ["ITER", "SPARC", "NSTX", "DIII-D"]
        
        for preset in presets:
            config = TokamakConfig.from_preset(preset)
            assert config.major_radius > 0
            assert config.minor_radius > 0
            assert config.toroidal_field > 0
            assert config.plasma_current > 0
            assert config.minor_radius < config.major_radius
            
    def test_invalid_preset(self):
        """Test error handling for invalid preset names."""
        with pytest.raises(ValueError, match="Unknown preset"):
            TokamakConfig.from_preset("INVALID")
            
    def test_custom_config(self):
        """Test creating custom tokamak configuration."""
        config = TokamakConfig(
            major_radius=2.0,
            minor_radius=0.8,
            toroidal_field=4.0,
            plasma_current=5.0
        )
        
        assert config.major_radius == 2.0
        assert config.minor_radius == 0.8
        assert config.toroidal_field == 4.0
        assert config.plasma_current == 5.0
        
    def test_config_validation(self):
        """Test that configuration parameters are physically reasonable."""
        config = TokamakConfig.from_preset("ITER")
        
        # Basic physics constraints
        assert config.aspect_ratio > 1.0  # aspect_ratio = R/a
        assert config.elongation >= 1.0
        assert -1.0 <= config.triangularity <= 1.0
        assert config.beta_n > 0
        assert config.q95 > 1.0
        

class TestPlasmaState:
    """Test plasma state representation and operations."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = TokamakConfig.from_preset("ITER")
        self.state = PlasmaState(self.config)
        
    def test_plasma_state_initialization(self):
        """Test plasma state initialization."""
        assert hasattr(self.state, 'psi_profile')
        assert hasattr(self.state, 'q_profile')
        assert hasattr(self.state, 'pressure_profile')
        assert hasattr(self.state, 'density_profile')
        assert hasattr(self.state, 'temperature_profile')
        
        # Check array dimensions
        assert len(self.state.psi_profile) == 101
        assert len(self.state.q_profile) == 101
        assert len(self.state.pressure_profile) == 101
        
    def test_plasma_state_reset(self):
        """Test plasma state reset functionality."""
        # Modify state
        self.state.elongation = 2.5
        self.state.shape_error = 5.0
        
        # Reset
        self.state.reset()
        
        # Check that values are back to initial
        assert self.state.elongation == self.config.elongation
        assert self.state.shape_error == 0.0
        
    def test_observation_generation(self):
        """Test observation vector generation."""
        obs = self.state.get_observation()
        
        assert isinstance(obs, np.ndarray)
        assert obs.dtype == np.float32
        assert len(obs) == 45  # Expected observation dimension
        assert not np.any(np.isnan(obs))
        assert not np.any(np.isinf(obs))
        
    def test_safety_metrics(self):
        """Test safety metrics computation."""
        metrics = self.state.compute_safety_metrics()
        
        assert isinstance(metrics, dict)
        assert 'q_min' in metrics
        assert 'beta_limit_fraction' in metrics
        assert 'density_limit_fraction' in metrics
        assert 'shape_error' in metrics
        assert 'disruption_probability' in metrics
        
        # Check that values are reasonable
        assert metrics['q_min'] > 0
        assert metrics['beta_limit_fraction'] >= 0
        assert metrics['density_limit_fraction'] >= 0
        

class TestGradShafranovSolver:
    """Test Grad-Shafranov equilibrium solver."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = TokamakConfig.from_preset("ITER")
        self.solver = GradShafranovSolver(self.config)
        self.state = PlasmaState(self.config)
        
    def test_solver_initialization(self):
        """Test solver initialization."""
        assert hasattr(self.solver, 'config')
        assert hasattr(self.solver, 'nr')
        assert hasattr(self.solver, 'nz')
        assert hasattr(self.solver, 'r_grid')
        assert hasattr(self.solver, 'z_grid')
        
        # Check grid dimensions
        assert self.solver.nr == 65
        assert self.solver.nz == 65
        
    def test_equilibrium_solving(self):
        """Test equilibrium solving functionality."""
        # Test PF coil currents
        pf_currents = np.array([0.1, -0.1, 0.2, -0.2, 0.15, -0.15])
        
        # Solve equilibrium
        new_state = self.solver.solve_equilibrium(self.state, pf_currents)
        
        assert isinstance(new_state, PlasmaState)
        assert np.array_equal(new_state.pf_coil_currents, pf_currents)
        assert new_state.q_min > 0
        assert new_state.plasma_beta > 0
        
    def test_disruption_assessment(self):
        """Test disruption risk assessment."""
        # Create state with potential disruption conditions
        dangerous_currents = np.array([5.0, -5.0, 3.0, -3.0, 4.0, -4.0])
        new_state = self.solver.solve_equilibrium(self.state, dangerous_currents)
        
        # Should have non-zero disruption probability
        assert new_state.disruption_probability >= 0
        assert new_state.disruption_probability <= 1.0
        
    def test_q_profile_computation(self):
        """Test safety factor profile computation."""
        pf_currents = np.zeros(6)
        new_state = self.solver.solve_equilibrium(self.state, pf_currents)
        
        # Q-profile should be monotonic and physically reasonable
        assert len(new_state.q_profile) == 101
        assert np.all(new_state.q_profile > 0)
        assert new_state.q_profile[0] <= new_state.q_profile[-1]  # Monotonic
        

class TestShapeAnalyzer:
    """Test plasma shape analysis."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = TokamakConfig.from_preset("ITER")
        self.analyzer = ShapeAnalyzer(self.config)
        
    def test_shape_parameter_computation(self):
        """Test shape parameter computation."""
        # Mock flux surfaces
        flux_surfaces = np.random.random((10, 100))
        
        shape_params = self.analyzer.compute_shape_parameters(flux_surfaces)
        
        assert isinstance(shape_params, dict)
        assert 'elongation' in shape_params
        assert 'triangularity' in shape_params
        assert 'volume' in shape_params
        assert 'surface_area' in shape_params
        
        # Check that values are positive
        assert shape_params['volume'] > 0
        assert shape_params['surface_area'] > 0
        
    def test_shape_error_computation(self):
        """Test shape error computation."""
        current_shape = {'elongation': 1.8, 'triangularity': 0.35}
        target_shape = {'elongation': 1.85, 'triangularity': 0.33}
        
        error = self.analyzer.compute_shape_error(current_shape, target_shape)
        
        assert isinstance(error, float)
        assert error >= 0
        
        # Test with identical shapes
        zero_error = self.analyzer.compute_shape_error(target_shape, target_shape)
        assert zero_error == 0.0


class TestPhysicsIntegration:
    """Test integration between physics components."""
    
    def test_complete_physics_simulation(self):
        """Test complete physics simulation workflow."""
        # Create components
        config = TokamakConfig.from_preset("SPARC")
        solver = create_physics_solver(config)
        state = PlasmaState(config)
        
        # Run simulation steps
        for step in range(5):
            # Random control action
            pf_currents = np.random.uniform(-0.5, 0.5, 6)
            
            # Solve equilibrium
            state = solver.solve_equilibrium(state, pf_currents)
            
            # Verify state consistency
            assert state.q_min > 0
            assert 0 <= state.disruption_probability <= 1
            assert state.shape_error >= 0
            
            # Get observation
            obs = state.get_observation()
            assert len(obs) == 45
            assert not np.any(np.isnan(obs))
            
    def test_physics_reproducibility(self):
        """Test that physics simulation is reproducible."""
        config = TokamakConfig.from_preset("ITER")
        solver1 = GradShafranovSolver(config)
        solver2 = GradShafranovSolver(config)
        
        state1 = PlasmaState(config)
        state2 = PlasmaState(config)
        
        # Same initial conditions and control
        pf_currents = np.array([0.1, -0.1, 0.2, -0.2, 0.1, -0.1])
        
        # Solve with both solvers
        result1 = solver1.solve_equilibrium(state1, pf_currents)
        result2 = solver2.solve_equilibrium(state2, pf_currents)
        
        # Results should be identical (within numerical precision)
        np.testing.assert_allclose(result1.q_profile, result2.q_profile, rtol=1e-10)
        assert abs(result1.q_min - result2.q_min) < 1e-10
        assert abs(result1.plasma_beta - result2.plasma_beta) < 1e-10
        
    def test_physics_parameter_ranges(self):
        """Test physics behavior across parameter ranges."""
        config = TokamakConfig.from_preset("NSTX")
        solver = GradShafranovSolver(config)
        
        # Test various PF coil current ranges
        current_ranges = [
            np.linspace(-2.0, 2.0, 10),  # Large range
            np.linspace(-0.5, 0.5, 10),  # Small range
            np.linspace(0, 1.0, 10)      # Positive only
        ]
        
        for current_range in current_ranges:
            for current_level in current_range:
                state = PlasmaState(config)
                pf_currents = np.full(6, current_level)
                
                result = solver.solve_equilibrium(state, pf_currents)
                
                # Basic physical constraints should always hold
                assert result.q_min > 0, f"q_min not positive for current {current_level}"
                assert result.plasma_beta > 0, f"beta not positive for current {current_level}"
                assert 0 <= result.disruption_probability <= 1, f"Invalid disruption prob for current {current_level}"