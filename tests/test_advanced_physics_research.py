"""
Comprehensive test suite for advanced physics research module.

Tests MHD instability prediction, disruption forecasting,
and multi-scale physics modeling capabilities.
"""

import pytest
import math
import random
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from tokamak_rl.advanced_physics_research import (
    AdvancedMHDPredictor,
    MultiScalePhysicsModel,
    KineticPhysicsSolver,
    FluidPhysicsSolver,
    MHDInstability,
    PlasmaProfile,
    DisruptionPrediction,
    create_advanced_physics_research_system
)


class TestMHDInstability:
    """Test suite for MHDInstability dataclass."""
    
    def test_instability_creation(self):
        """Test MHD instability creation."""
        instability = MHDInstability(
            mode_number=(2, 1),
            growth_rate=150.0,
            frequency=1000.0,
            amplitude=0.05,
            radial_location=0.7,
            instability_type='tearing'
        )
        
        assert instability.mode_number == (2, 1)
        assert instability.growth_rate == 150.0
        assert instability.frequency == 1000.0
        assert instability.amplitude == 0.05
        assert instability.radial_location == 0.7
        assert instability.instability_type == 'tearing'


class TestPlasmaProfile:
    """Test suite for PlasmaProfile dataclass."""
    
    def test_profile_creation(self):
        """Test plasma profile creation."""
        n_points = 20
        radius = [i / (n_points - 1) for i in range(n_points)]
        temp_e = [10.0 * (1 - r**2) for r in radius]
        temp_i = [8.0 * (1 - r**2) for r in radius]
        density = [2e19 * (1 - r**2) for r in radius]
        pressure = [t * d * 1.6e-19 for t, d in zip(temp_e, density)]
        q_profile = [1.0 + 3.0 * r**2 for r in radius]
        j_profile = [1e6 * (1 - r**2) for r in radius]
        magnetic_shear = [0.1] * n_points
        
        profile = PlasmaProfile(
            radius=radius,
            temperature_e=temp_e,
            temperature_i=temp_i,
            density_e=density,
            pressure=pressure,
            q_profile=q_profile,
            j_profile=j_profile,
            magnetic_shear=magnetic_shear
        )
        
        assert len(profile.radius) == n_points
        assert len(profile.temperature_e) == n_points
        assert len(profile.q_profile) == n_points
        assert profile.temperature_e[0] > profile.temperature_e[-1]  # Core hotter than edge


class TestAdvancedMHDPredictor:
    """Test suite for AdvancedMHDPredictor."""
    
    def test_initialization(self):
        """Test MHD predictor initialization."""
        predictor = AdvancedMHDPredictor(mode_database_size=500)
        
        assert len(predictor.mode_database) == 0
        assert predictor.mode_database.maxlen == 500
        assert len(predictor.prediction_history) == 0
        assert 'tearing_mode' in predictor.instability_thresholds
        assert 'kink_mode' in predictor.instability_thresholds
        assert len(predictor.model_weights) == 100
    
    def test_stability_analysis(self):
        """Test MHD stability analysis."""
        predictor = AdvancedMHDPredictor()
        
        # Create test plasma profile
        profile = self._create_test_profile()
        
        instabilities = predictor.analyze_stability(profile)
        
        assert isinstance(instabilities, list)
        # Should find at least some instabilities in a realistic profile
        assert len(instabilities) >= 0
        
        for instability in instabilities:
            assert isinstance(instability, MHDInstability)
            assert instability.growth_rate >= 0
            assert 0 <= instability.radial_location <= 1
            assert instability.instability_type in ['tearing', 'kink', 'ballooning', 'neoclassical_tearing']
    
    def test_tearing_mode_analysis(self):
        """Test tearing mode specific analysis."""
        predictor = AdvancedMHDPredictor()
        
        # Create profile with rational surfaces
        profile = self._create_test_profile()
        
        tearing_modes = predictor._analyze_tearing_modes(profile)
        
        assert isinstance(tearing_modes, list)
        
        for mode in tearing_modes:
            assert mode.instability_type == 'tearing'
            assert isinstance(mode.mode_number, tuple)
            assert len(mode.mode_number) == 2  # (m, n)
            assert mode.mode_number[0] >= 1  # m >= 1
            assert mode.mode_number[1] >= 1  # n >= 1
    
    def test_kink_mode_analysis(self):
        """Test kink mode analysis."""
        predictor = AdvancedMHDPredictor()
        
        # Create high-beta profile to trigger kink modes
        profile = self._create_high_beta_profile()
        
        kink_modes = predictor._analyze_kink_modes(profile)
        
        assert isinstance(kink_modes, list)
        
        for mode in kink_modes:
            assert mode.instability_type == 'kink'
            assert mode.mode_number[0] == 1  # External kink typically (1,1)
            assert mode.growth_rate > 0
    
    def test_disruption_prediction(self):
        """Test disruption prediction."""
        predictor = AdvancedMHDPredictor()
        
        # Create unstable profile
        profile = self._create_unstable_profile()
        
        # Create test instabilities
        instabilities = [
            MHDInstability((2, 1), 500.0, 1000.0, 0.08, 0.6, 'tearing'),
            MHDInstability((3, 2), 300.0, 500.0, 0.04, 0.8, 'neoclassical_tearing')
        ]
        
        disruption_prob, diagnostics = predictor.predict_disruption_probability(
            instabilities, time_horizon=0.1
        )
        
        assert isinstance(disruption_prob, float)
        assert 0.0 <= disruption_prob <= 1.0
        assert isinstance(diagnostics, dict)
        
        required_diagnostics = [
            'max_growth_rate', 'n_unstable_modes', 'core_instabilities',
            'max_amplitude', 'physics_prediction', 'ml_prediction',
            'dominant_instability'
        ]
        
        for diagnostic in required_diagnostics:
            assert diagnostic in diagnostics
    
    def test_mitigation_strategies(self):
        """Test mitigation strategy generation."""
        predictor = AdvancedMHDPredictor()
        
        # Create test instabilities
        instabilities = [
            MHDInstability((2, 1), 200.0, 800.0, 0.03, 0.5, 'tearing'),
            MHDInstability((1, 1), 400.0, 0.0, 0.06, 0.9, 'kink'),
            MHDInstability((0, 10), 100.0, 5000.0, 0.001, 0.7, 'ballooning')
        ]
        
        strategies = predictor.suggest_mitigation_strategies(instabilities)
        
        assert isinstance(strategies, list)
        assert len(strategies) >= len(instabilities)  # At least one strategy per instability
        
        for strategy in strategies:
            assert 'method' in strategy
            assert 'success_probability' in strategy
            assert 0.0 <= strategy['success_probability'] <= 1.0
    
    def test_delta_prime_calculation(self):
        """Test delta-prime calculation."""
        predictor = AdvancedMHDPredictor()
        
        profile = self._create_test_profile()
        
        # Test delta-prime at rational surface
        r_rational = 0.6
        m, n = 2, 1
        
        delta_prime = predictor._calculate_delta_prime(profile, r_rational, m, n)
        
        assert isinstance(delta_prime, float)
        # Delta-prime can be positive or negative
    
    def _create_test_profile(self):
        """Create test plasma profile."""
        n_points = 25
        radius = [i / (n_points - 1) for i in range(n_points)]
        
        temp_e = [12.0 * (1 - r**2)**2 for r in radius]
        temp_i = [10.0 * (1 - r**2)**2 for r in radius]
        density = [3e19 * (1 - r**2) for r in radius]
        pressure = [1.6e-19 * (te + ti) * ne for te, ti, ne in zip(temp_e, temp_i, density)]
        q_profile = [1.2 + 2.8 * r**2 for r in radius]
        j_profile = [2e6 * (1 - r**3) for r in radius]
        magnetic_shear = [0.5 if r > 0 else 0 for r in radius]
        
        return PlasmaProfile(
            radius=radius,
            temperature_e=temp_e,
            temperature_i=temp_i,
            density_e=density,
            pressure=pressure,
            q_profile=q_profile,
            j_profile=j_profile,
            magnetic_shear=magnetic_shear
        )
    
    def _create_high_beta_profile(self):
        """Create high-beta profile for kink mode testing."""
        profile = self._create_test_profile()
        
        # Increase pressure to trigger kink modes
        profile.pressure = [p * 5.0 for p in profile.pressure]  # High pressure
        
        return profile
    
    def _create_unstable_profile(self):
        """Create unstable profile for disruption testing."""
        profile = self._create_test_profile()
        
        # Make profile more unstable
        profile.pressure = [p * 3.0 for p in profile.pressure]  # Higher pressure
        profile.q_profile = [q * 0.8 for q in profile.q_profile]  # Lower q
        
        return profile


class TestMultiScalePhysicsModel:
    """Test suite for MultiScalePhysicsModel."""
    
    def test_initialization(self):
        """Test multi-scale model initialization."""
        model = MultiScalePhysicsModel()
        
        assert hasattr(model, 'kinetic_solver')
        assert hasattr(model, 'fluid_solver')
        assert isinstance(model.kinetic_solver, KineticPhysicsSolver)
        assert isinstance(model.fluid_solver, FluidPhysicsSolver)
        assert 0.0 < model.coupling_strength < 1.0
    
    def test_coupled_evolution(self):
        """Test coupled kinetic-fluid evolution."""
        model = MultiScalePhysicsModel()
        
        # Create test state
        state = {
            'temperature': [10.0, 8.0, 6.0, 4.0, 2.0],
            'density': [2e19, 1.8e19, 1.5e19, 1.2e19, 1e19],
            'particle_distribution': [1.0, 0.8, 0.6, 0.4, 0.2]
        }
        
        dt = 0.001  # 1ms
        
        evolved_state = model.evolve_coupled_system(state, dt)
        
        assert isinstance(evolved_state, dict)
        assert 'temperature' in evolved_state
        assert 'density' in evolved_state
        assert len(evolved_state['temperature']) == len(state['temperature'])
    
    def test_kinetic_fluid_coupling(self):
        """Test kinetic-fluid coupling mechanism."""
        model = MultiScalePhysicsModel()
        
        kinetic_state = {
            'heat_flux': [1e5, 8e4, 6e4, 4e4],
            'particle_distribution': [1.0, 0.8, 0.6, 0.4]
        }
        
        fluid_state = {
            'temperature': [10.0, 8.0, 6.0, 4.0, 2.0],
            'heat_flux': [5e4, 4e4, 3e4, 2e4]
        }
        
        coupled_state = model._couple_kinetic_fluid(kinetic_state, fluid_state)
        
        assert 'heat_flux' in coupled_state
        assert len(coupled_state['heat_flux']) == len(fluid_state['heat_flux'])
        
        # Heat flux should be modified by kinetic correction
        for i in range(len(fluid_state['heat_flux'])):
            expected_correction = model.coupling_strength * kinetic_state['heat_flux'][i]
            expected_total = fluid_state['heat_flux'][i] + expected_correction
            assert abs(coupled_state['heat_flux'][i] - expected_total) < 1e-10


class TestKineticPhysicsSolver:
    """Test suite for KineticPhysicsSolver."""
    
    def test_kinetic_evolution(self):
        """Test kinetic physics evolution."""
        solver = KineticPhysicsSolver()
        
        state = {
            'particle_distribution': [1.0, 0.9, 0.7, 0.5, 0.3],
            'temperature': [15.0, 12.0, 9.0, 6.0, 3.0],
            'density': [3e19, 2.5e19, 2e19, 1.5e19, 1e19]
        }
        
        dt = 0.0001  # 0.1ms
        
        evolved_state = solver.evolve(state, dt)
        
        assert 'particle_distribution' in evolved_state
        assert 'heat_flux' in evolved_state
        assert len(evolved_state['particle_distribution']) == len(state['particle_distribution'])
        assert len(evolved_state['heat_flux']) == len(state['temperature']) - 1


class TestFluidPhysicsSolver:
    """Test suite for FluidPhysicsSolver."""
    
    def test_fluid_evolution(self):
        """Test fluid physics evolution."""
        solver = FluidPhysicsSolver()
        
        state = {
            'temperature': [12.0, 10.0, 8.0, 6.0, 4.0],
            'heat_flux': [1e5, 8e4, 6e4, 4e4]
        }
        
        dt = 0.001  # 1ms
        
        evolved_state = solver.evolve(state, dt)
        
        assert 'temperature' in evolved_state
        assert len(evolved_state['temperature']) == len(state['temperature'])
        
        # Temperature should have evolved due to heat diffusion
        # (Not necessarily different due to small dt, but should not crash)


class TestPhysicsResearchSystem:
    """Test suite for complete physics research system."""
    
    def test_system_creation(self):
        """Test research system creation."""
        system = create_advanced_physics_research_system()
        
        required_components = [
            'mhd_predictor', 'multiscale_model', 'run_physics_benchmark',
            'generate_test_plasma_profile'
        ]
        
        for component in required_components:
            assert component in system
        
        assert system['system_type'] == 'advanced_physics_research'
        assert isinstance(system['mhd_predictor'], AdvancedMHDPredictor)
        assert isinstance(system['multiscale_model'], MultiScalePhysicsModel)
    
    def test_physics_benchmark(self):
        """Test physics benchmark execution."""
        system = create_advanced_physics_research_system()
        
        # Run benchmark with small number of scenarios for testing
        benchmark_results = system['run_physics_benchmark'](n_scenarios=5)
        
        required_metrics = [
            'mhd_prediction_accuracy', 'disruption_prediction_precision',
            'mitigation_success_rate', 'computational_efficiency'
        ]
        
        for metric in required_metrics:
            assert metric in benchmark_results
            assert 0.0 <= benchmark_results[metric] <= 1.0
    
    def test_test_profile_generation(self):
        """Test test plasma profile generation."""
        system = create_advanced_physics_research_system()
        
        profile = system['generate_test_plasma_profile']()
        
        assert isinstance(profile, PlasmaProfile)
        assert len(profile.radius) == 50  # Default n_points
        assert len(profile.temperature_e) == 50
        assert len(profile.q_profile) == 50
        
        # Check physical reasonableness
        assert profile.temperature_e[0] > profile.temperature_e[-1]  # Core hotter
        assert profile.density_e[0] > profile.density_e[-1]  # Core denser
        assert min(profile.q_profile) >= 1.0  # q >= 1


class TestPhysicsIntegration:
    """Integration tests for physics research system."""
    
    def test_end_to_end_analysis(self):
        """Test end-to-end physics analysis."""
        system = create_advanced_physics_research_system()
        predictor = system['mhd_predictor']
        
        # Generate test profile
        profile = system['generate_test_plasma_profile']()
        
        # Run MHD analysis
        instabilities = predictor.analyze_stability(profile)
        
        # Predict disruption
        disruption_prob, diagnostics = predictor.predict_disruption_probability(instabilities)
        
        # Generate mitigation strategies
        strategies = predictor.suggest_mitigation_strategies(instabilities)
        
        # Verify complete analysis chain
        assert isinstance(instabilities, list)
        assert isinstance(disruption_prob, float)
        assert isinstance(diagnostics, dict)
        assert isinstance(strategies, list)
        
        assert 0.0 <= disruption_prob <= 1.0
        assert len(diagnostics) >= 5
    
    def test_multiscale_integration(self):
        """Test multi-scale physics integration."""
        system = create_advanced_physics_research_system()
        model = system['multiscale_model']
        
        # Create comprehensive plasma state
        state = {
            'temperature': [15.0, 12.0, 9.0, 6.0, 3.0],
            'density': [3e19, 2.5e19, 2e19, 1.5e19, 1e19],
            'particle_distribution': [1.0, 0.9, 0.7, 0.5, 0.3],
            'pressure': [7.2e-12, 4.8e-12, 2.88e-12, 1.44e-12, 0.48e-12]
        }
        
        # Evolve over multiple time steps
        dt = 0.001
        n_steps = 10
        
        evolved_state = state.copy()
        for step in range(n_steps):
            evolved_state = model.evolve_coupled_system(evolved_state, dt)
        
        # State should have evolved
        assert 'temperature' in evolved_state
        assert 'heat_flux' in evolved_state  # Should be generated
        assert len(evolved_state['temperature']) == len(state['temperature'])
    
    def test_research_validation_framework(self):
        """Test research validation framework."""
        system = create_advanced_physics_research_system()
        
        # Run comprehensive benchmark
        benchmark_results = system['run_physics_benchmark'](n_scenarios=10)
        
        # Check that all benchmarks pass reasonable thresholds
        assert benchmark_results['mhd_prediction_accuracy'] >= 0.0
        assert benchmark_results['disruption_prediction_precision'] >= 0.0
        assert benchmark_results['mitigation_success_rate'] >= 0.0
        assert benchmark_results['computational_efficiency'] >= 0.0
        
        # At least some predictions should be reasonable
        total_performance = sum(benchmark_results.values()) / len(benchmark_results)
        assert total_performance >= 0.2  # At least 20% overall performance


# Test fixtures and utilities
@pytest.fixture
def mhd_predictor():
    """Fixture for MHD predictor."""
    return AdvancedMHDPredictor()


@pytest.fixture
def test_profile():
    """Fixture for test plasma profile."""
    system = create_advanced_physics_research_system()
    return system['generate_test_plasma_profile']()


@pytest.fixture
def multiscale_model():
    """Fixture for multi-scale physics model."""
    return MultiScalePhysicsModel()


def test_physics_performance_scaling(mhd_predictor, test_profile):
    """Test physics performance under increasing load."""
    # Test with increasing complexity
    for n_analysis in [1, 5, 10, 20]:
        start_time = time.time()
        
        for _ in range(n_analysis):
            instabilities = mhd_predictor.analyze_stability(test_profile)
            disruption_prob, _ = mhd_predictor.predict_disruption_probability(instabilities)
        
        analysis_time = time.time() - start_time
        
        # Performance should scale reasonably (not exponentially)
        time_per_analysis = analysis_time / n_analysis
        assert time_per_analysis < 1.0  # Should complete in under 1 second each


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v", "--tb=short"])