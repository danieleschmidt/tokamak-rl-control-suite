"""
Test data fixtures for tokamak-rl-control-suite.

This module provides standardized test data for physics validation,
performance benchmarking, and integration testing.
"""

import numpy as np
from typing import Dict, Any, List, Tuple
from pathlib import Path


class TestDataGenerator:
    """Generates consistent test data for various test scenarios."""
    
    def __init__(self, seed: int = 42):
        """Initialize with reproducible random seed."""
        self.rng = np.random.RandomState(seed)
    
    def generate_equilibrium_data(self, 
                                grid_size: int = 64,
                                major_radius: float = 6.2,
                                minor_radius: float = 2.0) -> Dict[str, np.ndarray]:
        """Generate synthetic equilibrium data for testing."""
        # Create spatial grid
        r_grid = np.linspace(major_radius - minor_radius, 
                           major_radius + minor_radius, grid_size)
        z_grid = np.linspace(-minor_radius, minor_radius, grid_size)
        R, Z = np.meshgrid(r_grid, z_grid)
        
        # Generate realistic flux surfaces (Solov'ev solution approximation)
        psi_axis = 0.0
        psi_boundary = 1.0
        
        # Simplified flux function
        normalized_r = (R - major_radius) / minor_radius
        normalized_z = Z / minor_radius
        rho_squared = normalized_r**2 + normalized_z**2
        
        # Create flux surfaces with some noise for realism
        psi = psi_axis + (psi_boundary - psi_axis) * rho_squared
        psi += 0.01 * self.rng.randn(*psi.shape)  # Add noise
        
        # Generate pressure and safety factor profiles
        rho_1d = np.linspace(0, 1, 32)
        pressure = 1e5 * (1 - rho_1d**2)**2  # Parabolic pressure profile
        q_profile = 1.0 + 2.5 * rho_1d**2    # Realistic q-profile
        
        return {
            "psi": psi,
            "R_grid": R,
            "Z_grid": Z,
            "pressure": pressure,
            "q_profile": q_profile,
            "rho": rho_1d,
            "psi_axis": psi_axis,
            "psi_boundary": psi_boundary,
        }
    
    def generate_control_trajectory(self, 
                                  num_steps: int = 1000,
                                  num_actuators: int = 8) -> Dict[str, np.ndarray]:
        """Generate realistic control trajectory for testing."""
        time = np.linspace(0, 10, num_steps)  # 10 seconds
        
        # PF coil currents (6 coils)
        pf_currents = np.zeros((num_steps, 6))
        for i in range(6):
            # Each coil has different characteristic frequency
            freq = 0.1 + 0.05 * i
            pf_currents[:, i] = 0.5 * np.sin(2 * np.pi * freq * time)
            pf_currents[:, i] += 0.1 * self.rng.randn(num_steps)
        
        # Gas puffing (0 to 1)
        gas_puff = 0.3 + 0.2 * np.sin(2 * np.pi * 0.05 * time)
        gas_puff = np.clip(gas_puff + 0.05 * self.rng.randn(num_steps), 0, 1)
        
        # Auxiliary heating (0 to 1)
        heating = 0.7 + 0.1 * np.sin(2 * np.pi * 0.02 * time)
        heating = np.clip(heating + 0.03 * self.rng.randn(num_steps), 0, 1)
        
        # Combine into action array
        actions = np.column_stack([
            pf_currents,
            gas_puff,
            heating
        ])
        
        return {
            "time": time,
            "actions": actions,
            "pf_currents": pf_currents,
            "gas_puff": gas_puff,
            "heating": heating,
        }
    
    def generate_plasma_states(self, 
                             num_states: int = 100) -> Dict[str, np.ndarray]:
        """Generate diverse plasma states for testing."""
        states = {
            "plasma_current": self.rng.uniform(5.0, 20.0, num_states),      # MA
            "plasma_beta": self.rng.uniform(0.01, 0.05, num_states),        # normalized
            "elongation": self.rng.uniform(1.5, 2.2, num_states),           # shape
            "triangularity": self.rng.uniform(0.1, 0.5, num_states),        # shape
            "density": self.rng.uniform(3e19, 1.5e20, num_states),          # m^-3
            "temperature": self.rng.uniform(5.0, 25.0, num_states),         # keV
        }
        
        # Generate q-profiles for each state
        q_profiles = np.zeros((num_states, 10))
        for i in range(num_states):
            q_axis = self.rng.uniform(0.8, 1.2)
            q_edge = self.rng.uniform(2.5, 4.5)
            rho = np.linspace(0, 1, 10)
            q_profiles[i] = q_axis + (q_edge - q_axis) * rho**2
        
        states["q_profiles"] = q_profiles
        
        return states
    
    def generate_disruption_scenarios(self) -> List[Dict[str, Any]]:
        """Generate test scenarios for disruption prediction."""
        scenarios = [
            {
                "name": "density_limit_disruption",
                "trigger": "density",
                "evolution": self._density_limit_evolution(),
                "expected_disruption": True,
                "disruption_time": 0.8,  # seconds
            },
            {
                "name": "beta_limit_disruption", 
                "trigger": "beta",
                "evolution": self._beta_limit_evolution(),
                "expected_disruption": True,
                "disruption_time": 1.2,
            },
            {
                "name": "tearing_mode_disruption",
                "trigger": "magnetic",
                "evolution": self._tearing_mode_evolution(),
                "expected_disruption": True,
                "disruption_time": 2.5,
            },
            {
                "name": "stable_operation",
                "trigger": None,
                "evolution": self._stable_evolution(),
                "expected_disruption": False,
                "disruption_time": None,
            },
        ]
        
        return scenarios
    
    def _density_limit_evolution(self) -> Dict[str, np.ndarray]:
        """Generate density limit disruption evolution."""
        time = np.linspace(0, 2.0, 200)
        density = 4e19 * (1 + 2 * time)  # Rapid density rise
        q_min = 2.0 - 0.5 * time         # Decreasing q
        beta = 0.03 * np.ones_like(time) # Stable beta
        
        return {"time": time, "density": density, "q_min": q_min, "beta": beta}
    
    def _beta_limit_evolution(self) -> Dict[str, np.ndarray]:
        """Generate beta limit disruption evolution."""
        time = np.linspace(0, 2.0, 200)
        density = 8e19 * np.ones_like(time)  # Stable density
        q_min = 1.8 * np.ones_like(time)     # Stable q
        beta = 0.02 + 0.025 * time           # Rising beta
        
        return {"time": time, "density": density, "q_min": q_min, "beta": beta}
    
    def _tearing_mode_evolution(self) -> Dict[str, np.ndarray]:
        """Generate tearing mode disruption evolution."""
        time = np.linspace(0, 3.0, 300)
        density = 8e19 * np.ones_like(time)  # Stable density
        beta = 0.03 * np.ones_like(time)     # Stable beta
        
        # Slowly decreasing q with oscillations (tearing mode)
        q_base = 2.0 - 0.1 * time
        q_oscillation = 0.1 * np.sin(2 * np.pi * 50 * time) * np.exp(time/3)
        q_min = q_base + q_oscillation
        
        return {"time": time, "density": density, "q_min": q_min, "beta": beta}
    
    def _stable_evolution(self) -> Dict[str, np.ndarray]:
        """Generate stable plasma evolution."""
        time = np.linspace(0, 5.0, 500)
        
        # All parameters stay within safe limits
        density = 8e19 * (1 + 0.1 * np.sin(2 * np.pi * 0.1 * time))
        q_min = 2.0 + 0.2 * np.sin(2 * np.pi * 0.05 * time)
        beta = 0.025 + 0.005 * np.sin(2 * np.pi * 0.02 * time)
        
        return {"time": time, "density": density, "q_min": q_min, "beta": beta}


class BenchmarkDatasets:
    """Standardized datasets for performance benchmarking."""
    
    @staticmethod
    def small_dataset() -> Dict[str, Any]:
        """Small dataset for quick performance tests."""
        generator = TestDataGenerator(seed=42)
        return {
            "size": "small",
            "num_episodes": 10,
            "max_steps": 100,
            "states": generator.generate_plasma_states(1000),
            "controls": generator.generate_control_trajectory(1000, 8),
        }
    
    @staticmethod
    def medium_dataset() -> Dict[str, Any]:
        """Medium dataset for standard benchmarks."""
        generator = TestDataGenerator(seed=42)
        return {
            "size": "medium", 
            "num_episodes": 100,
            "max_steps": 1000,
            "states": generator.generate_plasma_states(10000),
            "controls": generator.generate_control_trajectory(10000, 8),
        }
    
    @staticmethod
    def large_dataset() -> Dict[str, Any]:
        """Large dataset for stress testing."""
        generator = TestDataGenerator(seed=42)
        return {
            "size": "large",
            "num_episodes": 1000, 
            "max_steps": 10000,
            "states": generator.generate_plasma_states(100000),
            "controls": generator.generate_control_trajectory(100000, 8),
        }


def save_test_data(data: Dict[str, Any], filepath: Path) -> None:
    """Save test data to file."""
    np.savez_compressed(filepath, **data)


def load_test_data(filepath: Path) -> Dict[str, Any]:
    """Load test data from file."""
    with np.load(filepath) as data:
        return dict(data)


# Pre-generated test data
STANDARD_TEST_CASES = {
    "iter_baseline": {
        "major_radius": 6.2,
        "minor_radius": 2.0,
        "plasma_current": 15.0,
        "magnetic_field": 5.3,
        "expected_q_min": 1.8,
        "expected_beta": 0.025,
    },
    "sparc_baseline": {
        "major_radius": 1.85,
        "minor_radius": 0.57,
        "plasma_current": 8.7,
        "magnetic_field": 12.2,
        "expected_q_min": 2.0,
        "expected_beta": 0.04,
    },
    "nstx_baseline": {
        "major_radius": 0.93,
        "minor_radius": 0.67,
        "plasma_current": 2.0,
        "magnetic_field": 1.0,
        "expected_q_min": 1.5,
        "expected_beta": 0.06,
    },
}