"""
Physics simulation components for tokamak plasma control.

This module implements the core physics solver for tokamak plasma equilibrium
using the Grad-Shafranov equation and related plasma physics calculations.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import warnings


@dataclass
class TokamakConfig:
    """Configuration parameters for tokamak geometry and physics."""
    
    # Geometric parameters
    major_radius: float  # Major radius (m)
    minor_radius: float  # Minor radius (m)
    elongation: float = 1.7  # Plasma elongation
    triangularity: float = 0.4  # Plasma triangularity
    
    # Magnetic field parameters
    toroidal_field: float  # Toroidal field at major radius (T)
    plasma_current: float  # Plasma current (MA)
    num_pf_coils: int = 6  # Number of poloidal field coils
    
    # Physics parameters
    beta_n: float = 2.5  # Normalized beta
    q95: float = 3.5  # Safety factor at 95% flux surface
    density_limit_factor: float = 0.8  # Fraction of Greenwald limit
    
    # Control parameters
    control_frequency: float = 100.0  # Hz
    simulation_timestep: float = 0.001  # seconds
    
    @classmethod
    def from_preset(cls, preset: str) -> "TokamakConfig":
        """Create configuration from preset tokamak parameters."""
        presets = {
            "ITER": cls(
                major_radius=6.2,
                minor_radius=2.0,
                elongation=1.85,
                triangularity=0.33,
                toroidal_field=5.3,
                plasma_current=15.0,
                num_pf_coils=6,
                beta_n=1.8,
                q95=3.0
            ),
            "SPARC": cls(
                major_radius=1.85,
                minor_radius=0.57,
                elongation=1.97,
                triangularity=0.5,
                toroidal_field=12.2,
                plasma_current=8.7,
                num_pf_coils=8,
                beta_n=2.5,
                q95=3.5
            ),
            "NSTX": cls(
                major_radius=0.93,
                minor_radius=0.67,
                elongation=2.5,
                triangularity=0.7,
                toroidal_field=1.0,
                plasma_current=2.0,
                num_pf_coils=6,
                beta_n=6.0,
                q95=8.0
            ),
            "DIII-D": cls(
                major_radius=1.67,
                minor_radius=0.67,
                elongation=1.8,
                triangularity=0.4,
                toroidal_field=2.2,
                plasma_current=1.5,
                num_pf_coils=6,
                beta_n=2.8,
                q95=4.0
            )
        }
        
        if preset not in presets:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")
        
        return presets[preset]


class PlasmaState:
    """Encapsulates tokamak plasma physics state."""
    
    def __init__(self, config: TokamakConfig):
        self.config = config
        self.reset()
        
    def reset(self) -> None:
        """Reset plasma to initial equilibrium state."""
        # Initialize plasma profiles
        self.psi_profile = np.linspace(0, 1, 101)  # Normalized poloidal flux
        self.pressure_profile = self._initial_pressure_profile()
        self.q_profile = self._initial_q_profile()
        self.density_profile = self._initial_density_profile()
        self.temperature_profile = self._initial_temperature_profile()
        
        # Shape parameters
        self.elongation = self.config.elongation
        self.triangularity = self.config.triangularity
        
        # Current state
        self.plasma_current = self.config.plasma_current
        self.plasma_beta = 0.02  # Initial beta
        self.pf_coil_currents = np.zeros(self.config.num_pf_coils)
        
        # Safety metrics
        self.q_min = np.min(self.q_profile)
        self.disruption_probability = 0.0
        self.shape_error = 0.0
        
    def _initial_pressure_profile(self) -> np.ndarray:
        """Initialize realistic pressure profile."""
        psi = self.psi_profile
        # Parabolic pressure profile: P(psi) = P0 * (1 - psi^2)^2
        return 1.0 * (1 - psi**2)**2
        
    def _initial_q_profile(self) -> np.ndarray:
        """Initialize safety factor profile."""
        psi = self.psi_profile
        # Monotonic q profile: q(psi) = q0 + (q95 - q0) * psi^2
        q0 = 1.0
        q95 = self.config.q95
        return q0 + (q95 - q0) * psi**2
        
    def _initial_density_profile(self) -> np.ndarray:
        """Initialize electron density profile."""
        psi = self.psi_profile
        # Peaked density profile
        n0 = 1.0e20  # Central density (m^-3)
        return n0 * (1 - 0.8 * psi**2)
        
    def _initial_temperature_profile(self) -> np.ndarray:
        """Initialize temperature profile."""
        psi = self.psi_profile
        # Peaked temperature profile
        T0 = 20.0  # Central temperature (keV)
        return T0 * (1 - psi**2)**0.5
        
    def get_observation(self) -> np.ndarray:
        """Get current state as observation vector (45-dim)."""
        obs = np.concatenate([
            [self.plasma_current],  # 1
            [self.plasma_beta],  # 1
            self.q_profile[::10],  # 10 (every 10th point)
            [self.elongation, self.triangularity, 0.0, 0.0, 0.0, 0.0],  # 6 shape params
            self.pf_coil_currents,  # 6 PF coil currents  
            self.density_profile[::10],  # 10 (every 10th point)
            self.temperature_profile[::20],  # 5 (every 20th point)
            [self.shape_error]  # 1
        ])
        return obs.astype(np.float32)
        
    def compute_safety_metrics(self) -> Dict[str, float]:
        """Compute disruption-related safety metrics."""
        metrics = {
            'q_min': np.min(self.q_profile),
            'beta_limit_fraction': self.plasma_beta / 0.04,  # Troyon limit
            'density_limit_fraction': np.max(self.density_profile) / (1.2e20),
            'shape_error': self.shape_error,
            'disruption_probability': self.disruption_probability
        }
        return metrics


class GradShafranovSolver:
    """Grad-Shafranov equilibrium solver for tokamak plasma."""
    
    def __init__(self, config: TokamakConfig):
        self.config = config
        self._setup_grid()
        
    def _setup_grid(self) -> None:
        """Setup computational grid for equilibrium solving."""
        # Simplified rectangular grid
        self.nr = 65
        self.nz = 65
        
        r_min = self.config.major_radius - 1.5 * self.config.minor_radius
        r_max = self.config.major_radius + 1.5 * self.config.minor_radius
        z_min = -1.5 * self.config.minor_radius * self.config.elongation
        z_max = 1.5 * self.config.minor_radius * self.config.elongation
        
        self.r_grid = np.linspace(r_min, r_max, self.nr)
        self.z_grid = np.linspace(z_min, z_max, self.nz)
        self.R, self.Z = np.meshgrid(self.r_grid, self.z_grid)
        
    def solve_equilibrium(self, state: PlasmaState, 
                         pf_currents: np.ndarray) -> PlasmaState:
        """
        Solve Grad-Shafranov equation for new equilibrium.
        
        Simplified implementation using analytical approximations.
        Production version would use finite element methods.
        """
        # Update PF coil currents
        state.pf_coil_currents = pf_currents.copy()
        
        # Compute shape parameters from PF coil configuration
        # This is a simplified model - real implementation would solve PDE
        kappa_target = self.config.elongation
        delta_target = self.config.triangularity
        
        # PF coil feedback on shape (simplified)
        pf_effect = np.sum(pf_currents) / self.config.num_pf_coils
        state.elongation = kappa_target + 0.1 * pf_effect
        state.triangularity = delta_target + 0.05 * pf_effect
        
        # Update shape error based on deviation from target
        target_shape = np.array([kappa_target, delta_target])
        current_shape = np.array([state.elongation, state.triangularity])
        state.shape_error = np.linalg.norm(target_shape - current_shape) * 100  # cm
        
        # Update q-profile based on current profile (simplified)
        # Real solver would compute this from flux function
        state.q_profile = self._compute_q_profile(state)
        state.q_min = np.min(state.q_profile)
        
        # Update beta based on pressure and magnetic field
        state.plasma_beta = self._compute_beta(state)
        
        # Compute disruption probability
        state.disruption_probability = self._assess_disruption_risk(state)
        
        return state
        
    def _compute_q_profile(self, state: PlasmaState) -> np.ndarray:
        """Compute safety factor profile."""
        psi = state.psi_profile
        
        # Include effect of PF coils on q-profile
        pf_effect = np.mean(state.pf_coil_currents) * 0.1
        q_base = 1.0 + pf_effect
        q_edge = self.config.q95
        
        # Ensure monotonic profile with realistic curvature
        q_profile = q_base + (q_edge - q_base) * psi**2
        
        # Apply current profile effects
        current_peaking = 2.0 - 0.5 * np.abs(np.mean(state.pf_coil_currents))
        q_profile *= (1 + 0.2 * current_peaking * (1 - psi))
        
        return np.maximum(q_profile, 0.5)  # Ensure physical values
        
    def _compute_beta(self, state: PlasmaState) -> float:
        """Compute plasma beta from profiles."""
        # Volume-averaged beta from pressure and temperature
        pressure_avg = np.trapz(state.pressure_profile, state.psi_profile)
        temp_avg = np.trapz(state.temperature_profile, state.psi_profile)
        
        # Simplified beta calculation
        beta = pressure_avg * temp_avg / (self.config.toroidal_field**2) * 1e-6
        return np.clip(beta, 0.001, 0.1)
        
    def _assess_disruption_risk(self, state: PlasmaState) -> float:
        """Assess disruption probability based on plasma parameters."""
        risk_factors = []
        
        # Low q-min increases disruption risk
        if state.q_min < 1.5:
            risk_factors.append((1.5 - state.q_min) * 0.3)
            
        # High beta increases risk
        beta_limit = 0.04
        if state.plasma_beta > beta_limit:
            risk_factors.append((state.plasma_beta - beta_limit) * 2.0)
            
        # High density increases risk
        density_max = np.max(state.density_profile)
        density_limit = 1.2e20
        if density_max > density_limit:
            risk_factors.append((density_max - density_limit) / density_limit * 0.5)
            
        # Large shape errors increase risk
        if state.shape_error > 5.0:
            risk_factors.append((state.shape_error - 5.0) / 10.0 * 0.2)
            
        # Combine risk factors
        total_risk = sum(risk_factors) if risk_factors else 0.0
        return np.clip(total_risk, 0.0, 1.0)


class ShapeAnalyzer:
    """Analyzes plasma shape and boundary characteristics."""
    
    def __init__(self, config: TokamakConfig):
        self.config = config
        
    def compute_shape_parameters(self, flux_surfaces: np.ndarray) -> Dict[str, float]:
        """Compute shape parameters from flux surfaces."""
        # In a real implementation, this would analyze the last closed flux surface
        # For now, return simplified parameters
        
        return {
            'elongation': self.config.elongation,
            'triangularity': self.config.triangularity,
            'squareness': 0.0,
            'volume': np.pi**2 * self.config.major_radius * self.config.minor_radius**2,
            'surface_area': 4 * np.pi**2 * self.config.major_radius * self.config.minor_radius,
            'cross_sectional_area': np.pi * self.config.minor_radius**2 * self.config.elongation
        }
        
    def compute_shape_error(self, current_shape: Dict[str, float], 
                          target_shape: Dict[str, float]) -> float:
        """Compute RMS shape error in cm."""
        errors = []
        for param in ['elongation', 'triangularity']:
            if param in current_shape and param in target_shape:
                error = (current_shape[param] - target_shape[param]) * 100  # Convert to cm
                errors.append(error**2)
                
        return np.sqrt(np.mean(errors)) if errors else 0.0


def create_physics_solver(config: TokamakConfig) -> GradShafranovSolver:
    """Factory function to create physics solver."""
    return GradShafranovSolver(config)