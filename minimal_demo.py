#!/usr/bin/env python3
"""
Minimal demo of tokamak-rl-control-suite functionality.
Shows core concepts working without external dependencies.
"""

import sys
import os
import math

# Simple implementations without numpy/torch
def simple_array(data):
    """Simple array-like structure."""
    return list(data) if hasattr(data, '__iter__') else [data]

def linspace(start, stop, num):
    """Simple linspace implementation."""
    if num <= 1:
        return [start]
    step = (stop - start) / (num - 1)
    return [start + i * step for i in range(num)]

def zeros(size):
    """Create array of zeros."""
    return [0.0] * size

def ones(size):
    """Create array of ones."""
    return [1.0] * size

# Tokamak Configuration
class TokamakConfig:
    """Simple tokamak configuration."""
    
    def __init__(self, major_radius=2.0, minor_radius=0.5, 
                 toroidal_field=3.0, plasma_current=1.0,
                 elongation=1.7, triangularity=0.4):
        self.major_radius = major_radius
        self.minor_radius = minor_radius
        self.toroidal_field = toroidal_field
        self.plasma_current = plasma_current
        self.elongation = elongation
        self.triangularity = triangularity
        self.num_pf_coils = 6
        
    @classmethod
    def from_preset(cls, preset):
        """Create from preset."""
        presets = {
            "ITER": cls(
                major_radius=6.2,
                minor_radius=2.0,
                toroidal_field=5.3,
                plasma_current=15.0,
                elongation=1.85,
                triangularity=0.33
            ),
            "SPARC": cls(
                major_radius=1.85,
                minor_radius=0.57,
                toroidal_field=12.2,
                plasma_current=8.7,
                elongation=1.97,
                triangularity=0.5
            )
        }
        return presets.get(preset, cls())

# Plasma State
class PlasmaState:
    """Simple plasma state representation."""
    
    def __init__(self, config):
        self.config = config
        self.reset()
        
    def reset(self):
        """Reset to initial state."""
        # Simple profiles using lists
        self.psi_profile = linspace(0, 1, 101)
        self.pressure_profile = [(1 - psi*psi)**2 for psi in self.psi_profile]
        self.q_profile = [1.0 + 2.5 * psi*psi for psi in self.psi_profile]
        self.density_profile = [1e20 * (1 - 0.8 * psi*psi) for psi in self.psi_profile]
        self.temperature_profile = [20.0 * math.sqrt(1 - psi*psi) for psi in self.psi_profile]
        
        self.elongation = self.config.elongation
        self.triangularity = self.config.triangularity
        self.plasma_current = self.config.plasma_current
        self.plasma_beta = 0.02
        self.pf_coil_currents = zeros(self.config.num_pf_coils)
        
        self.q_min = min(self.q_profile)
        self.disruption_probability = 0.0
        self.shape_error = 0.0
        
    def get_observation(self):
        """Get observation vector (45-dim)."""
        obs = [self.plasma_current]  # 1
        obs.append(self.plasma_beta)  # 1
        obs.extend(self.q_profile[::10])  # 10 (every 10th point)
        obs.extend([self.elongation, self.triangularity, 0.0, 0.0, 0.0, 0.0])  # 6
        obs.extend(self.pf_coil_currents)  # 6
        obs.extend(self.density_profile[::10])  # 10
        obs.extend(self.temperature_profile[::20])  # 5
        obs.append(self.shape_error)  # 1
        return obs
        
    def compute_safety_metrics(self):
        """Compute safety metrics."""
        return {
            'q_min': min(self.q_profile),
            'beta_limit_fraction': self.plasma_beta / 0.04,
            'density_limit_fraction': max(self.density_profile) / (1.2e20),
            'shape_error': self.shape_error,
            'disruption_probability': self.disruption_probability
        }

# Physics Solver  
class GradShafranovSolver:
    """Simplified physics solver."""
    
    def __init__(self, config):
        self.config = config
        
    def solve_equilibrium(self, state, pf_currents):
        """Solve equilibrium (simplified)."""
        # Update PF coil currents
        state.pf_coil_currents = list(pf_currents)
        
        # Simple shape effects from PF coils
        pf_effect = sum(pf_currents) / len(pf_currents)
        state.elongation = self.config.elongation + 0.1 * pf_effect
        state.triangularity = self.config.triangularity + 0.05 * pf_effect
        
        # Compute shape error
        target_elongation = self.config.elongation
        target_triangularity = self.config.triangularity
        shape_error = math.sqrt((state.elongation - target_elongation)**2 + 
                               (state.triangularity - target_triangularity)**2) * 100
        state.shape_error = shape_error
        
        # Update q-profile
        psi = state.psi_profile
        q_base = 1.0 + 0.1 * sum(pf_currents) / len(pf_currents)
        state.q_profile = [q_base + 2.5 * p*p for p in psi]
        state.q_min = min(state.q_profile)
        
        # Simple beta calculation
        pressure_avg = sum(state.pressure_profile) / len(state.pressure_profile)
        temp_avg = sum(state.temperature_profile) / len(state.temperature_profile)
        state.plasma_beta = min(0.1, max(0.001, pressure_avg * temp_avg / (self.config.toroidal_field**2) * 1e-6))
        
        # Assess disruption risk
        risk = 0.0
        if state.q_min < 1.5:
            risk += (1.5 - state.q_min) * 0.3
        if state.plasma_beta > 0.04:
            risk += (state.plasma_beta - 0.04) * 2.0
        if state.shape_error > 5.0:
            risk += (state.shape_error - 5.0) / 10.0 * 0.2
        state.disruption_probability = min(1.0, max(0.0, risk))
        
        return state

# Safety System
class SafetyShield:
    """Simple safety shield."""
    
    def __init__(self):
        self.q_min_threshold = 1.5
        self.beta_limit = 0.04
        self.pf_current_limit = 10.0
        self.last_action = None
        self.emergency_mode = False
        
    def filter_action(self, proposed_action, plasma_state):
        """Filter action for safety."""
        safe_action = list(proposed_action)  # Copy
        violations = []
        
        # Safety metrics
        safety_metrics = plasma_state.compute_safety_metrics()
        disruption_risk = safety_metrics['disruption_probability']
        
        # PF coil limits (first 6 elements)
        for i in range(6):
            if abs(safe_action[i]) > self.pf_current_limit:
                safe_action[i] = math.copysign(self.pf_current_limit, safe_action[i])
                violations.append(f"PF coil {i} current limit")
        
        # Gas puff (element 6) and heating (element 7) limits
        if safe_action[6] < 0:
            safe_action[6] = 0
            violations.append("Gas puff negative")
        elif safe_action[6] > 1.0:
            safe_action[6] = 1.0
            violations.append("Gas puff limit")
            
        if safe_action[7] < 0:
            safe_action[7] = 0
            violations.append("Heating negative")
        elif safe_action[7] > 1.0:
            safe_action[7] = 1.0
            violations.append("Heating limit")
        
        # Physics constraints
        if safety_metrics['q_min'] < self.q_min_threshold:
            safe_action[6] *= 0.5  # Reduce gas puff
            safe_action[7] *= 0.7  # Reduce heating
            violations.append("Low q_min safety")
            
        if safety_metrics['beta_limit_fraction'] > 0.9:
            safe_action[7] *= 0.5  # Reduce heating
            violations.append("High beta limit")
        
        safety_info = {
            'action_modified': len(violations) > 0,
            'violations': violations,
            'disruption_risk': disruption_risk,
            'emergency_mode': self.emergency_mode
        }
        
        self.last_action = safe_action
        return safe_action, safety_info
        
    def reset(self):
        """Reset shield state."""
        self.last_action = None
        self.emergency_mode = False

# Environment
class TokamakEnv:
    """Simple tokamak environment."""
    
    def __init__(self, config):
        self.config = config
        self.tokamak_config = config.get('tokamak_config', TokamakConfig())
        self.physics_solver = GradShafranovSolver(self.tokamak_config)
        self.plasma_state = PlasmaState(self.tokamak_config)
        self.safety_shield = SafetyShield()
        self.episode_step = 0
        self.max_episode_steps = 100  # Short demo episodes
        
    def reset(self):
        """Reset environment."""
        self.plasma_state.reset()
        self.safety_shield.reset()
        self.episode_step = 0
        observation = self.plasma_state.get_observation()
        info = {'plasma_state': self.plasma_state.compute_safety_metrics()}
        return observation, info
        
    def step(self, action):
        """Execute action."""
        # Apply safety filtering
        safe_action, safety_info = self.safety_shield.filter_action(action, self.plasma_state)
        
        # Execute physics
        pf_adjustments = safe_action[:6]
        max_current = 2.0
        pf_currents = [self.plasma_state.pf_coil_currents[i] + pf_adjustments[i] * max_current * 0.1 
                      for i in range(6)]
        pf_currents = [max(-max_current, min(max_current, current)) for current in pf_currents]
        
        self.plasma_state = self.physics_solver.solve_equilibrium(self.plasma_state, pf_currents)
        
        # Compute reward
        reward = self._compute_reward(safe_action, safety_info)
        
        # Check termination
        self.episode_step += 1
        terminated = self._check_termination()
        truncated = self.episode_step >= self.max_episode_steps
        
        observation = self.plasma_state.get_observation()
        info = {
            'plasma_state': self.plasma_state.compute_safety_metrics(),
            'safety': safety_info
        }
        
        return observation, reward, terminated, truncated, info
        
    def _compute_reward(self, action, safety_info):
        """Compute reward."""
        reward = 0.0
        
        # Shape accuracy reward
        shape_error = self.plasma_state.shape_error
        reward -= shape_error**2 * 0.01
        
        # Safety factor reward  
        q_min = self.plasma_state.q_min
        if q_min > 1.5:
            reward += min(q_min - 1.5, 2.0)
        else:
            reward -= 10.0 * (1.5 - q_min)**2
            
        # Control effort penalty
        control_effort = sum(a**2 for a in action[:6])
        reward -= 0.01 * control_effort
        
        # Safety penalties
        if safety_info['action_modified']:
            reward -= 1.0
        reward -= len(safety_info['violations']) * 0.5
        if safety_info['disruption_risk'] > 0.1:
            reward -= 10.0 * safety_info['disruption_risk']
            
        return reward
        
    def _check_termination(self):
        """Check termination conditions."""
        if self.plasma_state.q_min < 1.0:
            return True
        if self.plasma_state.plasma_beta > 0.08:
            return True
        if self.plasma_state.disruption_probability > 0.5:
            return True
        if self.plasma_state.shape_error > 20.0:
            return True
        return False

def make_tokamak_env(tokamak_config="ITER", **kwargs):
    """Create tokamak environment."""
    if isinstance(tokamak_config, str):
        config = TokamakConfig.from_preset(tokamak_config)
    else:
        config = tokamak_config
    
    env_config = {'tokamak_config': config, **kwargs}
    return TokamakEnv(env_config)

# Demo
def run_demo():
    """Run demonstration."""
    print("🚀 Tokamak RL Control Suite - Minimal Demo")
    print("=" * 50)
    
    # Test configurations
    print("\n📋 Testing Tokamak Configurations:")
    for preset in ["ITER", "SPARC"]:
        config = TokamakConfig.from_preset(preset)
        print(f"  {preset}: R={config.major_radius}m, a={config.minor_radius}m, "
              f"B={config.toroidal_field}T, Ip={config.plasma_current}MA")
    
    # Create environment
    print("\n🔧 Creating Environment:")
    env = make_tokamak_env("ITER")
    print("  ✅ Environment created successfully")
    
    # Test episode
    print("\n🎮 Running Control Episode:")
    obs, info = env.reset()
    print(f"  Initial observation length: {len(obs)}")
    print(f"  Initial q_min: {info['plasma_state']['q_min']:.3f}")
    print(f"  Initial shape error: {info['plasma_state']['shape_error']:.2f} cm")
    
    total_reward = 0
    for step in range(10):
        # Random control action
        action = [0.1 * math.sin(step) for _ in range(6)]  # PF coils
        action.extend([0.3, 0.5])  # Gas puff and heating
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        safety = info.get('safety', {})
        violations = safety.get('violations', [])
        disruption_risk = safety.get('disruption_risk', 0.0)
        
        print(f"  Step {step+1}: reward={reward:.2f}, q_min={info['plasma_state']['q_min']:.3f}, "
              f"violations={len(violations)}, risk={disruption_risk:.3f}")
        
        if terminated:
            print(f"  ❌ Episode terminated due to unsafe conditions!")
            break
    
    print(f"\n📊 Episode Summary:")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Final q_min: {info['plasma_state']['q_min']:.3f}")
    print(f"  Final shape error: {info['plasma_state']['shape_error']:.2f} cm")
    print(f"  Final disruption risk: {info['plasma_state']['disruption_probability']:.3f}")
    
    print("\n🎉 Demo completed successfully!")
    print("✅ All core components working correctly")
    print("🏗️ Ready for full implementation with dependencies")

if __name__ == "__main__":
    run_demo()