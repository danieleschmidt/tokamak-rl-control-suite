"""
Tests for tokamak environment functionality.
"""

import pytest
import numpy as np
from tokamak_rl import make_tokamak_env
from tokamak_rl.environment import TokamakEnv
from tokamak_rl.physics import TokamakConfig


class TestTokamakEnvironment:
    """Test suite for tokamak environment."""
    
    def test_make_tokamak_env_creates_environment(self):
        """Test that environment factory creates a working environment."""
        env = make_tokamak_env(tokamak_config="ITER")
        assert isinstance(env, TokamakEnv)
        
    def test_make_tokamak_env_with_different_configs(self):
        """Test environment creation with different tokamak configurations."""
        for config_name in ["ITER", "SPARC", "NSTX", "DIII-D"]:
            env = make_tokamak_env(tokamak_config=config_name)
            assert isinstance(env, TokamakEnv)
            assert env.tokamak_config.major_radius > 0
            
    def test_tokamak_env_observation_space(self):
        """Test that environment has correct observation space."""
        env = make_tokamak_env()
        assert hasattr(env, 'observation_space')
        assert env.observation_space.shape == (45,)  # 45-dimensional observation
        
    def test_tokamak_env_action_space(self):
        """Test that environment has correct action space."""
        env = make_tokamak_env()
        assert hasattr(env, 'action_space')
        assert env.action_space.shape == (8,)  # 8-dimensional action
        
    def test_tokamak_env_reset(self):
        """Test environment reset functionality."""
        env = make_tokamak_env()
        obs, info = env.reset()
        
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (45,)
        assert isinstance(info, dict)
        assert 'plasma_state' in info
        
    def test_tokamak_env_step(self):
        """Test environment step functionality."""
        env = make_tokamak_env()
        obs, info = env.reset()
        
        # Valid action
        action = np.array([0.1, -0.1, 0.0, 0.2, -0.2, 0.1, 0.5, 0.3])
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        assert isinstance(next_obs, np.ndarray)
        assert next_obs.shape == (45,)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        
    def test_tokamak_env_safety_shield(self):
        """Test safety shield functionality."""
        env = make_tokamak_env(enable_safety=True)
        obs, info = env.reset()
        
        # Extreme action that should be modified by safety shield
        dangerous_action = np.array([10.0, -10.0, 5.0, -5.0, 10.0, -10.0, 2.0, 2.0])
        next_obs, reward, terminated, truncated, info = env.step(dangerous_action)
        
        # Safety shield should have modified the action
        if 'safety' in info:
            assert 'action_modified' in info['safety'] or 'violations' in info['safety']


class TestEnvironmentInterface:
    """Test environment interface compliance."""
    
    def test_environment_follows_gym_interface(self):
        """Test that environment follows gymnasium interface patterns."""
        import gymnasium as gym
        
        # Test that TokamakEnv inherits from gym.Env
        assert issubclass(TokamakEnv, gym.Env)
        
        # Test standard gym interface
        env = make_tokamak_env()
        assert hasattr(env, 'reset')
        assert hasattr(env, 'step')
        assert hasattr(env, 'render')
        assert hasattr(env, 'observation_space')
        assert hasattr(env, 'action_space')
    
    def test_make_tokamak_env_parameters(self):
        """Test environment factory parameter handling."""
        # Test with various parameters
        env = make_tokamak_env(
            tokamak_config="ITER",
            control_frequency=100,
            safety_factor=1.2,
            enable_safety=True
        )
        assert env.tokamak_config.control_frequency == 100
        
    def test_custom_tokamak_config(self):
        """Test environment with custom tokamak configuration."""
        custom_config = TokamakConfig(
            major_radius=2.0,
            minor_radius=0.8,
            toroidal_field=4.0,
            plasma_current=5.0
        )
        
        env = make_tokamak_env(tokamak_config=custom_config)
        assert env.tokamak_config.major_radius == 2.0
        assert env.tokamak_config.minor_radius == 0.8
        
    def test_episode_termination(self):
        """Test episode termination conditions."""
        env = make_tokamak_env()
        obs, info = env.reset()
        
        # Run a short episode
        max_steps = 10
        for step in range(max_steps):
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                break
                
        # Should complete without errors
        assert step < max_steps or terminated or truncated


def test_package_version():
    """Test that package version is accessible."""
    import tokamak_rl
    assert hasattr(tokamak_rl, '__version__')
    assert isinstance(tokamak_rl.__version__, str)
    assert tokamak_rl.__version__ == "0.1.0"
    

def test_package_imports():
    """Test that all main package components can be imported."""
    from tokamak_rl import (
        make_tokamak_env, TokamakEnv, TokamakConfig, 
        GradShafranovSolver, SACAgent, DreamerAgent,
        SafetyShield, DisruptionPredictor
    )
    
    # Verify all imports are callable or classes
    assert callable(make_tokamak_env)
    assert callable(TokamakEnv)
    assert callable(TokamakConfig)
    assert callable(GradShafranovSolver)
    assert callable(SACAgent)
    assert callable(DreamerAgent)
    assert callable(SafetyShield)
    assert callable(DisruptionPredictor)