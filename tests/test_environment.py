"""
Tests for tokamak environment functionality.
"""

import pytest
from tokamak_rl import make_tokamak_env
from tokamak_rl.environment import TokamakEnv


class TestTokamakEnvironment:
    """Test suite for tokamak environment."""
    
    def test_make_tokamak_env_imports(self):
        """Test that environment factory can be imported."""
        assert callable(make_tokamak_env)
    
    def test_make_tokamak_env_not_implemented(self):
        """Test that environment factory raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            make_tokamak_env()
    
    def test_tokamak_env_class_exists(self):
        """Test that TokamakEnv class can be instantiated."""
        config = {"test": True}
        env = TokamakEnv(config)
        assert env.config == config
    
    def test_tokamak_env_methods_not_implemented(self):
        """Test that core environment methods raise NotImplementedError."""
        env = TokamakEnv({})
        
        with pytest.raises(NotImplementedError):
            env.reset()
            
        with pytest.raises(NotImplementedError):
            env.step([0.0])
            
        with pytest.raises(NotImplementedError):
            env.render()


class TestEnvironmentInterface:
    """Test environment interface compliance."""
    
    def test_environment_follows_gym_interface(self):
        """Test that environment follows gymnasium interface patterns."""
        import gymnasium as gym
        
        # Test that TokamakEnv inherits from gym.Env
        assert issubclass(TokamakEnv, gym.Env)
    
    def test_make_tokamak_env_parameters(self):
        """Test environment factory parameter handling."""
        # Should not raise exception for parameter validation
        try:
            make_tokamak_env(
                tokamak_config="ITER",
                control_frequency=100,
                safety_factor=1.2
            )
        except NotImplementedError:
            # Expected - implementation pending
            pass
        except Exception as e:
            pytest.fail(f"Parameter validation failed: {e}")


def test_package_version():
    """Test that package version is accessible."""
    import tokamak_rl
    assert hasattr(tokamak_rl, '__version__')
    assert isinstance(tokamak_rl.__version__, str)