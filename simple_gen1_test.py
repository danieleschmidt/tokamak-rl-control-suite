#!/usr/bin/env python3
"""
Simplified Generation 1 test - basic functionality only
Tests core components directly without complex imports.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Test individual modules directly
def test_physics_module():
    """Test physics module functionality."""
    print("🔬 Testing Physics Module...")
    
    from tokamak_rl.physics import TokamakConfig, PlasmaState, GradShafranovSolver
    
    # Test configuration
    config = TokamakConfig.from_preset("ITER")
    print(f"  ✓ ITER config: R={config.major_radius}m, a={config.minor_radius}m")
    
    # Test plasma state
    state = PlasmaState(config)
    obs = state.get_observation()
    print(f"  ✓ Plasma state observation shape: {len(obs)}")
    
    # Test physics solver
    solver = GradShafranovSolver(config)
    new_state = solver.solve_equilibrium(state, [0, 0, 0, 0, 0, 0])
    print(f"  ✓ Physics solver - Q-min: {new_state.q_min:.2f}")
    
    return True

def test_environment_module():
    """Test environment module functionality."""
    print("\n🌍 Testing Environment Module...")
    
    from tokamak_rl.environment import make_tokamak_env, TokamakEnv
    
    # Test environment creation
    env = make_tokamak_env("ITER", enable_safety=False)  # Disable safety for simplicity
    print(f"  ✓ Environment created: {type(env).__name__}")
    
    # Test reset
    obs, info = env.reset()
    print(f"  ✓ Environment reset - obs shape: {obs.shape}")
    print(f"  ✓ Initial Q-min: {info['plasma_state']['q_min']:.2f}")
    
    # Test step
    action = [0.1, -0.1, 0.0, 0.0, 0.0, 0.0, 0.5, 0.3]  # 8-dim action
    next_obs, reward, terminated, truncated, step_info = env.step(action)
    print(f"  ✓ Environment step - reward: {reward:.3f}")
    
    return True

def test_agents_module():
    """Test agents module functionality."""
    print("\n🤖 Testing Agents Module...")
    
    import numpy as np
    from tokamak_rl.agents import create_agent
    from tokamak_rl.environment import make_tokamak_env
    
    # Create simple environment for agent testing
    env = make_tokamak_env("ITER", enable_safety=False)
    
    # Test SAC agent creation
    agent = create_agent("SAC", env.observation_space, env.action_space)
    print(f"  ✓ SAC agent created: {type(agent).__name__}")
    
    # Test action selection
    obs, _ = env.reset()
    action = agent.act(obs, deterministic=True)
    print(f"  ✓ Agent action - shape: {action.shape}, range: [{action.min():.2f}, {action.max():.2f}]")
    
    # Test experience addition (without learning)
    next_obs, reward, done, truncated, _ = env.step(action)
    agent.add_experience(obs, action, reward, next_obs, done)
    print(f"  ✓ Experience added to replay buffer")
    
    return True

def test_safety_module():
    """Test safety module functionality."""
    print("\n🛡️ Testing Safety Module...")
    
    from tokamak_rl.safety import SafetyLimits, create_safety_system
    from tokamak_rl.physics import TokamakConfig
    
    # Test safety limits
    limits = SafetyLimits()
    print(f"  ✓ Safety limits - Q-min threshold: {limits.q_min_threshold}")
    
    # Test safety system creation
    config = TokamakConfig.from_preset("ITER")
    safety_system = create_safety_system(config)
    print(f"  ✓ Safety system created: {type(safety_system).__name__}")
    
    return True

def run_integration_test():
    """Run basic integration test."""
    print("\n🔄 Running Integration Test...")
    
    from tokamak_rl.environment import make_tokamak_env
    from tokamak_rl.agents import create_agent
    
    # Create environment with safety
    env = make_tokamak_env("ITER", safety_factor=1.2)
    agent = create_agent("SAC", env.observation_space, env.action_space)
    
    # Run a few steps
    obs, _ = env.reset()
    total_reward = 0
    
    for step in range(5):
        action = agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"  Step {step + 1}: reward={reward:.3f}, Q-min={info['plasma_state']['q_min']:.2f}")
        
        if terminated or truncated:
            break
    
    print(f"  ✓ Integration test completed - Total reward: {total_reward:.3f}")
    
    return True

if __name__ == "__main__":
    print("TOKAMAK RL CONTROL SUITE - GENERATION 1 BASIC TEST")
    print("=" * 60)
    
    tests = [
        test_physics_module,
        test_environment_module, 
        test_agents_module,
        test_safety_module,
        run_integration_test
    ]
    
    all_passed = True
    
    for test in tests:
        try:
            result = test()
            if not result:
                all_passed = False
        except Exception as e:
            print(f"❌ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    if all_passed:
        print("\n🎉 GENERATION 1 SUCCESS: Core functionality working!")
        print("Ready to proceed to Generation 2: Make it Robust")
        sys.exit(0)
    else:
        print("\n⚠️ GENERATION 1 INCOMPLETE: Some tests failed")
        sys.exit(1)