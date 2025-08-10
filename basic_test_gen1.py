#!/usr/bin/env python3
"""
Basic functionality test for Generation 1: Make it Work
Tests core tokamak RL environment functionality.
"""

import numpy as np
from tokamak_rl import make_tokamak_env, create_agent
import sys
import traceback

def test_basic_functionality():
    """Test basic tokamak environment and agent functionality."""
    print("🔬 GENERATION 1 TEST: Make it Work")
    print("=" * 50)
    
    try:
        # Test 1: Create environment
        print("✓ Testing environment creation...")
        env = make_tokamak_env(
            tokamak_config="ITER",
            control_frequency=100,
            safety_factor=1.2
        )
        print(f"  Environment created: {type(env).__name__}")
        print(f"  Observation space: {env.observation_space.shape}")
        print(f"  Action space: {env.action_space.shape}")
        
        # Test 2: Reset environment
        print("\n✓ Testing environment reset...")
        obs, info = env.reset()
        print(f"  Initial observation shape: {obs.shape}")
        print(f"  Q-min: {info['plasma_state']['q_min']:.2f}")
        print(f"  Shape error: {info['plasma_state']['shape_error']:.2f} cm")
        
        # Test 3: Create agent
        print("\n✓ Testing agent creation...")
        agent = create_agent("SAC", env.observation_space, env.action_space)
        print(f"  Agent created: {type(agent).__name__}")
        
        # Test 4: Agent action selection
        print("\n✓ Testing agent action selection...")
        action = agent.act(obs, deterministic=True)
        print(f"  Action shape: {action.shape}")
        print(f"  Action range: [{action.min():.2f}, {action.max():.2f}]")
        
        # Test 5: Environment step
        print("\n✓ Testing environment step...")
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        print(f"  Step completed successfully")
        print(f"  Reward: {reward:.3f}")
        print(f"  Next Q-min: {step_info['plasma_state']['q_min']:.2f}")
        print(f"  Terminated: {terminated}, Truncated: {truncated}")
        
        # Test 6: Multiple steps simulation
        print("\n✓ Testing multi-step simulation...")
        total_reward = 0
        for step in range(10):
            action = agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                print(f"  Episode ended at step {step + 1}")
                break
                
        print(f"  Total reward over {step + 1} steps: {total_reward:.3f}")
        print(f"  Final shape error: {info['plasma_state']['shape_error']:.2f} cm")
        
        # Test 7: Episode metrics
        print("\n✓ Testing episode metrics...")
        metrics = env.get_episode_metrics()
        if metrics:
            print(f"  Mean shape error: {metrics['mean_shape_error']:.2f} cm")
            print(f"  Episode length: {metrics['episode_length']} steps")
        
        print("\n🎉 GENERATION 1 SUCCESS: All basic functionality working!")
        return True
        
    except Exception as e:
        print(f"\n❌ GENERATION 1 FAILED: {str(e)}")
        print(f"Error details: {traceback.format_exc()}")
        return False

def test_different_configurations():
    """Test different tokamak configurations."""
    print("\n🔧 Testing different tokamak configurations...")
    
    configs = ["ITER", "SPARC", "NSTX", "DIII-D"]
    
    for config_name in configs:
        try:
            env = make_tokamak_env(tokamak_config=config_name)
            obs, info = env.reset()
            print(f"  ✓ {config_name}: Q-min={info['plasma_state']['q_min']:.2f}")
        except Exception as e:
            print(f"  ❌ {config_name}: {str(e)}")
            return False
    
    return True

if __name__ == "__main__":
    print("TOKAMAK RL CONTROL SUITE - GENERATION 1 VERIFICATION")
    print("=" * 60)
    
    # Run basic functionality test
    success1 = test_basic_functionality()
    
    # Run configuration test
    success2 = test_different_configurations()
    
    if success1 and success2:
        print("\n🌟 GENERATION 1 COMPLETE: Ready for Generation 2!")
        sys.exit(0)
    else:
        print("\n⚠️  GENERATION 1 INCOMPLETE: Fixes needed before proceeding")
        sys.exit(1)