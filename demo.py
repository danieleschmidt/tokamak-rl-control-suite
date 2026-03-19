#!/usr/bin/env python3
"""
Tokamak RL Control Suite — Demo

Trains a PPO agent for 100 episodes on the TokamakEnv and measures
disruption rate reduction over time.

Usage:
    python demo.py
"""

import sys
import numpy as np

sys.path.insert(0, "src")

from tokamak_rl import TokamakEnv, PPOAgent, DisruptionPredictor


def measure_disruption_rate(agent, n_eval: int = 20, deterministic: bool = True) -> float:
    """Evaluate agent's disruption rate over n_eval episodes."""
    import torch
    disruptions = 0
    for _ in range(n_eval):
        obs = agent.env.reset()
        done = False
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            action, _, _ = agent.policy.act(obs_t, deterministic=deterministic)
            obs, _, done, info = agent.env.step(action.squeeze(0).detach().numpy())
        if info.get("disrupted", False):
            disruptions += 1
    return disruptions / n_eval


def main():
    print("=" * 60)
    print("  Tokamak RL Control Suite — PPO Training Demo")
    print("=" * 60)
    print()

    # Initialize environment and agent
    env = TokamakEnv(max_steps=200, seed=42)
    agent = PPOAgent(env, lr_actor=3e-4, lr_critic=1e-3, n_epochs=4, device="cpu")

    print("Environment:")
    print(f"  Observation dim: {env.obs_dim} (Ip, Te, ne, beta, q)")
    print(f"  Action dim:      {env.act_dim} (heating, gas, B-trim)")
    print(f"  Max steps/ep:    {env.max_steps}")
    print()

    print("Baseline (random policy):")
    # Random baseline before any training
    baseline_disruptions = 0
    for _ in range(20):
        obs = env.reset()
        done = False
        while not done:
            action = np.random.uniform(-1.0, 1.0, size=env.act_dim).astype(np.float32)
            obs, _, done, info = env.step(action)
        if info.get("disrupted", False):
            baseline_disruptions += 1
    baseline_rate = baseline_disruptions / 20
    print(f"  Disruption rate: {baseline_rate:.2f} ({baseline_disruptions}/20 episodes)")
    print()

    # Train PPO
    print("Training PPO agent (100 episodes):")
    print("-" * 60)
    stats = agent.train(n_episodes=100, log_interval=10, verbose=True)
    print("-" * 60)
    print()

    # Post-training evaluation
    print("Post-training evaluation (20 episodes, deterministic policy):")
    trained_rate = measure_disruption_rate(agent, n_eval=20, deterministic=True)
    print(f"  Disruption rate: {trained_rate:.2f} ({int(trained_rate*20)}/20 episodes)")
    print()

    # Summary statistics
    first_10  = [s for s in stats if s["episode"] <= 10]
    last_10   = [s for s in stats if s["episode"] > 90]
    avg_reward_first = np.mean([s["episode_reward"] for s in first_10])
    avg_reward_last  = np.mean([s["episode_reward"] for s in last_10])
    dr_first = np.mean([s["disrupted"] for s in first_10])
    dr_last  = np.mean([s["disrupted"] for s in last_10])

    print("Training summary:")
    print(f"  Episodes 1-10:  avg_reward={avg_reward_first:7.3f}  disruption_rate={dr_first:.2f}")
    print(f"  Episodes 91-100: avg_reward={avg_reward_last:7.3f}  disruption_rate={dr_last:.2f}")
    improvement = (dr_first - dr_last) / max(dr_first, 1e-6) * 100
    print(f"  Disruption rate improvement: {improvement:.1f}%")
    print()

    # DisruptionPredictor demo
    print("DisruptionPredictor demo:")
    predictor = DisruptionPredictor(risk_threshold=0.5)

    nominal_state = np.array([2.0, 10.0, 1.0, 0.05, 1.8], dtype=np.float32)
    risky_state   = np.array([2.0, 12.0, 2.2, 0.095, 1.05], dtype=np.float32)
    safe_state    = np.array([1.8, 8.0, 0.8, 0.03, 2.2], dtype=np.float32)

    for name, state in [("Nominal", nominal_state), ("Risky", risky_state), ("Safe", safe_state)]:
        label, score = predictor.predict(state)
        status = "PRE-DISRUPTION" if label else "SAFE"
        print(f"  {name:8s}: risk={score:.3f}  prediction={status}")

    print()
    print("Done. ✓")


if __name__ == "__main__":
    main()
