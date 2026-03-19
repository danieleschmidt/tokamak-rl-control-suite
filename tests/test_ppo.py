"""Tests for PPOAgent."""

import sys
import pytest
import numpy as np
import torch

sys.path.insert(0, "src")

from tokamak_rl.environment import TokamakEnv
from tokamak_rl.ppo import PPOAgent, RolloutBuffer


class TestRolloutBuffer:

    def test_clear_empties_buffer(self):
        buf = RolloutBuffer()
        buf.rewards.append(1.0)
        buf.dones.append(False)
        buf.clear()
        assert len(buf) == 0

    def test_len_matches_rewards(self):
        buf = RolloutBuffer()
        for i in range(5):
            buf.rewards.append(float(i))
        assert len(buf) == 5


class TestPPOAgent:

    def setup_method(self):
        self.env = TokamakEnv(max_steps=50, seed=99)
        self.agent = PPOAgent(
            self.env,
            lr_actor=3e-4,
            lr_critic=1e-3,
            n_epochs=2,
            batch_size=32,
            device="cpu",
        )

    def test_collect_rollout_fills_buffer(self):
        info = self.agent.collect_rollout()
        assert len(self.agent.buffer) > 0
        assert "episode_reward" in info
        assert "disrupted" in info

    def test_rollout_buffer_lengths_consistent(self):
        self.agent.collect_rollout()
        buf = self.agent.buffer
        n = len(buf)
        assert len(buf.observations) == n
        assert len(buf.actions)      == n
        assert len(buf.rewards)      == n
        assert len(buf.log_probs)    == n
        assert len(buf.values)       == n
        assert len(buf.dones)        == n

    def test_update_returns_loss_dict(self):
        self.agent.collect_rollout()
        update_info = self.agent.update()
        assert "loss_actor"  in update_info
        assert "loss_critic" in update_info
        assert "entropy"     in update_info
        assert np.isfinite(update_info["loss_actor"])
        assert np.isfinite(update_info["loss_critic"])

    def test_compute_advantages_correct_length(self):
        self.agent.collect_rollout()
        advantages, returns = self.agent.compute_advantages()
        n = len(self.agent.buffer)
        assert advantages.shape == (n,)
        assert returns.shape    == (n,)

    def test_train_short_run(self):
        """Train for 5 episodes, collect stats."""
        stats = self.agent.train(n_episodes=5, log_interval=5, verbose=False)
        assert len(stats) == 5
        for s in stats:
            assert "episode_reward" in s
            assert "disrupted" in s
            assert "episode" in s
            assert np.isfinite(s["episode_reward"])

    def test_policy_parameters_change_after_update(self):
        """Policy weights should change after a PPO update."""
        # Grab initial params
        initial_params = [p.clone().detach() for p in self.agent.policy.parameters()]

        # Run one rollout + update
        self.agent.collect_rollout()
        self.agent.update()

        updated_params = [p.clone().detach() for p in self.agent.policy.parameters()]

        # At least some parameters should have changed
        changed = any(
            not torch.allclose(p0, p1, atol=1e-8)
            for p0, p1 in zip(initial_params, updated_params)
        )
        assert changed, "Policy parameters did not change after update"

    def test_deterministic_action_consistent(self):
        """Deterministic policy should give the same action for the same obs."""
        obs_t = torch.randn(1, 5)
        a1, _, _ = self.agent.policy.act(obs_t, deterministic=True)
        a2, _, _ = self.agent.policy.act(obs_t, deterministic=True)
        assert torch.allclose(a1, a2)
