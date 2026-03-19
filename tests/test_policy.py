"""Tests for SafePolicyNetwork."""

import sys
import pytest
import numpy as np
import torch

sys.path.insert(0, "src")

from tokamak_rl.policy import SafePolicyNetwork, SAFE_ACTION_BOUNDS


class TestSafePolicyNetwork:

    def setup_method(self):
        self.policy = SafePolicyNetwork(obs_dim=5, act_dim=3)

    def test_forward_output_shape(self):
        obs = torch.randn(4, 5)  # batch=4
        mean, value = self.policy(obs)
        assert mean.shape  == (4, 3)
        assert value.shape == (4,)

    def test_action_within_safe_bounds(self):
        """Output actions must stay within safe bounds."""
        obs = torch.randn(100, 5)
        mean, _ = self.policy(obs)
        mean_np = mean.detach().numpy()
        assert np.all(mean_np[:, 0] >= SAFE_ACTION_BOUNDS["heating_min"] - 1e-5)
        assert np.all(mean_np[:, 0] <= SAFE_ACTION_BOUNDS["heating_max"] + 1e-5)
        assert np.all(mean_np[:, 1] >= SAFE_ACTION_BOUNDS["gas_min"]     - 1e-5)
        assert np.all(mean_np[:, 1] <= SAFE_ACTION_BOUNDS["gas_max"]     + 1e-5)
        assert np.all(mean_np[:, 2] >= SAFE_ACTION_BOUNDS["btrim_min"]   - 1e-5)
        assert np.all(mean_np[:, 2] <= SAFE_ACTION_BOUNDS["btrim_max"]   + 1e-5)

    def test_act_returns_correct_shapes(self):
        obs = torch.randn(1, 5)
        action, log_prob, value = self.policy.act(obs)
        assert action.shape   == (1, 3)
        assert log_prob.shape == (1,)
        assert value.shape    == (1,)

    def test_act_deterministic(self):
        """Deterministic mode should produce identical results."""
        obs = torch.randn(1, 5)
        a1, _, _ = self.policy.act(obs, deterministic=True)
        a2, _, _ = self.policy.act(obs, deterministic=True)
        assert torch.allclose(a1, a2)

    def test_get_dist_returns_distribution(self):
        obs = torch.randn(8, 5)
        dist, value = self.policy.get_dist(obs)
        assert dist.batch_shape == (8, 3)
        assert value.shape      == (8,)

    def test_safety_projection_clips_correctly(self):
        """Project safe should clip to bounds."""
        policy = SafePolicyNetwork(obs_dim=5, act_dim=3)
        # Test a deliberately out-of-bounds action
        action = torch.tensor([[2.0, 2.0, 2.0]])  # All exceed bounds
        projected = policy.project_safe(action)
        assert float(projected[0, 0]) <= SAFE_ACTION_BOUNDS["heating_max"] + 1e-5
        assert float(projected[0, 1]) <= SAFE_ACTION_BOUNDS["gas_max"]     + 1e-5

    def test_gradient_flows(self):
        """Should be able to backprop through the policy."""
        obs = torch.randn(4, 5, requires_grad=False)
        dist, value = self.policy.get_dist(obs)
        loss = -dist.log_prob(dist.mean).sum()
        loss.backward()
        # If no error, gradient flow is intact

    def test_parameter_count_reasonable(self):
        """Network should be appropriately sized — not bloated."""
        n_params = sum(p.numel() for p in self.policy.parameters())
        assert n_params < 100_000, f"Network too large: {n_params} params"
        assert n_params > 1_000,   f"Network too small: {n_params} params"
