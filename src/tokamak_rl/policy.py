"""
SafePolicyNetwork: MLP actor-critic policy with safety projection.

The safety projection clips actions in physical space before returning them
to ensure the policy cannot command states that trivially guarantee disruption:
  - Cap heating power to avoid runaway beta
  - Cap gas injection to avoid Greenwald density limit
  - Ensure magnetic field trim stays within magnet ratings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Tuple


# Action safety bounds (in normalized [-1,1] space after projection)
# Derived from physical constraints:
#   heating:  max ~8 MW → 8/5 = 1.6 → clip to 1.0 (already bounded by tanh)
#   gas:      positive injection allowed, but extraction limited (can't pump plasma out)
#   B trim:   symmetric fine trim
SAFE_ACTION_BOUNDS = {
    "heating_min": -0.8,   # Allow some power reduction but not full cutoff
    "heating_max":  0.9,   # Cap max heating to prevent beta runaway
    "gas_min":     -0.5,   # Limited density removal
    "gas_max":      0.7,   # Cap fueling before Greenwald limit
    "btrim_min":   -1.0,
    "btrim_max":    1.0,
}

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


class SafePolicyNetwork(nn.Module):
    """
    Actor-critic network for PPO with built-in safety projection.

    Architecture:
      - Shared trunk: obs → 128 → 128 → 64
      - Actor head:   64 → (mean, log_std) for each action dim
      - Critic head:  64 → scalar value

    Safety projection:
      Actions are drawn from a Gaussian, passed through tanh (SAC-style bounding),
      then soft-clipped to per-actuator safety bounds.
    """

    def __init__(self, obs_dim: int = 5, act_dim: int = 3, hidden_size: int = 128):
        super().__init__()
        self.obs_dim  = obs_dim
        self.act_dim  = act_dim

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 64),
            nn.Tanh(),
        )

        # Actor heads
        self.mean_head    = nn.Linear(64, act_dim)
        self.log_std_head = nn.Linear(64, act_dim)

        # Critic head
        self.value_head = nn.Linear(64, 1)

        # Safety bounds as buffers (not learned)
        bounds_low  = torch.tensor([
            SAFE_ACTION_BOUNDS["heating_min"],
            SAFE_ACTION_BOUNDS["gas_min"],
            SAFE_ACTION_BOUNDS["btrim_min"],
        ])
        bounds_high = torch.tensor([
            SAFE_ACTION_BOUNDS["heating_max"],
            SAFE_ACTION_BOUNDS["gas_max"],
            SAFE_ACTION_BOUNDS["btrim_max"],
        ])
        self.register_buffer("bounds_low",  bounds_low)
        self.register_buffer("bounds_high", bounds_high)

        self._init_weights()

    def _init_weights(self):
        for layer in self.trunk:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1.0)
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (action_mean_safe, value)."""
        features = self.trunk(obs)
        mean = self.mean_head(features)
        value = self.value_head(features).squeeze(-1)
        # Apply safety projection to mean
        mean_safe = self.project_safe(torch.tanh(mean))
        return mean_safe, value

    def get_dist(self, obs: torch.Tensor) -> Tuple[Normal, torch.Tensor]:
        """Returns (action_distribution, value) for PPO training."""
        features = self.trunk(obs)
        mean = torch.tanh(self.mean_head(features))
        log_std = self.log_std_head(features).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        dist = Normal(mean, std)
        value = self.value_head(features).squeeze(-1)
        return dist, value

    def project_safe(self, action: torch.Tensor) -> torch.Tensor:
        """
        Safety projection: hard-clip actions to per-actuator safety bounds.
        This ensures the policy cannot command physically dangerous actuator states.
        """
        return torch.max(
            torch.min(action, self.bounds_high),
            self.bounds_low
        )

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action for environment interaction.

        Returns:
            action:    safe clipped action in [-1,1]
            log_prob:  log probability of raw action
            value:     critic estimate
        """
        dist, value = self.get_dist(obs)
        if deterministic:
            raw_action = dist.mean
        else:
            raw_action = dist.rsample()
        log_prob = dist.log_prob(raw_action).sum(-1)
        safe_action = self.project_safe(torch.tanh(raw_action))
        return safe_action, log_prob, value
