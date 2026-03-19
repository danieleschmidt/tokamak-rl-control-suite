"""
PPOAgent: Simplified Proximal Policy Optimization for tokamak control.

Key choices:
  - Clipped surrogate objective (ε = 0.2)
  - Generalized Advantage Estimation (λ = 0.95)
  - Entropy bonus (encourages exploration in high-dimensional plasma space)
  - Separate actor/critic learning rates
  - Mini-batch updates from rollout buffer

Reference: Schulman et al. (2017) "Proximal Policy Optimization Algorithms"
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field

from tokamak_rl.policy import SafePolicyNetwork
from tokamak_rl.environment import TokamakEnv


@dataclass
class RolloutBuffer:
    """Stores a single episode/rollout for PPO update."""
    observations: List[np.ndarray] = field(default_factory=list)
    actions:      List[np.ndarray] = field(default_factory=list)
    rewards:      List[float]      = field(default_factory=list)
    log_probs:    List[float]      = field(default_factory=list)
    values:       List[float]      = field(default_factory=list)
    dones:        List[bool]       = field(default_factory=list)

    def clear(self):
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()

    def __len__(self):
        return len(self.rewards)


class PPOAgent:
    """
    Proximal Policy Optimization agent for tokamak plasma control.

    Trains a SafePolicyNetwork using on-policy rollouts.
    """

    def __init__(
        self,
        env: TokamakEnv,
        lr_actor:  float = 3e-4,
        lr_critic: float = 1e-3,
        gamma:     float = 0.99,
        lam:       float = 0.95,      # GAE lambda
        clip_eps:  float = 0.2,       # PPO clip ratio
        entropy_coef: float = 0.01,
        vf_coef:   float = 0.5,
        n_epochs:  int   = 4,         # update epochs per rollout
        batch_size: int  = 64,
        device:    str   = "cpu",
    ):
        self.env = env
        self.gamma     = gamma
        self.lam       = lam
        self.clip_eps  = clip_eps
        self.entropy_coef = entropy_coef
        self.vf_coef   = vf_coef
        self.n_epochs  = n_epochs
        self.batch_size = batch_size
        self.device    = torch.device(device)

        self.policy = SafePolicyNetwork(
            obs_dim=env.obs_dim,
            act_dim=env.act_dim,
        ).to(self.device)

        # Separate LRs for actor vs critic is common in tokamak control
        actor_params  = list(self.policy.trunk.parameters()) + \
                        list(self.policy.mean_head.parameters()) + \
                        list(self.policy.log_std_head.parameters())
        critic_params = list(self.policy.value_head.parameters())

        self.optimizer = optim.Adam([
            {"params": actor_params,  "lr": lr_actor},
            {"params": critic_params, "lr": lr_critic},
        ])

        self.buffer = RolloutBuffer()
        self.training_stats: List[Dict] = []

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def collect_rollout(self) -> Dict:
        """Run one episode and store experience in buffer."""
        self.buffer.clear()
        obs = self.env.reset()
        episode_reward = 0.0
        disrupted = False
        n_steps = 0

        while True:
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action_t, log_prob_t, value_t = self.policy.act(obs_t)

            action = action_t.squeeze(0).cpu().numpy()
            next_obs, reward, done, info = self.env.step(action)

            self.buffer.observations.append(obs.copy())
            self.buffer.actions.append(action.copy())
            self.buffer.rewards.append(reward)
            self.buffer.log_probs.append(float(log_prob_t.item()))
            self.buffer.values.append(float(value_t.item()))
            self.buffer.dones.append(done)

            episode_reward += reward
            n_steps += 1
            obs = next_obs

            if done:
                disrupted = info.get("disrupted", False)
                break

        return {
            "episode_reward": episode_reward,
            "episode_length": n_steps,
            "disrupted": disrupted,
            "disruption_risk": info.get("disruption_risk", 0.0),
        }

    # ------------------------------------------------------------------
    # GAE advantage estimation
    # ------------------------------------------------------------------

    def compute_advantages(self, last_value: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and discounted returns."""
        rewards  = self.buffer.rewards
        values   = self.buffer.values + [last_value]
        dones    = self.buffer.dones

        advantages = []
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t+1] * (1 - dones[t]) - values[t]
            gae   = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        advantages_t = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        returns_t    = advantages_t + torch.tensor(self.buffer.values, dtype=torch.float32).to(self.device)

        # Normalize advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        return advantages_t, returns_t

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def update(self) -> Dict:
        """Perform PPO update on collected rollout."""
        advantages, returns = self.compute_advantages()

        obs_t      = torch.tensor(np.array(self.buffer.observations), dtype=torch.float32).to(self.device)
        actions_t  = torch.tensor(np.array(self.buffer.actions),      dtype=torch.float32).to(self.device)
        old_lps_t  = torch.tensor(self.buffer.log_probs,              dtype=torch.float32).to(self.device)

        n = len(self.buffer)
        total_loss_actor  = 0.0
        total_loss_critic = 0.0
        total_entropy     = 0.0
        n_updates = 0

        for _ in range(self.n_epochs):
            # Mini-batch sampling
            indices = torch.randperm(n)
            for start in range(0, n, self.batch_size):
                idx = indices[start:start + self.batch_size]
                if len(idx) < 4:
                    continue

                obs_b    = obs_t[idx]
                act_b    = actions_t[idx]
                adv_b    = advantages[idx]
                ret_b    = returns[idx]
                old_lp_b = old_lps_t[idx]

                # Re-evaluate actions under current policy
                dist, values_b = self.policy.get_dist(obs_b)
                new_lp_b = dist.log_prob(act_b).sum(-1)
                entropy  = dist.entropy().sum(-1).mean()

                # Ratio & clipped surrogate objective
                ratio       = torch.exp(new_lp_b - old_lp_b)
                surr1       = ratio * adv_b
                surr2       = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_b
                loss_actor  = -torch.min(surr1, surr2).mean()

                # Critic loss
                loss_critic = nn.functional.mse_loss(values_b, ret_b)

                # Total loss
                loss = loss_actor + self.vf_coef * loss_critic - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()

                total_loss_actor  += loss_actor.item()
                total_loss_critic += loss_critic.item()
                total_entropy     += entropy.item()
                n_updates += 1

        n_u = max(n_updates, 1)
        return {
            "loss_actor":  total_loss_actor  / n_u,
            "loss_critic": total_loss_critic / n_u,
            "entropy":     total_entropy     / n_u,
        }

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(
        self,
        n_episodes: int = 100,
        log_interval: int = 10,
        verbose: bool = True,
    ) -> List[Dict]:
        """
        Train PPO agent for n_episodes.

        Returns list of per-episode stats.
        """
        all_stats = []
        disruption_window = []  # rolling window for disruption rate

        for ep in range(1, n_episodes + 1):
            # Collect rollout
            ep_info = self.collect_rollout()

            # PPO update
            update_info = self.update()

            stats = {**ep_info, **update_info, "episode": ep}
            all_stats.append(stats)
            disruption_window.append(int(ep_info["disrupted"]))
            if len(disruption_window) > 20:
                disruption_window.pop(0)

            if verbose and ep % log_interval == 0:
                dr_rate = sum(disruption_window) / len(disruption_window)
                print(
                    f"Ep {ep:4d}/{n_episodes}  "
                    f"reward={ep_info['episode_reward']:7.3f}  "
                    f"steps={ep_info['episode_length']:4d}  "
                    f"disrupted={'Y' if ep_info['disrupted'] else 'N'}  "
                    f"risk={ep_info['disruption_risk']:.3f}  "
                    f"dr_rate(20)={dr_rate:.2f}  "
                    f"entropy={update_info['entropy']:.3f}"
                )

        self.training_stats = all_stats
        return all_stats
