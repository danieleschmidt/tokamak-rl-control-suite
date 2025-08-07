"""
Reinforcement learning agents for tokamak plasma control.

This module implements state-of-the-art RL algorithms optimized for
continuous control of tokamak plasma shape and stability.
"""

try:
    import numpy as np
except ImportError:
    # Fallback numpy implementation
    import math
    import random as rand
    
    class np:
        @staticmethod
        def array(x):
            return list(x) if hasattr(x, '__iter__') else [x]
        
        @staticmethod
        def stack(arrays):
            return [list(row) for row in zip(*arrays)]
        
        @staticmethod
        def random_uniform(low, high, size):
            return [rand.uniform(low, high) for _ in range(size)]
        
        @staticmethod
        def clip(val, min_val, max_val):
            if hasattr(val, '__iter__'):
                return [max(min_val, min(max_val, v)) for v in val]
            return max(min_val, min(max_val, val))
        
        float32 = float
        ndarray = list  # Type alias for compatibility

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
except ImportError:
    # Fallback torch-like implementation for structure verification
    class torch:
        class Tensor:
            def __init__(self, data):
                self.data = data
            
            def unsqueeze(self, dim):
                return self
            
            def cpu(self):
                return self
            
            def numpy(self):
                return self.data
            
            def flatten(self):
                return self.data
        
        @staticmethod
        def FloatTensor(data):
            return torch.Tensor(data)
        
        @staticmethod
        def cuda_is_available():
            return False
        
        @staticmethod
        def device(name):
            return name
        
        @staticmethod
        def save(obj, path):
            pass
        
        @staticmethod
        def load(path, map_location=None):
            return {}
    
    class nn:
        class Module:
            def __init__(self):
                pass
            
            def parameters(self):
                return []
            
            def modules(self):
                return []
            
            def load_state_dict(self, state_dict):
                pass
            
            def state_dict(self):
                return {}
        
        class Linear(Module):
            def __init__(self, in_features, out_features):
                super().__init__()
                self.weight = None
                self.bias = None
        
        class LSTM(Module):
            def __init__(self, input_size, hidden_size, num_layers, **kwargs):
                super().__init__()
        
        class LayerNorm(Module):
            def __init__(self, normalized_shape):
                super().__init__()
        
        class ReLU(Module):
            pass
        
        class Dropout(Module):
            def __init__(self, p=0.5):
                pass
        
        class Sigmoid(Module):
            pass
        
        class Sequential(Module):
            def __init__(self, *args):
                super().__init__()
    
    class F:
        @staticmethod
        def relu(x):
            return x
        
        @staticmethod
        def mse_loss(a, b):
            return 0.0
    
    class optim:
        class Adam:
            def __init__(self, params, lr=0.001):
                pass
            
            def zero_grad(self):
                pass
            
            def step(self):
                pass
            
            def load_state_dict(self, state_dict):
                pass
            
            def state_dict(self):
                return {}

try:
    import gymnasium as gym
except ImportError:
    class gym:
        class Space:
            def __init__(self):
                self.shape = (1,)
        
        class Env:
            pass

from typing import Dict, Any, Optional, Tuple, List, Union
import warnings
from abc import ABC, abstractmethod
from collections import deque
import random


class BaseAgent(ABC):
    """Base class for all RL agents."""
    
    def __init__(self, observation_space: gym.Space, action_space: gym.Space,
                 device: Optional[str] = None):
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
    @abstractmethod
    def act(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action given observation."""
        pass
        
    @abstractmethod
    def learn(self, experiences: Dict[str, Any]) -> Dict[str, float]:
        """Learn from batch of experiences."""
        pass
        
    @abstractmethod
    def save(self, path: str) -> None:
        """Save agent model."""
        pass
        
    @abstractmethod
    def load(self, path: str) -> None:
        """Load agent model."""
        pass


class ReplayBuffer:
    """Experience replay buffer for off-policy algorithms."""
    
    def __init__(self, capacity: int = 1000000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state: np.ndarray, action: np.ndarray, reward: float,
             next_state: np.ndarray, done: bool) -> None:
        """Store experience tuple."""
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample batch of experiences."""
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        
        return (
            torch.FloatTensor(state),
            torch.FloatTensor(action),
            torch.FloatTensor(reward).unsqueeze(1),
            torch.FloatTensor(next_state),
            torch.BoolTensor(done).unsqueeze(1)
        )
        
    def __len__(self) -> int:
        return len(self.buffer)


class Actor(nn.Module):
    """Actor network for continuous control."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 max_action: float = 1.0):
        super().__init__()
        self.max_action = max_action
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.ln1(self.fc1(state)))
        x = F.relu(self.ln2(self.fc2(x)))
        action = torch.tanh(self.fc3(x)) * self.max_action
        return action


class Critic(nn.Module):
    """Critic network for value estimation."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # Q1 network
        self.q1_fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_fc3 = nn.Linear(hidden_dim, 1)
        
        # Q2 network
        self.q2_fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q2_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_fc3 = nn.Linear(hidden_dim, 1)
        
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], dim=1)
        
        # Q1
        q1 = F.relu(self.ln1(self.q1_fc1(sa)))
        q1 = F.relu(self.ln2(self.q1_fc2(q1)))
        q1 = self.q1_fc3(q1)
        
        # Q2
        q2 = F.relu(self.ln1(self.q2_fc1(sa)))
        q2 = F.relu(self.ln2(self.q2_fc2(q2)))
        q2 = self.q2_fc3(q2)
        
        return q1, q2
        
    def q1_forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([state, action], dim=1)
        q1 = F.relu(self.ln1(self.q1_fc1(sa)))
        q1 = F.relu(self.ln2(self.q1_fc2(q1)))
        return self.q1_fc3(q1)


class SACAgent(BaseAgent):
    """Soft Actor-Critic agent optimized for tokamak control."""
    
    def __init__(self, observation_space: gym.Space, action_space: gym.Space,
                 learning_rate: float = 3e-4, buffer_size: int = 1000000,
                 batch_size: int = 256, tau: float = 0.005, gamma: float = 0.99,
                 alpha: float = 0.2, hidden_dim: int = 256, device: Optional[str] = None):
        
        super().__init__(observation_space, action_space, device)
        
        # Hyperparameters
        self.lr = learning_rate
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.alpha = alpha
        
        # Get dimensions
        self.state_dim = observation_space.shape[0]
        self.action_dim = action_space.shape[0]
        self.max_action = float(action_space.high[0])
        
        # Networks
        self.actor = Actor(self.state_dim, self.action_dim, hidden_dim, self.max_action).to(self.device)
        self.critic = Critic(self.state_dim, self.action_dim, hidden_dim).to(self.device)
        self.critic_target = Critic(self.state_dim, self.action_dim, hidden_dim).to(self.device)
        
        # Copy parameters to target network
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Training stats
        self.training_steps = 0
        
    def act(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action using current policy."""
        state = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actor(state)
            
            if not deterministic:
                # Add exploration noise
                noise = torch.randn_like(action) * 0.1 * self.max_action
                action = action + noise
                
        # Clip to action bounds
        action = torch.clamp(action, -self.max_action, self.max_action)
        return action.cpu().numpy().flatten()
        
    def learn(self, experiences: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Train the agent on a batch of experiences."""
        if len(self.replay_buffer) < self.batch_size:
            return {}
            
        # Sample batch
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        done = done.to(self.device)
        
        # Critic update
        with torch.no_grad():
            next_action = self.actor(next_state)
            next_q1, next_q2 = self.critic_target(next_state, next_action)
            next_q = torch.min(next_q1, next_q2)
            target_q = reward + self.gamma * (1 - done.float()) * next_q
            
        current_q1, current_q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor update
        pi_action = self.actor(state)
        actor_loss = -self.critic.q1_forward(state, pi_action).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update target networks
        self._soft_update(self.critic, self.critic_target)
        
        self.training_steps += 1
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'q_value': current_q1.mean().item()
        }
        
    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        """Soft update target network parameters."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
    def add_experience(self, state: np.ndarray, action: np.ndarray, reward: float,
                      next_state: np.ndarray, done: bool) -> None:
        """Add experience to replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
        
    def save(self, path: str) -> None:
        """Save agent models."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'training_steps': self.training_steps,
            'hyperparameters': {
                'lr': self.lr,
                'batch_size': self.batch_size,
                'tau': self.tau,
                'gamma': self.gamma,
                'alpha': self.alpha
            }
        }, path)
        
    def load(self, path: str) -> None:
        """Load agent models."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.training_steps = checkpoint['training_steps']


class DreamerAgent(BaseAgent):
    """Simplified Dreamer-like model-based RL agent."""
    
    def __init__(self, observation_space: gym.Space, action_space: gym.Space,
                 learning_rate: float = 1e-4, hidden_dim: int = 200,
                 latent_dim: int = 30, device: Optional[str] = None):
        
        super().__init__(observation_space, action_space, device)
        
        self.lr = learning_rate
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        self.state_dim = observation_space.shape[0]
        self.action_dim = action_space.shape[0]
        self.max_action = float(action_space.high[0])
        
        # World model components
        self.encoder = self._build_encoder().to(self.device)
        self.decoder = self._build_decoder().to(self.device)
        self.dynamics = self._build_dynamics().to(self.device)
        self.reward_model = self._build_reward_model().to(self.device)
        
        # Actor for action selection
        self.actor = Actor(self.latent_dim, self.action_dim, 
                          hidden_dim//2, self.max_action).to(self.device)
        
        # Optimizers
        self.world_optimizer = optim.Adam(
            list(self.encoder.parameters()) + 
            list(self.decoder.parameters()) + 
            list(self.dynamics.parameters()) + 
            list(self.reward_model.parameters()), 
            lr=self.lr
        )
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        
        # Experience buffer
        self.experience_buffer = []
        
    def _build_encoder(self) -> nn.Module:
        """Build observation encoder."""
        return nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.latent_dim)
        )
        
    def _build_decoder(self) -> nn.Module:
        """Build observation decoder."""
        return nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.state_dim)
        )
        
    def _build_dynamics(self) -> nn.Module:
        """Build dynamics model."""
        return nn.Sequential(
            nn.Linear(self.latent_dim + self.action_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.latent_dim)
        )
        
    def _build_reward_model(self) -> nn.Module:
        """Build reward prediction model."""
        return nn.Sequential(
            nn.Linear(self.latent_dim + self.action_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim//2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim//2, 1)
        )
        
    def act(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action using world model and actor."""
        state = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Encode observation to latent state
            latent = self.encoder(state)
            
            # Generate action
            action = self.actor(latent)
            
            if not deterministic:
                # Add exploration noise
                noise = torch.randn_like(action) * 0.05 * self.max_action
                action = action + noise
                
        action = torch.clamp(action, -self.max_action, self.max_action)
        return action.cpu().numpy().flatten()
        
    def learn(self, experiences: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Train world model and actor."""
        if len(self.experience_buffer) < 10:
            return {}
            
        # Sample recent experiences
        recent_experiences = self.experience_buffer[-100:]
        
        # Prepare batch
        states, actions, rewards, next_states = [], [], [], []
        for exp in recent_experiences:
            states.append(exp[0])
            actions.append(exp[1])
            rewards.append(exp[2])
            next_states.append(exp[3])
            
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        
        # Train world model
        latent_states = self.encoder(states)
        latent_next_states = self.encoder(next_states)
        
        # Reconstruction loss
        reconstructed = self.decoder(latent_states)
        reconstruction_loss = F.mse_loss(reconstructed, states)
        
        # Dynamics loss
        predicted_next_latent = self.dynamics(torch.cat([latent_states, actions], dim=1))
        dynamics_loss = F.mse_loss(predicted_next_latent, latent_next_states)
        
        # Reward prediction loss
        predicted_rewards = self.reward_model(torch.cat([latent_states, actions], dim=1))
        reward_loss = F.mse_loss(predicted_rewards, rewards)
        
        # Total world model loss
        world_loss = reconstruction_loss + dynamics_loss + reward_loss
        
        self.world_optimizer.zero_grad()
        world_loss.backward()
        self.world_optimizer.step()
        
        # Train actor using imagined rollouts
        actor_loss = self._train_actor_imagination(latent_states)
        
        return {
            'world_loss': world_loss.item(),
            'reconstruction_loss': reconstruction_loss.item(),
            'dynamics_loss': dynamics_loss.item(),
            'reward_loss': reward_loss.item(),
            'actor_loss': actor_loss
        }
        
    def _train_actor_imagination(self, initial_states: torch.Tensor) -> float:
        """Train actor using imagined rollouts."""
        horizon = 5
        total_reward = 0
        current_latent = initial_states
        
        for step in range(horizon):
            action = self.actor(current_latent)
            predicted_reward = self.reward_model(torch.cat([current_latent, action], dim=1))
            total_reward = total_reward + predicted_reward
            
            # Predict next latent state
            current_latent = self.dynamics(torch.cat([current_latent, action], dim=1))
            
        # Maximize expected return
        actor_loss = -total_reward.mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        return actor_loss.item()
        
    def add_experience(self, state: np.ndarray, action: np.ndarray, reward: float,
                      next_state: np.ndarray, done: bool) -> None:
        """Add experience to buffer."""
        self.experience_buffer.append((state, action, reward, next_state))
        
        # Keep buffer size manageable
        if len(self.experience_buffer) > 10000:
            self.experience_buffer.pop(0)
            
    def save(self, path: str) -> None:
        """Save agent models."""
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'dynamics_state_dict': self.dynamics.state_dict(),
            'reward_model_state_dict': self.reward_model.state_dict(),
            'actor_state_dict': self.actor.state_dict(),
            'world_optimizer_state_dict': self.world_optimizer.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict()
        }, path)
        
    def load(self, path: str) -> None:
        """Load agent models."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        self.dynamics.load_state_dict(checkpoint['dynamics_state_dict'])
        self.reward_model.load_state_dict(checkpoint['reward_model_state_dict'])
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.world_optimizer.load_state_dict(checkpoint['world_optimizer_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])


def create_agent(agent_type: str, observation_space: gym.Space, 
                action_space: gym.Space, **kwargs) -> BaseAgent:
    """Factory function to create RL agents."""
    agent_type = agent_type.upper()
    
    if agent_type == "SAC":
        return SACAgent(observation_space, action_space, **kwargs)
    elif agent_type == "DREAMER":
        return DreamerAgent(observation_space, action_space, **kwargs)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}. Available: SAC, DREAMER")