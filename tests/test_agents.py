"""
Tests for RL agent implementations.
"""

import pytest
import numpy as np
import torch
import gymnasium as gym
from tokamak_rl.agents import (
    SACAgent, DreamerAgent, create_agent, BaseAgent,
    ReplayBuffer, Actor, Critic
)


class TestAgentBase:
    """Test base agent functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.obs_space = gym.spaces.Box(low=-1, high=1, shape=(45,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)
        
    def test_base_agent_interface(self):
        """Test that BaseAgent defines required interface."""
        # BaseAgent is abstract, so we can't instantiate it directly
        assert hasattr(BaseAgent, 'act')
        assert hasattr(BaseAgent, 'learn')
        assert hasattr(BaseAgent, 'save')
        assert hasattr(BaseAgent, 'load')
        
    def test_agent_factory(self):
        """Test agent factory function."""
        # Test SAC agent creation
        sac_agent = create_agent("SAC", self.obs_space, self.action_space)
        assert isinstance(sac_agent, SACAgent)
        
        # Test Dreamer agent creation
        dreamer_agent = create_agent("DREAMER", self.obs_space, self.action_space)
        assert isinstance(dreamer_agent, DreamerAgent)
        
        # Test invalid agent type
        with pytest.raises(ValueError, match="Unknown agent type"):
            create_agent("INVALID", self.obs_space, self.action_space)


class TestReplayBuffer:
    """Test replay buffer implementation."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.buffer = ReplayBuffer(capacity=1000)
        
    def test_buffer_initialization(self):
        """Test buffer initialization."""
        assert len(self.buffer) == 0
        assert self.buffer.capacity == 1000
        
    def test_buffer_push_and_sample(self):
        """Test adding experiences and sampling."""
        # Add some experiences
        for i in range(100):
            state = np.random.random(45).astype(np.float32)
            action = np.random.random(8).astype(np.float32)
            reward = np.random.random()
            next_state = np.random.random(45).astype(np.float32)
            done = np.random.random() > 0.9
            
            self.buffer.push(state, action, reward, next_state, done)
            
        assert len(self.buffer) == 100
        
        # Sample batch
        batch = self.buffer.sample(32)
        assert len(batch) == 5  # state, action, reward, next_state, done
        
        states, actions, rewards, next_states, dones = batch
        assert states.shape == (32, 45)
        assert actions.shape == (32, 8)
        assert rewards.shape == (32, 1)
        assert next_states.shape == (32, 45)
        assert dones.shape == (32, 1)
        
    def test_buffer_capacity_limit(self):
        """Test that buffer respects capacity limit."""
        # Fill buffer beyond capacity
        for i in range(1500):
            state = np.random.random(45).astype(np.float32)
            action = np.random.random(8).astype(np.float32)
            reward = 0.0
            next_state = np.random.random(45).astype(np.float32)
            done = False
            
            self.buffer.push(state, action, reward, next_state, done)
            
        # Should not exceed capacity
        assert len(self.buffer) == 1000


class TestNeuralNetworks:
    """Test neural network components."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.state_dim = 45
        self.action_dim = 8
        
    def test_actor_network(self):
        """Test actor network."""
        actor = Actor(self.state_dim, self.action_dim, hidden_dim=128)
        
        # Test forward pass
        state = torch.randn(32, self.state_dim)
        action = actor(state)
        
        assert action.shape == (32, self.action_dim)
        assert torch.all(action >= -1) and torch.all(action <= 1)  # Tanh output
        
    def test_critic_network(self):
        """Test critic network."""
        critic = Critic(self.state_dim, self.action_dim, hidden_dim=128)
        
        # Test forward pass
        state = torch.randn(32, self.state_dim)
        action = torch.randn(32, self.action_dim)
        q1, q2 = critic(state, action)
        
        assert q1.shape == (32, 1)
        assert q2.shape == (32, 1)
        
        # Test single Q-function
        q1_only = critic.q1_forward(state, action)
        assert q1_only.shape == (32, 1)


class TestSACAgent:
    """Test SAC agent implementation."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.obs_space = gym.spaces.Box(low=-1, high=1, shape=(45,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)
        self.agent = SACAgent(
            self.obs_space, self.action_space,
            learning_rate=1e-3, buffer_size=1000, batch_size=32
        )
        
    def test_sac_initialization(self):
        """Test SAC agent initialization."""
        assert hasattr(self.agent, 'actor')
        assert hasattr(self.agent, 'critic')
        assert hasattr(self.agent, 'critic_target')
        assert hasattr(self.agent, 'replay_buffer')
        
        assert self.agent.state_dim == 45
        assert self.agent.action_dim == 8
        
    def test_sac_action_selection(self):
        """Test SAC action selection."""
        obs = np.random.random(45).astype(np.float32)
        
        # Test deterministic action
        action_det = self.agent.act(obs, deterministic=True)
        assert action_det.shape == (8,)
        assert np.all(action_det >= -1) and np.all(action_det <= 1)
        
        # Test stochastic action
        action_stoch = self.agent.act(obs, deterministic=False)
        assert action_stoch.shape == (8,)
        assert np.all(action_stoch >= -1) and np.all(action_stoch <= 1)
        
    def test_sac_experience_storage(self):
        """Test experience storage."""
        state = np.random.random(45).astype(np.float32)
        action = np.random.random(8).astype(np.float32)
        reward = 1.0
        next_state = np.random.random(45).astype(np.float32)
        done = False
        
        self.agent.add_experience(state, action, reward, next_state, done)
        assert len(self.agent.replay_buffer) == 1
        
    def test_sac_learning(self):
        """Test SAC learning process."""
        # Fill buffer with some experiences
        for _ in range(100):
            state = np.random.random(45).astype(np.float32)
            action = np.random.random(8).astype(np.float32)
            reward = np.random.random()
            next_state = np.random.random(45).astype(np.float32)
            done = np.random.random() > 0.9
            
            self.agent.add_experience(state, action, reward, next_state, done)
            
        # Test learning
        metrics = self.agent.learn()
        
        # Should return training metrics
        assert isinstance(metrics, dict)
        if metrics:  # Only check if buffer is large enough
            assert 'critic_loss' in metrics
            assert 'actor_loss' in metrics
            assert 'q_value' in metrics
            
    def test_sac_save_load(self):
        """Test SAC model saving and loading."""
        import tempfile
        import os
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            save_path = f.name
            
        try:
            self.agent.save(save_path)
            assert os.path.exists(save_path)
            
            # Create new agent and load
            new_agent = SACAgent(self.obs_space, self.action_space)
            new_agent.load(save_path)
            
            # Test that loaded agent can act
            obs = np.random.random(45).astype(np.float32)
            action = new_agent.act(obs)
            assert action.shape == (8,)
            
        finally:
            if os.path.exists(save_path):
                os.unlink(save_path)


class TestDreamerAgent:
    """Test Dreamer agent implementation."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.obs_space = gym.spaces.Box(low=-1, high=1, shape=(45,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)
        self.agent = DreamerAgent(
            self.obs_space, self.action_space,
            learning_rate=1e-3, hidden_dim=64, latent_dim=16
        )
        
    def test_dreamer_initialization(self):
        """Test Dreamer agent initialization."""
        assert hasattr(self.agent, 'encoder')
        assert hasattr(self.agent, 'decoder')
        assert hasattr(self.agent, 'dynamics')
        assert hasattr(self.agent, 'reward_model')
        assert hasattr(self.agent, 'actor')
        
        assert self.agent.state_dim == 45
        assert self.agent.action_dim == 8
        assert self.agent.latent_dim == 16
        
    def test_dreamer_action_selection(self):
        """Test Dreamer action selection."""
        obs = np.random.random(45).astype(np.float32)
        
        action = self.agent.act(obs, deterministic=True)
        assert action.shape == (8,)
        assert np.all(action >= -1) and np.all(action <= 1)
        
    def test_dreamer_experience_storage(self):
        """Test experience storage."""
        state = np.random.random(45).astype(np.float32)
        action = np.random.random(8).astype(np.float32)
        reward = 1.0
        next_state = np.random.random(45).astype(np.float32)
        done = False
        
        self.agent.add_experience(state, action, reward, next_state, done)
        assert len(self.agent.experience_buffer) == 1
        
    def test_dreamer_learning(self):
        """Test Dreamer learning process."""
        # Fill buffer with some experiences
        for _ in range(50):
            state = np.random.random(45).astype(np.float32)
            action = np.random.random(8).astype(np.float32)
            reward = np.random.random()
            next_state = np.random.random(45).astype(np.float32)
            done = False
            
            self.agent.add_experience(state, action, reward, next_state, done)
            
        # Test learning
        metrics = self.agent.learn()
        
        # Should return training metrics
        assert isinstance(metrics, dict)
        if metrics:  # Only check if buffer has enough data
            assert 'world_loss' in metrics
            assert 'actor_loss' in metrics
            
    def test_dreamer_world_model(self):
        """Test world model components."""
        # Test encoder
        state = torch.randn(10, 45)
        latent = self.agent.encoder(state)
        assert latent.shape == (10, 16)
        
        # Test decoder
        reconstructed = self.agent.decoder(latent)
        assert reconstructed.shape == (10, 45)
        
        # Test dynamics
        action = torch.randn(10, 8)
        next_latent = self.agent.dynamics(torch.cat([latent, action], dim=1))
        assert next_latent.shape == (10, 16)
        
        # Test reward model
        reward_pred = self.agent.reward_model(torch.cat([latent, action], dim=1))
        assert reward_pred.shape == (10, 1)


class TestAgentIntegration:
    """Test agent integration with environment."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.obs_space = gym.spaces.Box(low=-1, high=1, shape=(45,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)
        
    def test_agent_environment_compatibility(self):
        """Test that agents are compatible with environment spaces."""
        agents = [
            create_agent("SAC", self.obs_space, self.action_space),
            create_agent("DREAMER", self.obs_space, self.action_space)
        ]
        
        for agent in agents:
            # Test action generation
            obs = self.obs_space.sample()
            action = agent.act(obs, deterministic=True)
            
            assert self.action_space.contains(action), f"Action {action} not in action space"
            
    def test_agent_training_loop(self):
        """Test basic training loop with agents."""
        agent = create_agent("SAC", self.obs_space, self.action_space, batch_size=16)
        
        # Simulate training episodes
        for episode in range(3):
            obs = self.obs_space.sample()
            
            for step in range(10):
                action = agent.act(obs, deterministic=False)
                next_obs = self.obs_space.sample()
                reward = np.random.random()
                done = step == 9
                
                # Store experience
                if hasattr(agent, 'add_experience'):
                    agent.add_experience(obs, action, reward, next_obs, done)
                
                # Learn
                if hasattr(agent, 'learn') and len(getattr(agent, 'replay_buffer', [])) >= 16:
                    metrics = agent.learn()
                    assert isinstance(metrics, dict)
                
                obs = next_obs
                
    def test_agent_determinism(self):
        """Test agent deterministic behavior."""
        agent = create_agent("SAC", self.obs_space, self.action_space)
        
        obs = np.random.random(45).astype(np.float32)
        
        # Multiple deterministic actions should be identical
        action1 = agent.act(obs, deterministic=True)
        action2 = agent.act(obs, deterministic=True)
        
        np.testing.assert_allclose(action1, action2, rtol=1e-6)
        
    def test_agent_device_handling(self):
        """Test agent device handling (CPU/GPU)."""
        # Test CPU device
        agent_cpu = create_agent("SAC", self.obs_space, self.action_space, device="cpu")
        assert agent_cpu.device == "cpu"
        
        obs = np.random.random(45).astype(np.float32)
        action = agent_cpu.act(obs)
        assert action.shape == (8,)
        
        # Test CUDA device if available
        if torch.cuda.is_available():
            agent_cuda = create_agent("SAC", self.obs_space, self.action_space, device="cuda")
            assert "cuda" in agent_cuda.device
            
            action_cuda = agent_cuda.act(obs)
            assert action_cuda.shape == (8,)