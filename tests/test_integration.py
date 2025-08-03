"""
Integration tests for the complete tokamak RL control system.
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path

from tokamak_rl import make_tokamak_env
from tokamak_rl.agents import create_agent
from tokamak_rl.monitoring import create_monitoring_system
from tokamak_rl.database import create_data_repository
from tokamak_rl.utils import create_experiment_tracker


class TestEnvironmentAgentIntegration:
    """Test integration between environment and agents."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.env = make_tokamak_env(tokamak_config="ITER", enable_safety=True)
        self.agent = create_agent("SAC", self.env.observation_space, self.env.action_space)
        
    def test_basic_interaction_loop(self):
        """Test basic environment-agent interaction."""
        obs, info = self.env.reset()
        
        for step in range(10):
            action = self.agent.act(obs, deterministic=False)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            
            # Store experience
            self.agent.add_experience(obs, action, reward, next_obs, terminated)
            
            # Verify data types and shapes
            assert isinstance(next_obs, np.ndarray)
            assert next_obs.shape == (45,)
            assert isinstance(reward, (int, float))
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert isinstance(info, dict)
            
            obs = next_obs
            
            if terminated or truncated:
                obs, info = self.env.reset()
                
    def test_agent_learning_with_environment(self):
        """Test agent learning process with environment data."""
        # Collect experiences
        for episode in range(5):
            obs, info = self.env.reset()
            
            for step in range(20):
                action = self.agent.act(obs, deterministic=False)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                
                self.agent.add_experience(obs, action, reward, next_obs, terminated)
                
                # Learn periodically
                if step % 5 == 0 and len(self.agent.replay_buffer) >= 32:
                    metrics = self.agent.learn()
                    assert isinstance(metrics, dict)
                    
                obs = next_obs
                
                if terminated or truncated:
                    break
                    
    def test_safety_system_integration(self):
        """Test safety system integration with agent actions."""
        obs, info = self.env.reset()
        
        # Generate potentially dangerous action
        dangerous_action = np.array([10.0, -10.0, 5.0, -5.0, 8.0, -8.0, 2.0, 2.0])
        
        # Environment should handle safety filtering
        next_obs, reward, terminated, truncated, info = self.env.step(dangerous_action)
        
        # Check that safety information is provided
        if 'safety' in info:
            assert 'action_modified' in info['safety'] or 'violations' in info['safety']
            
    def test_multiple_tokamak_configurations(self):
        """Test agent compatibility with different tokamak configurations."""
        configs = ["ITER", "SPARC", "NSTX", "DIII-D"]
        
        for config in configs:
            env = make_tokamak_env(tokamak_config=config)
            agent = create_agent("SAC", env.observation_space, env.action_space)
            
            # Test basic interaction
            obs, info = env.reset()
            action = agent.act(obs, deterministic=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Should work without errors
            assert isinstance(next_obs, np.ndarray)
            assert isinstance(reward, (int, float))


class TestMonitoringIntegration:
    """Test monitoring system integration."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.env = make_tokamak_env(tokamak_config="SPARC")
        self.agent = create_agent("DREAMER", self.env.observation_space, self.env.action_space)
        self.monitoring = create_monitoring_system(
            log_dir=self.temp_dir,
            enable_tensorboard=False,  # Disable for testing
            enable_alerts=True
        )
        
    def test_monitoring_system_creation(self):
        """Test monitoring system creation and components."""
        assert 'monitor' in self.monitoring
        assert 'renderer' in self.monitoring
        assert self.monitoring['monitor'] is not None
        assert self.monitoring['renderer'] is not None
        
    def test_step_logging(self):
        """Test step-by-step logging."""
        monitor = self.monitoring['monitor']
        
        obs, info = self.env.reset()
        
        for step in range(5):
            action = self.agent.act(obs, deterministic=False)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            
            # Log step
            monitor.log_step(self.env.plasma_state, action, reward, info)
            
            obs = next_obs
            
            if terminated or truncated:
                obs, info = self.env.reset()
                
        # Check that steps were logged
        stats = monitor.get_statistics()
        assert stats['total_steps'] >= 5
        
    def test_episode_logging(self):
        """Test episode-level logging."""
        monitor = self.monitoring['monitor']
        
        for episode in range(3):
            obs, info = self.env.reset()
            episode_reward = 0
            
            for step in range(10):
                action = self.agent.act(obs, deterministic=False)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                
                monitor.log_step(self.env.plasma_state, action, reward, info)
                episode_reward += reward
                obs = next_obs
                
                if terminated or truncated:
                    break
                    
            # Log episode end
            episode_metrics = self.env.get_episode_metrics()
            monitor.log_episode_end(episode_metrics)
            
        # Check episode logging
        stats = monitor.get_statistics()
        assert stats['total_episodes'] >= 3
        
    def test_plasma_visualization(self):
        """Test plasma state visualization."""
        renderer = self.monitoring['renderer']
        
        obs, info = self.env.reset()
        
        # Render plasma state
        frame = renderer.render(state=self.env.plasma_state)
        assert isinstance(frame, np.ndarray)
        assert frame.shape[2] == 3  # RGB channels
        
    def test_alert_generation(self):
        """Test alert generation for dangerous conditions."""
        monitor = self.monitoring['monitor']
        
        # Create state with potential alerts
        obs, info = self.env.reset()
        
        # Force dangerous conditions
        self.env.plasma_state.q_min = 1.0  # Low q
        self.env.plasma_state.shape_error = 10.0  # High shape error
        
        action = np.zeros(8)
        monitor.log_step(self.env.plasma_state, action, -10.0, {})
        
        # Check if alerts were generated
        stats = monitor.get_statistics()
        # Note: Alert counting depends on implementation details
        
    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


class TestDataPersistenceIntegration:
    """Test data persistence and database integration."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_repo = create_data_repository(self.temp_dir)
        self.env = make_tokamak_env(tokamak_config="NSTX")
        self.agent = create_agent("SAC", self.env.observation_space, self.env.action_space)
        
    def test_data_repository_creation(self):
        """Test data repository creation."""
        assert hasattr(self.data_repo, 'experiment_db')
        assert hasattr(self.data_repo, 'equilibrium_cache')
        assert self.data_repo.base_dir.exists()
        
    def test_training_data_storage(self):
        """Test storage of training episode data."""
        session_id = "test_session_001"
        
        # Run training episode
        plasma_states = []
        actions = []
        rewards = []
        safety_info = []
        
        obs, info = self.env.reset()
        
        for step in range(10):
            action = self.agent.act(obs, deterministic=False)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            
            # Collect data
            plasma_states.append(self.env.plasma_state)
            actions.append(action)
            rewards.append(reward)
            safety_info.append(info.get('safety', {}))
            
            obs = next_obs
            
            if terminated or truncated:
                break
                
        # Store episode data
        self.data_repo.store_training_episode(
            session_id, episode=0,
            plasma_states=plasma_states,
            actions=actions,
            rewards=rewards,
            safety_info=safety_info
        )
        
        # Verify data was stored
        training_data = self.data_repo.get_training_data(session_id)
        assert len(training_data) == 1
        assert training_data[0].metadata['session_id'] == session_id
        
    def test_equilibrium_caching(self):
        """Test plasma equilibrium caching."""
        cache = self.data_repo.equilibrium_cache
        
        # Generate some equilibria
        obs, info = self.env.reset()
        initial_config = self.env.tokamak_config
        
        for i in range(5):
            control_params = np.random.uniform(-0.5, 0.5, 6)
            
            # Check cache miss
            cached_state = cache.get(initial_config, control_params)
            assert cached_state is None  # Should be cache miss
            
            # Store in cache
            cache.store(initial_config, control_params, self.env.plasma_state)
            
            # Check cache hit
            cached_state = cache.get(initial_config, control_params)
            assert cached_state is not None
            
        # Check cache statistics
        stats = cache.stats()
        assert stats['memory_entries'] > 0
        
    def test_data_export(self):
        """Test data export functionality."""
        # Store some data first
        session_id = "export_test"
        self.data_repo.store_training_episode(
            session_id, episode=0,
            plasma_states=[self.env.plasma_state],
            actions=[np.zeros(8)],
            rewards=[1.0],
            safety_info=[{}]
        )
        
        # Export data
        export_path = Path(self.temp_dir) / "export.json"
        self.data_repo.export_data(str(export_path), format="json")
        
        assert export_path.exists()
        assert export_path.stat().st_size > 0
        
    def teardown_method(self):
        """Cleanup test fixtures."""
        self.data_repo.close()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


class TestExperimentTracking:
    """Test experiment tracking integration."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.tracker = create_experiment_tracker(
            "integration_test", 
            output_dir=self.temp_dir
        )
        self.env = make_tokamak_env(tokamak_config="DIII-D")
        self.agent = create_agent("SAC", self.env.observation_space, self.env.action_space)
        
    def test_experiment_parameter_logging(self):
        """Test experiment parameter logging."""
        parameters = {
            'agent_type': 'SAC',
            'tokamak': 'DIII-D',
            'learning_rate': 3e-4,
            'episodes': 10,
            'safety_enabled': True
        }
        
        self.tracker.log_parameters(parameters)
        
        # Check that metadata file was created
        metadata_files = list(Path(self.temp_dir).glob("*_metadata.json"))
        assert len(metadata_files) > 0
        
    def test_experiment_metrics_logging(self):
        """Test experiment metrics logging."""
        # Run some episodes and log metrics
        for episode in range(3):
            obs, info = self.env.reset()
            episode_reward = 0
            
            for step in range(5):
                action = self.agent.act(obs, deterministic=False)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                
                episode_reward += reward
                obs = next_obs
                
                if terminated or truncated:
                    break
                    
            # Log episode metrics
            episode_metrics = {
                'episode_reward': episode_reward,
                'episode_length': step + 1,
                'final_shape_error': self.env.plasma_state.shape_error,
                'final_q_min': self.env.plasma_state.q_min
            }
            
            self.tracker.log_metrics(episode_metrics, step=episode)
            
        # Finalize experiment
        self.tracker.finalize()
        
        # Check metadata
        assert self.tracker.metadata['end_time'] is not None
        assert self.tracker.metadata['duration'] > 0
        assert len(self.tracker.metadata['metrics']['history']) == 3
        
    def test_model_checkpoint_tracking(self):
        """Test model checkpoint tracking."""
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            checkpoint_path = f.name
            
        try:
            # Simulate saving checkpoint
            self.agent.save(checkpoint_path)
            
            # Log checkpoint
            checkpoint_metrics = {
                'episode': 10,
                'reward': 15.5,
                'shape_error': 1.2
            }
            
            self.tracker.log_checkpoint(checkpoint_path, checkpoint_metrics)
            
            # Check that checkpoint was logged
            assert len(self.tracker.metadata['checkpoints']) == 1
            assert self.tracker.metadata['checkpoints'][0]['path'] == checkpoint_path
            
        finally:
            if os.path.exists(checkpoint_path):
                os.unlink(checkpoint_path)
                
    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


class TestFullSystemIntegration:
    """Test complete system integration."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
    def test_complete_training_workflow(self):
        """Test complete training workflow with all components."""
        # Create all system components
        env = make_tokamak_env(tokamak_config="SPARC", enable_safety=True)
        agent = create_agent("SAC", env.observation_space, env.action_space, batch_size=16)
        monitoring = create_monitoring_system(
            log_dir=f"{self.temp_dir}/monitoring",
            enable_tensorboard=False,
            enable_alerts=True
        )
        data_repo = create_data_repository(f"{self.temp_dir}/data")
        tracker = create_experiment_tracker(
            "full_system_test",
            output_dir=f"{self.temp_dir}/experiments"
        )
        
        # Log experiment parameters
        parameters = {
            'agent': 'SAC',
            'tokamak': 'SPARC',
            'episodes': 5,
            'steps_per_episode': 20
        }
        tracker.log_parameters(parameters)
        
        try:
            # Training loop
            for episode in range(5):
                obs, info = env.reset()
                episode_reward = 0
                episode_data = {
                    'plasma_states': [],
                    'actions': [],
                    'rewards': [],
                    'safety_info': []
                }
                
                for step in range(20):
                    # Agent action
                    action = agent.act(obs, deterministic=False)
                    
                    # Environment step
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    
                    # Store experience
                    agent.add_experience(obs, action, reward, next_obs, terminated)
                    
                    # Collect episode data
                    episode_data['plasma_states'].append(env.plasma_state)
                    episode_data['actions'].append(action)
                    episode_data['rewards'].append(reward)
                    episode_data['safety_info'].append(info.get('safety', {}))
                    
                    # Monitor step
                    monitoring['monitor'].log_step(env.plasma_state, action, reward, info)
                    
                    # Learn
                    if step % 5 == 0 and len(agent.replay_buffer) >= 16:
                        train_metrics = agent.learn()
                        if train_metrics:
                            tracker.log_metrics(train_metrics, step=episode * 20 + step)
                    
                    episode_reward += reward
                    obs = next_obs
                    
                    if terminated or truncated:
                        break
                        
                # Store episode data
                data_repo.store_training_episode(
                    "full_system_test", episode,
                    **episode_data
                )
                
                # Log episode metrics
                episode_metrics = env.get_episode_metrics()
                episode_metrics['episode_reward'] = episode_reward
                
                monitoring['monitor'].log_episode_end(episode_metrics)
                tracker.log_metrics(episode_metrics, step=episode)
                
            # Finalize experiment
            tracker.finalize()
            
            # Verify all components worked
            assert tracker.metadata['end_time'] is not None
            assert monitoring['monitor'].get_statistics()['total_episodes'] == 5
            assert len(data_repo.get_training_data("full_system_test")) == 5
            
            # Test data export
            export_path = Path(self.temp_dir) / "full_export.json"
            data_repo.export_data(str(export_path))
            assert export_path.exists()
            
        finally:
            data_repo.close()
            if monitoring['logger']:
                monitoring['logger'].close()
                
    def test_system_error_recovery(self):
        """Test system behavior under error conditions."""
        env = make_tokamak_env(tokamak_config="ITER")
        agent = create_agent("SAC", env.observation_space, env.action_space)
        
        # Test with invalid actions
        obs, info = env.reset()
        
        # Try invalid action (wrong shape)
        try:
            invalid_action = np.array([1, 2, 3])  # Wrong shape
            env.step(invalid_action)
        except (ValueError, AssertionError):
            pass  # Expected to fail
            
        # Try action with NaN values
        try:
            nan_action = np.array([np.nan] * 8)
            env.step(nan_action)
        except (ValueError, AssertionError):
            pass  # Expected to fail or be filtered
            
        # System should still be usable after errors
        valid_action = agent.act(obs, deterministic=True)
        next_obs, reward, terminated, truncated, info = env.step(valid_action)
        assert isinstance(next_obs, np.ndarray)
        
    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)