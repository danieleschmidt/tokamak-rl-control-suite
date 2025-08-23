"""
Comprehensive test suite for quantum plasma control system.

Tests quantum-enhanced algorithms, superposition state management,
and performance validation against classical methods.
"""

import pytest
import math
import random
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from tokamak_rl.quantum_plasma_control import (
    QuantumPlasmaController, 
    QuantumEnhancedSAC,
    QuantumPlasmaState,
    create_quantum_enhanced_training_system,
    validate_quantum_advantage
)


class TestQuantumPlasmaController:
    """Test suite for QuantumPlasmaController."""
    
    def test_initialization(self):
        """Test quantum controller initialization."""
        controller = QuantumPlasmaController(n_qubits=6, coherence_time=2e-3)
        
        assert controller.n_qubits == 6
        assert controller.coherence_time == 2e-3
        assert len(controller.quantum_state.amplitude_real) == 2**6
        assert len(controller.quantum_state.amplitude_imag) == 2**6
        assert len(controller.quantum_state.basis_states) == 2**6
        
        # Check normalization (approximately)
        total_amplitude = sum(abs(complex(r, i)) for r, i in zip(
            controller.quantum_state.amplitude_real,
            controller.quantum_state.amplitude_imag
        ))
        assert 0.9 < total_amplitude < 1.1  # Should be close to 1
    
    def test_quantum_plasma_evolution(self):
        """Test quantum plasma evolution process."""
        controller = QuantumPlasmaController(n_qubits=4)
        
        plasma_obs = [1.0, 0.5, -0.3, 0.8, 0.2]
        control_action = [0.1, -0.2, 0.3, -0.1]
        
        optimal_control, quantum_advantage = controller.quantum_plasma_evolution(
            plasma_obs, control_action
        )
        
        # Check output format
        assert isinstance(optimal_control, list)
        assert len(optimal_control) >= len(control_action)
        assert isinstance(quantum_advantage, float)
        assert 0.0 <= quantum_advantage <= 1.0
        
        # Check control bounds
        for control_val in optimal_control:
            assert -1.0 <= control_val <= 1.0
    
    def test_quantum_state_encoding(self):
        """Test plasma state encoding into quantum amplitudes."""
        controller = QuantumPlasmaController(n_qubits=4)
        
        plasma_obs = [0.5, -0.5, 0.0, 1.0]
        encoded_state = controller._encode_plasma_state(plasma_obs)
        
        assert len(encoded_state) == 2**4
        
        # All encoded states should be complex numbers
        for amplitude in encoded_state:
            assert isinstance(amplitude, complex)
    
    def test_quantum_interference(self):
        """Test quantum interference pattern calculation."""
        controller = QuantumPlasmaController(n_qubits=3)
        
        # Create test evolved state
        evolved_state = [complex(random.random(), random.random()) for _ in range(8)]
        
        interference_pattern = controller._calculate_interference(evolved_state)
        
        assert isinstance(interference_pattern, list)
        assert len(interference_pattern) > 0
        
        # Each interference value should be real
        for interference in interference_pattern:
            assert isinstance(interference, (int, float))
    
    def test_quantum_measurement(self):
        """Test quantum measurement process."""
        controller = QuantumPlasmaController(n_qubits=4)
        
        interference_pattern = [0.5, -0.3, 0.8, -0.2, 0.1]
        
        optimal_control = controller._quantum_measurement(interference_pattern)
        
        assert isinstance(optimal_control, list)
        assert len(optimal_control) <= 8  # Limited to action space
        
        # All control values should be bounded
        for control in optimal_control:
            assert -1.0 <= control <= 1.0
    
    def test_quantum_advantage_calculation(self):
        """Test quantum advantage metric calculation."""
        controller = QuantumPlasmaController(n_qubits=3)
        
        # Create test evolved state
        evolved_state = [complex(0.3, 0.4), complex(0.2, 0.1), complex(0.5, 0.2)]
        
        quantum_advantage = controller._calculate_quantum_advantage(evolved_state)
        
        assert isinstance(quantum_advantage, float)
        assert 0.0 <= quantum_advantage <= 1.0
    
    def test_decoherence_application(self):
        """Test quantum decoherence effects."""
        controller = QuantumPlasmaController(n_qubits=3, coherence_time=1e-3)
        
        # Store initial amplitudes
        initial_amplitudes = controller.quantum_state.amplitude_real.copy()
        
        # Apply decoherence
        controller._apply_decoherence()
        
        # Amplitudes should change (decay)
        final_amplitudes = controller.quantum_state.amplitude_real
        
        # Most amplitudes should be smaller (decayed)
        smaller_count = sum(1 for i, f in zip(initial_amplitudes, final_amplitudes) if abs(f) < abs(i))
        assert smaller_count >= len(initial_amplitudes) * 0.7  # At least 70% decayed
    
    def test_quantum_control_metrics(self):
        """Test quantum control metrics calculation."""
        controller = QuantumPlasmaController(n_qubits=4)
        
        metrics = controller.quantum_control_metrics()
        
        required_metrics = [
            'quantum_coherence', 'entanglement_entropy', 'measurement_consistency',
            'coherence_time_remaining', 'quantum_advantage'
        ]
        
        for metric in required_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
            
        assert 0.0 <= metrics['quantum_coherence'] <= 2.0
        assert metrics['entanglement_entropy'] >= 0.0
        assert 0.0 <= metrics['measurement_consistency'] <= 1.0


class TestQuantumEnhancedSAC:
    """Test suite for QuantumEnhancedSAC."""
    
    def test_initialization(self):
        """Test quantum-enhanced SAC initialization."""
        agent = QuantumEnhancedSAC(observation_dim=20, action_dim=6, quantum_enhancement=True)
        
        assert agent.observation_dim == 20
        assert agent.action_dim == 6
        assert agent.quantum_enhancement is True
        assert agent.quantum_controller is not None
        
        # Test without quantum enhancement
        classical_agent = QuantumEnhancedSAC(observation_dim=20, action_dim=6, quantum_enhancement=False)
        assert classical_agent.quantum_enhancement is False
        assert not hasattr(classical_agent, 'quantum_controller') or classical_agent.quantum_controller is None
    
    def test_action_selection_quantum(self):
        """Test quantum-enhanced action selection."""
        agent = QuantumEnhancedSAC(observation_dim=10, action_dim=4, quantum_enhancement=True)
        
        observation = [random.uniform(-1, 1) for _ in range(10)]
        
        action = agent.select_action(observation, training=True)
        
        assert isinstance(action, list)
        assert len(action) == 4
        
        # Actions should be bounded
        for a in action:
            assert -1.0 <= a <= 1.0
        
        # Quantum advantages should be recorded
        assert len(agent.quantum_advantages) >= 1
    
    def test_action_selection_classical(self):
        """Test classical action selection."""
        agent = QuantumEnhancedSAC(observation_dim=10, action_dim=4, quantum_enhancement=False)
        
        observation = [random.uniform(-1, 1) for _ in range(10)]
        
        action = agent.select_action(observation, training=False)
        
        assert isinstance(action, list)
        assert len(action) == 4
        
        for a in action:
            assert -1.0 <= a <= 1.0
    
    def test_transition_storage(self):
        """Test transition storage in replay buffer."""
        agent = QuantumEnhancedSAC(observation_dim=5, action_dim=3, quantum_enhancement=True)
        
        obs = [0.1, 0.2, 0.3, 0.4, 0.5]
        action = [0.1, -0.2, 0.3]
        reward = 1.5
        next_obs = [0.2, 0.3, 0.4, 0.5, 0.6]
        done = False
        
        agent.store_transition(obs, action, reward, next_obs, done)
        
        assert len(agent.replay_buffer) == 1
        
        stored_transition = agent.replay_buffer[0]
        assert stored_transition['observation'] == obs
        assert stored_transition['action'] == action
        assert stored_transition['reward'] == reward
        assert stored_transition['next_observation'] == next_obs
        assert stored_transition['done'] == done
    
    def test_agent_update(self):
        """Test agent update process."""
        agent = QuantumEnhancedSAC(observation_dim=5, action_dim=3, quantum_enhancement=True)
        
        # Fill replay buffer with some transitions
        for _ in range(10):
            obs = [random.uniform(-1, 1) for _ in range(5)]
            action = [random.uniform(-1, 1) for _ in range(3)]
            reward = random.uniform(-1, 1)
            next_obs = [random.uniform(-1, 1) for _ in range(5)]
            done = random.choice([True, False])
            
            agent.store_transition(obs, action, reward, next_obs, done)
        
        # Add some quantum advantages
        agent.quantum_advantages.extend([0.6, 0.7, 0.8])
        
        initial_coherence_time = agent.quantum_controller.coherence_time
        
        # Update should not raise errors
        agent.update(batch_size=5)
        
        # Coherence time might have changed
        assert agent.quantum_controller.coherence_time >= 0
    
    def test_metrics_collection(self):
        """Test metrics collection."""
        agent = QuantumEnhancedSAC(observation_dim=8, action_dim=4, quantum_enhancement=True)
        
        # Add some episode rewards and quantum advantages
        agent.episode_rewards = [1.0, 2.0, 1.5, 3.0]
        agent.quantum_advantages = [0.5, 0.6, 0.7, 0.8]
        
        metrics = agent.get_metrics()
        
        assert 'average_reward' in metrics
        assert 'buffer_size' in metrics
        assert 'learning_rate' in metrics
        assert 'quantum_quantum_coherence' in metrics  # Quantum metrics have 'quantum_' prefix
        assert 'average_quantum_advantage' in metrics
        
        assert isinstance(metrics['average_reward'], float)
        assert isinstance(metrics['buffer_size'], int)


class TestQuantumTrainingSystem:
    """Test suite for quantum training system."""
    
    def test_system_creation(self):
        """Test quantum training system creation."""
        system = create_quantum_enhanced_training_system(observation_dim=20, action_dim=6)
        
        required_components = ['agent', 'training_config', 'quantum_config', 'train_step']
        
        for component in required_components:
            assert component in system
        
        assert system['system_type'] == 'quantum_enhanced_plasma_control'
        assert isinstance(system['agent'], QuantumEnhancedSAC)
        assert system['agent'].quantum_enhancement is True
    
    def test_training_step(self):
        """Test single training step."""
        system = create_quantum_enhanced_training_system(observation_dim=15, action_dim=5)
        
        observation = [random.uniform(-1, 1) for _ in range(15)]
        
        action, metrics = system['train_step'](observation)
        
        assert isinstance(action, list)
        assert len(action) == 5
        assert isinstance(metrics, dict)
        
        for a in action:
            assert -1.0 <= a <= 1.0
    
    def test_system_configuration(self):
        """Test system configuration parameters."""
        system = create_quantum_enhanced_training_system()
        
        training_config = system['training_config']
        quantum_config = system['quantum_config']
        
        # Check training configuration
        assert training_config['total_timesteps'] > 0
        assert training_config['batch_size'] > 0
        assert training_config['eval_frequency'] > 0
        
        # Check quantum configuration
        assert isinstance(quantum_config['coherence_optimization'], bool)
        assert isinstance(quantum_config['entanglement_scheduling'], bool)
        assert 0.0 < quantum_config['quantum_advantage_threshold'] < 1.0


class TestQuantumAdvantageValidation:
    """Test suite for quantum advantage validation."""
    
    def test_validation_execution(self):
        """Test quantum advantage validation."""
        # Use small number of trials for testing
        results = validate_quantum_advantage(n_trials=10)
        
        required_keys = [
            'classical_mean_performance', 'quantum_mean_performance',
            'relative_improvement', 'quantum_advantage_significant', 'n_trials'
        ]
        
        for key in required_keys:
            assert key in results
        
        assert results['n_trials'] == 10
        assert isinstance(results['classical_mean_performance'], float)
        assert isinstance(results['quantum_mean_performance'], float)
        assert isinstance(results['relative_improvement'], float)
        assert isinstance(results['quantum_advantage_significant'], bool)
    
    def test_performance_comparison(self):
        """Test performance comparison logic."""
        results = validate_quantum_advantage(n_trials=5)
        
        classical_perf = results['classical_mean_performance']
        quantum_perf = results['quantum_mean_performance']
        improvement = results['relative_improvement']
        
        if classical_perf != 0:
            expected_improvement = (quantum_perf - classical_perf) / classical_perf
            assert abs(improvement - expected_improvement) < 1e-6
        
        # Significance threshold should be at 5%
        if abs(improvement) > 0.05:
            assert results['quantum_advantage_significant'] is True


class TestQuantumIntegration:
    """Integration tests for quantum plasma control system."""
    
    def test_end_to_end_control(self):
        """Test end-to-end quantum control process."""
        # Create quantum controller
        controller = QuantumPlasmaController(n_qubits=6)
        
        # Simulate multiple control steps
        plasma_observations = [
            [2.0, 0.025, 1.5e19, 12.0, 0.95],  # Normal operation
            [2.5, 0.035, 1.8e19, 15.0, 0.90],  # Higher performance
            [1.8, 0.020, 1.2e19, 10.0, 0.98]   # Conservative operation
        ]
        
        previous_control = [0.0] * 8
        
        for i, obs in enumerate(plasma_observations):
            optimal_control, quantum_advantage = controller.quantum_plasma_evolution(
                obs, previous_control
            )
            
            # Verify control output
            assert len(optimal_control) >= 4  # At least action dimension
            assert all(-1.0 <= c <= 1.0 for c in optimal_control)
            assert 0.0 <= quantum_advantage <= 1.0
            
            # Update for next iteration
            previous_control = optimal_control[:8]
            
            # Check that measurement history is building
            assert len(controller.measurement_history) == i + 1
    
    def test_quantum_enhanced_training_workflow(self):
        """Test complete quantum-enhanced training workflow."""
        system = create_quantum_enhanced_training_system(observation_dim=25, action_dim=8)
        agent = system['agent']
        
        # Simulate training episode
        episode_length = 50
        episode_rewards = []
        
        for step in range(episode_length):
            # Generate observation
            observation = [random.uniform(-2, 2) for _ in range(25)]
            
            # Get action from agent
            action = agent.select_action(observation, training=True)
            
            # Simulate environment step
            reward = random.uniform(-1, 1)
            next_observation = [obs + random.uniform(-0.1, 0.1) for obs in observation]
            done = step == episode_length - 1
            
            # Store transition
            agent.store_transition(observation, action, reward, next_observation, done)
            episode_rewards.append(reward)
            
            # Periodically update agent
            if step > 10 and step % 5 == 0:
                agent.update(batch_size=min(8, len(agent.replay_buffer)))
        
        # Verify training progress
        assert len(agent.replay_buffer) == episode_length
        assert len(agent.quantum_advantages) == episode_length
        
        # Check quantum metrics
        metrics = agent.get_metrics()
        assert 'quantum_quantum_coherence' in metrics
        assert 'average_quantum_advantage' in metrics
        
        final_quantum_advantage = metrics['average_quantum_advantage']
        assert 0.0 <= final_quantum_advantage <= 1.0
    
    def test_quantum_state_persistence(self):
        """Test quantum state persistence across operations."""
        controller = QuantumPlasmaController(n_qubits=4)
        
        # Perform multiple quantum evolutions
        initial_metrics = controller.quantum_control_metrics()
        
        for _ in range(10):
            obs = [random.uniform(-1, 1) for _ in range(10)]
            action = [random.uniform(-1, 1) for _ in range(4)]
            
            controller.quantum_plasma_evolution(obs, action)
        
        final_metrics = controller.quantum_control_metrics()
        
        # Quantum state should have evolved
        assert final_metrics['quantum_coherence'] >= 0  # Should not be negative
        assert len(controller.measurement_history) == 10
        
        # Entanglement should still exist
        assert final_metrics['entanglement_entropy'] >= 0


# Integration fixtures and test runners
@pytest.fixture
def quantum_controller():
    """Fixture for quantum plasma controller."""
    return QuantumPlasmaController(n_qubits=5, coherence_time=1e-3)


@pytest.fixture
def quantum_agent():
    """Fixture for quantum-enhanced SAC agent."""
    return QuantumEnhancedSAC(observation_dim=12, action_dim=6, quantum_enhancement=True)


@pytest.fixture
def quantum_training_system():
    """Fixture for quantum training system."""
    return create_quantum_enhanced_training_system(observation_dim=20, action_dim=8)


def test_quantum_system_stress(quantum_controller, quantum_agent):
    """Stress test for quantum systems."""
    # Test controller under high load
    for i in range(100):
        obs = [random.uniform(-5, 5) for _ in range(15)]
        action = [random.uniform(-2, 2) for _ in range(8)]
        
        try:
            optimal_control, quantum_advantage = quantum_controller.quantum_plasma_evolution(obs, action)
            assert len(optimal_control) > 0
            assert 0.0 <= quantum_advantage <= 1.0
        except Exception as e:
            pytest.fail(f"Quantum controller failed under stress at iteration {i}: {e}")
    
    # Test agent under high load
    for i in range(50):
        obs = [random.uniform(-3, 3) for _ in range(12)]
        
        try:
            action = quantum_agent.select_action(obs, training=True)
            assert len(action) == 6
            assert all(-1.0 <= a <= 1.0 for a in action)
        except Exception as e:
            pytest.fail(f"Quantum agent failed under stress at iteration {i}: {e}")


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v", "--tb=short"])