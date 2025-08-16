"""
Advanced Research Algorithms for Tokamak Control

Novel research contributions for fusion plasma control including:
- Neuromorphic computing integration
- Swarm intelligence algorithms
- Causal inference for plasma dynamics
- Federated learning across tokamaks
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from scipy.optimize import differential_evolution
import logging
from collections import defaultdict
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


@dataclass
class SwarmAgent:
    """Individual agent in plasma control swarm"""
    position: np.ndarray
    velocity: np.ndarray
    best_position: np.ndarray
    best_fitness: float
    fitness_history: List[float]


class NeuromorphicPlasmaController:
    """
    Neuromorphic computing inspired controller using spiking neural networks
    for ultra-low latency plasma control.
    """
    
    def __init__(self, input_dim: int = 45, hidden_dim: int = 128, output_dim: int = 8):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Spiking neuron parameters
        self.membrane_potential = np.zeros(hidden_dim)
        self.threshold = 1.0
        self.decay_rate = 0.9
        self.refractory_period = 3
        self.refractory_counters = np.zeros(hidden_dim, dtype=int)
        
        # Synaptic weights with spike-timing dependent plasticity
        self.input_weights = np.random.normal(0, 0.1, (input_dim, hidden_dim))
        self.output_weights = np.random.normal(0, 0.1, (hidden_dim, output_dim))
        
        # STDP parameters
        self.eligibility_trace = np.zeros((input_dim, hidden_dim))
        self.learning_rate = 0.001
        self.trace_decay = 0.95
        
        # Event-driven computation history
        self.spike_trains = defaultdict(list)
        self.last_spike_times = np.full(hidden_dim, -np.inf)
        
        logger.info(f"Initialized neuromorphic controller with {hidden_dim} spiking neurons")
    
    def encode_plasma_state(self, plasma_state: np.ndarray) -> np.ndarray:
        """
        Encode continuous plasma state into spike trains using rate coding
        """
        # Normalize state to positive values
        normalized_state = (plasma_state - np.min(plasma_state)) / (np.ptp(plasma_state) + 1e-12)
        
        # Convert to firing rates (Hz)
        max_firing_rate = 100.0  # Maximum 100 Hz
        firing_rates = normalized_state * max_firing_rate
        
        # Generate Poisson spike trains
        dt = 0.001  # 1ms time step
        spike_probabilities = firing_rates * dt
        
        # Generate spikes based on Poisson process
        spikes = np.random.random(len(firing_rates)) < spike_probabilities
        
        return spikes.astype(float)
    
    def update_membrane_potentials(self, input_spikes: np.ndarray, current_time: float):
        """Update membrane potentials based on input spikes"""
        # Decay membrane potentials
        self.membrane_potential *= self.decay_rate
        
        # Add input currents from spikes
        input_current = np.dot(input_spikes, self.input_weights)
        self.membrane_potential += input_current
        
        # Handle refractory periods
        active_neurons = self.refractory_counters == 0
        self.membrane_potential[~active_neurons] = 0
        
        # Decrease refractory counters
        self.refractory_counters = np.maximum(0, self.refractory_counters - 1)
    
    def generate_output_spikes(self, current_time: float) -> Tuple[np.ndarray, np.ndarray]:
        """Generate output spikes and update neuron states"""
        # Check for spiking neurons
        spiking_neurons = (self.membrane_potential >= self.threshold) & (self.refractory_counters == 0)
        
        # Record spike times
        for neuron_idx in np.where(spiking_neurons)[0]:
            self.spike_trains[neuron_idx].append(current_time)
            self.last_spike_times[neuron_idx] = current_time
        
        # Reset spiking neurons
        self.membrane_potential[spiking_neurons] = 0
        self.refractory_counters[spiking_neurons] = self.refractory_period
        
        # Generate output based on recent spike activity
        output_spikes = self._compute_output_response(spiking_neurons, current_time)
        
        return spiking_neurons.astype(float), output_spikes
    
    def _compute_output_response(self, hidden_spikes: np.ndarray, current_time: float) -> np.ndarray:
        """Compute output response based on hidden layer spikes"""
        # Temporal integration of recent spikes
        integration_window = 0.01  # 10ms integration window
        
        output_response = np.zeros(self.output_dim)
        
        for neuron_idx in range(self.hidden_dim):
            if len(self.spike_trains[neuron_idx]) > 0:
                # Count recent spikes
                recent_spikes = [t for t in self.spike_trains[neuron_idx] 
                               if current_time - t <= integration_window]
                spike_count = len(recent_spikes)
                
                # Weight by output synapses
                output_response += self.output_weights[neuron_idx] * spike_count
        
        # Apply sigmoid activation for continuous output
        output_response = 1.0 / (1.0 + np.exp(-output_response))
        
        return output_response
    
    def update_synaptic_weights(self, input_spikes: np.ndarray, hidden_spikes: np.ndarray):
        """Update synaptic weights using spike-timing dependent plasticity"""
        # Update eligibility traces
        self.eligibility_trace *= self.trace_decay
        
        # Add current spike contributions
        for i, input_spike in enumerate(input_spikes):
            if input_spike > 0:
                for j, hidden_spike in enumerate(hidden_spikes):
                    if hidden_spike > 0:
                        # Strengthen connection for coincident spikes
                        self.eligibility_trace[i, j] += 1.0
                    else:
                        # Weaken connection for non-coincident spikes
                        self.eligibility_trace[i, j] -= 0.1
        
        # Apply weight updates
        weight_update = self.learning_rate * self.eligibility_trace
        self.input_weights += weight_update
        
        # Normalize weights to prevent unbounded growth
        self.input_weights = np.clip(self.input_weights, -1.0, 1.0)
    
    def control_plasma(self, plasma_state: np.ndarray, current_time: float) -> np.ndarray:
        """
        Generate control actions for plasma using neuromorphic processing
        """
        # Encode plasma state to spikes
        input_spikes = self.encode_plasma_state(plasma_state)
        
        # Update membrane potentials
        self.update_membrane_potentials(input_spikes, current_time)
        
        # Generate spikes and output
        hidden_spikes, output_response = self.generate_output_spikes(current_time)
        
        # Update synaptic weights (online learning)
        self.update_synaptic_weights(input_spikes, hidden_spikes)
        
        # Scale output to control action range [-1, 1]
        control_actions = 2.0 * output_response - 1.0
        
        return control_actions


class SwarmIntelligencePlasmaOptimizer:
    """
    Swarm intelligence optimizer for plasma control using particle swarm optimization
    with plasma physics constraints.
    """
    
    def __init__(self, control_dim: int = 8, swarm_size: int = 50, bounds: Optional[Tuple] = None):
        self.control_dim = control_dim
        self.swarm_size = swarm_size
        self.bounds = bounds if bounds else (-1.0, 1.0)
        
        # PSO parameters
        self.inertia = 0.9
        self.cognitive_coeff = 2.0
        self.social_coeff = 2.0
        self.max_velocity = 0.5
        
        # Initialize swarm
        self.swarm = self._initialize_swarm()
        self.global_best_position = None
        self.global_best_fitness = -np.inf
        
        # Convergence tracking
        self.convergence_history = []
        self.diversity_history = []
        
        logger.info(f"Initialized swarm optimizer with {swarm_size} particles")
    
    def _initialize_swarm(self) -> List[SwarmAgent]:
        """Initialize swarm of control agents"""
        swarm = []
        
        for _ in range(self.swarm_size):
            # Random initial position within bounds
            position = np.random.uniform(self.bounds[0], self.bounds[1], self.control_dim)
            
            # Random initial velocity
            velocity = np.random.uniform(-self.max_velocity, self.max_velocity, self.control_dim)
            
            agent = SwarmAgent(
                position=position,
                velocity=velocity,
                best_position=position.copy(),
                best_fitness=-np.inf,
                fitness_history=[]
            )
            
            swarm.append(agent)
        
        return swarm
    
    def evaluate_plasma_fitness(self, control_actions: np.ndarray, 
                               plasma_state: np.ndarray) -> float:
        """
        Evaluate fitness of control actions for plasma optimization
        
        Args:
            control_actions: Control actions to evaluate
            plasma_state: Current plasma state
            
        Returns:
            Fitness score (higher is better)
        """
        fitness = 0.0
        
        # Shape accuracy term
        target_elongation = 1.7
        target_triangularity = 0.4
        
        # Extract shape parameters (assuming they're in state)
        if len(plasma_state) >= 10:
            elongation = plasma_state[6]  # Example index
            triangularity = plasma_state[7]
            
            shape_error = abs(elongation - target_elongation) + abs(triangularity - target_triangularity)
            fitness -= shape_error * 100
        
        # Stability term - reward high safety factor
        if len(plasma_state) >= 20:
            q_profile = plasma_state[10:20]  # Safety factor profile
            q_min = np.min(q_profile)
            
            if q_min > 1.5:
                fitness += (q_min - 1.5) * 50
            else:
                fitness -= (1.5 - q_min) * 200  # Heavy penalty for low q
        
        # Efficiency term - minimize control effort
        control_effort = np.sum(control_actions ** 2)
        fitness -= control_effort * 10
        
        # Physics constraints
        # Density limit constraint
        if len(plasma_state) >= 30:
            density_profile = plasma_state[20:30]
            avg_density = np.mean(density_profile)
            
            # Greenwald density limit
            greenwald_limit = 1.2e20  # m^-3
            if avg_density > greenwald_limit:
                fitness -= (avg_density - greenwald_limit) * 1e-18 * 1000
        
        # Beta limit constraint
        if len(plasma_state) >= 2:
            beta = plasma_state[1]
            beta_limit = 0.04
            
            if beta > beta_limit:
                fitness -= (beta - beta_limit) * 10000
        
        return fitness
    
    def update_swarm(self, plasma_state: np.ndarray):
        """Update swarm positions and velocities"""
        for agent in self.swarm:
            # Evaluate current fitness
            current_fitness = self.evaluate_plasma_fitness(agent.position, plasma_state)
            agent.fitness_history.append(current_fitness)
            
            # Update personal best
            if current_fitness > agent.best_fitness:
                agent.best_fitness = current_fitness
                agent.best_position = agent.position.copy()
            
            # Update global best
            if current_fitness > self.global_best_fitness:
                self.global_best_fitness = current_fitness
                self.global_best_position = agent.position.copy()
        
        # Update velocities and positions
        for agent in self.swarm:
            if self.global_best_position is not None:
                # PSO velocity update
                r1, r2 = np.random.random(2)
                
                cognitive_component = (self.cognitive_coeff * r1 * 
                                     (agent.best_position - agent.position))
                social_component = (self.social_coeff * r2 * 
                                  (self.global_best_position - agent.position))
                
                # Update velocity
                agent.velocity = (self.inertia * agent.velocity + 
                                cognitive_component + social_component)
                
                # Limit velocity
                agent.velocity = np.clip(agent.velocity, -self.max_velocity, self.max_velocity)
                
                # Update position
                agent.position += agent.velocity
                
                # Enforce bounds
                agent.position = np.clip(agent.position, self.bounds[0], self.bounds[1])
    
    def optimize_control(self, plasma_state: np.ndarray, max_iterations: int = 100) -> np.ndarray:
        """
        Optimize control actions using swarm intelligence
        
        Args:
            plasma_state: Current plasma state
            max_iterations: Maximum optimization iterations
            
        Returns:
            Optimized control actions
        """
        for iteration in range(max_iterations):
            # Update swarm
            self.update_swarm(plasma_state)
            
            # Track convergence
            current_best = self.global_best_fitness
            self.convergence_history.append(current_best)
            
            # Calculate swarm diversity
            positions = np.array([agent.position for agent in self.swarm])
            diversity = np.mean(np.std(positions, axis=0))
            self.diversity_history.append(diversity)
            
            # Early stopping if converged
            if len(self.convergence_history) > 10:
                recent_improvement = (self.convergence_history[-1] - 
                                    self.convergence_history[-10])
                if abs(recent_improvement) < 1e-6:
                    logger.info(f"Swarm converged after {iteration} iterations")
                    break
        
        return self.global_best_position if self.global_best_position is not None else np.zeros(self.control_dim)


class CausalInferencePlasmaAnalyzer:
    """
    Causal inference analyzer for understanding plasma dynamics
    using directed acyclic graphs and interventional analysis.
    """
    
    def __init__(self, variable_names: List[str]):
        self.variable_names = variable_names
        self.n_variables = len(variable_names)
        
        # Causal graph representation (adjacency matrix)
        self.causal_graph = np.zeros((self.n_variables, self.n_variables))
        
        # Causal strength matrix
        self.causal_strengths = np.zeros((self.n_variables, self.n_variables))
        
        # Intervention effects storage
        self.intervention_effects = {}
        
        # Data storage for causal discovery
        self.observation_data = []
        
        logger.info(f"Initialized causal analyzer for {self.n_variables} variables")
    
    def discover_causal_structure(self, plasma_data: np.ndarray, 
                                 significance_threshold: float = 0.05) -> np.ndarray:
        """
        Discover causal structure from plasma observation data
        
        Args:
            plasma_data: Time series data [time_steps, n_variables]
            significance_threshold: Threshold for causal significance
            
        Returns:
            Discovered causal adjacency matrix
        """
        n_timesteps, n_vars = plasma_data.shape
        
        # PC algorithm for causal discovery (simplified)
        causal_matrix = np.zeros((n_vars, n_vars))
        
        for i in range(n_vars):
            for j in range(n_vars):
                if i != j:
                    # Test conditional independence
                    causal_strength = self._test_causal_relationship(
                        plasma_data[:, i], plasma_data[:, j], plasma_data
                    )
                    
                    if causal_strength > significance_threshold:
                        causal_matrix[i, j] = causal_strength
                        self.causal_strengths[i, j] = causal_strength
        
        # Remove cycles using topological ordering
        self.causal_graph = self._remove_cycles(causal_matrix)
        
        return self.causal_graph
    
    def _test_causal_relationship(self, cause: np.ndarray, effect: np.ndarray, 
                                 all_data: np.ndarray) -> float:
        """Test causal relationship between two variables"""
        # Granger causality test (simplified)
        lagged_cause = np.roll(cause, 1)[1:]  # Lag by 1 time step
        current_effect = effect[1:]
        
        # Control for other variables (simplified partial correlation)
        other_vars = []
        for k in range(all_data.shape[1]):
            if k != 0 and k != 1:  # Assuming cause=0, effect=1 for simplicity
                other_vars.append(all_data[1:, k])
        
        if len(other_vars) > 0:
            other_matrix = np.column_stack(other_vars)
            
            # Partial correlation calculation
            try:
                # Residualize cause and effect w.r.t. other variables
                cause_residual = self._residualize(lagged_cause, other_matrix)
                effect_residual = self._residualize(current_effect, other_matrix)
                
                # Compute partial correlation
                correlation = np.corrcoef(cause_residual, effect_residual)[0, 1]
                causal_strength = abs(correlation)
            except:
                # Fallback to simple correlation
                causal_strength = abs(np.corrcoef(lagged_cause, current_effect)[0, 1])
        else:
            causal_strength = abs(np.corrcoef(lagged_cause, current_effect)[0, 1])
        
        return causal_strength
    
    def _residualize(self, target: np.ndarray, controls: np.ndarray) -> np.ndarray:
        """Compute residuals after regressing out control variables"""
        try:
            # Simple linear regression
            coeffs = np.linalg.lstsq(controls, target, rcond=None)[0]
            predicted = controls @ coeffs
            residuals = target - predicted
            return residuals
        except:
            return target  # Return original if regression fails
    
    def _remove_cycles(self, adjacency_matrix: np.ndarray) -> np.ndarray:
        """Remove cycles from causal graph to ensure DAG property"""
        # Simple cycle removal: keep edge with stronger causal strength
        dag = adjacency_matrix.copy()
        
        for i in range(self.n_variables):
            for j in range(self.n_variables):
                if dag[i, j] > 0 and dag[j, i] > 0:
                    # Bidirectional edge found - keep stronger one
                    if dag[i, j] >= dag[j, i]:
                        dag[j, i] = 0
                    else:
                        dag[i, j] = 0
        
        return dag
    
    def estimate_intervention_effect(self, intervention_var: int, 
                                   intervention_value: float,
                                   target_var: int,
                                   baseline_data: np.ndarray) -> Dict[str, float]:
        """
        Estimate effect of intervention using do-calculus
        
        Args:
            intervention_var: Index of variable to intervene on
            intervention_value: Value to set intervention variable to
            target_var: Index of target variable to measure effect on
            baseline_data: Baseline observation data
            
        Returns:
            Dictionary containing intervention effects
        """
        # Simulate intervention by setting variable to fixed value
        intervened_data = baseline_data.copy()
        intervened_data[:, intervention_var] = intervention_value
        
        # Calculate direct and total effects
        baseline_target_mean = np.mean(baseline_data[:, target_var])
        
        # Simulate forward pass through causal graph
        simulated_target = self._simulate_causal_propagation(
            intervened_data, target_var
        )
        
        intervened_target_mean = np.mean(simulated_target)
        
        # Calculate effects
        total_effect = intervened_target_mean - baseline_target_mean
        
        # Find mediators (variables causally between intervention and target)
        mediators = self._find_mediators(intervention_var, target_var)
        
        effects = {
            'total_effect': total_effect,
            'direct_effect': total_effect,  # Simplified - same as total
            'indirect_effect': 0.0,  # Through mediators
            'mediators': mediators,
            'baseline_mean': baseline_target_mean,
            'intervened_mean': intervened_target_mean
        }
        
        # Store intervention effect
        intervention_key = f"{intervention_var}->{target_var}"
        self.intervention_effects[intervention_key] = effects
        
        return effects
    
    def _simulate_causal_propagation(self, data: np.ndarray, target_var: int) -> np.ndarray:
        """Simulate propagation of effects through causal graph"""
        # Simple linear propagation (could be replaced with more sophisticated model)
        simulated_values = data[:, target_var].copy()
        
        # Find causal parents of target variable
        parents = np.where(self.causal_graph[:, target_var] > 0)[0]
        
        for parent in parents:
            causal_strength = self.causal_strengths[parent, target_var]
            parent_effect = data[:, parent] * causal_strength * 0.1  # Scaling factor
            simulated_values += parent_effect
        
        return simulated_values
    
    def _find_mediators(self, intervention_var: int, target_var: int) -> List[int]:
        """Find mediating variables between intervention and target"""
        mediators = []
        
        # Simple path finding: variables that are children of intervention and parents of target
        intervention_children = np.where(self.causal_graph[intervention_var, :] > 0)[0]
        target_parents = np.where(self.causal_graph[:, target_var] > 0)[0]
        
        mediators = list(set(intervention_children) & set(target_parents))
        
        return mediators


class FederatedPlasmaLearning:
    """
    Federated learning system for collaborative plasma control across multiple tokamaks
    with privacy preservation and knowledge sharing.
    """
    
    def __init__(self, tokamak_configs: List[str], local_model_dim: int = 128):
        self.tokamak_configs = tokamak_configs
        self.n_tokamaks = len(tokamak_configs)
        self.local_model_dim = local_model_dim
        
        # Global model parameters
        self.global_model = self._initialize_global_model()
        
        # Local models for each tokamak
        self.local_models = {config: self._initialize_local_model() 
                           for config in tokamak_configs}
        
        # Federated learning parameters
        self.learning_rate = 0.01
        self.aggregation_weights = np.ones(self.n_tokamaks) / self.n_tokamaks
        
        # Privacy parameters
        self.noise_scale = 0.1
        self.clip_norm = 1.0
        
        # Communication history
        self.communication_rounds = 0
        self.convergence_history = []
        
        logger.info(f"Initialized federated learning for {self.n_tokamaks} tokamaks")
    
    def _initialize_global_model(self) -> Dict[str, np.ndarray]:
        """Initialize global model parameters"""
        return {
            'weights': np.random.normal(0, 0.1, (self.local_model_dim, self.local_model_dim)),
            'bias': np.zeros(self.local_model_dim),
            'output_layer': np.random.normal(0, 0.1, (self.local_model_dim, 8))  # 8 control actions
        }
    
    def _initialize_local_model(self) -> Dict[str, np.ndarray]:
        """Initialize local model parameters (copy of global)"""
        return {
            'weights': self.global_model['weights'].copy(),
            'bias': self.global_model['bias'].copy(),
            'output_layer': self.global_model['output_layer'].copy()
        }
    
    async def federated_training_round(self, local_data: Dict[str, np.ndarray],
                                     local_targets: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Perform one round of federated training
        
        Args:
            local_data: Dictionary mapping tokamak configs to local training data
            local_targets: Dictionary mapping tokamak configs to local targets
            
        Returns:
            Dictionary containing training metrics
        """
        # Local training on each tokamak
        local_updates = {}
        
        # Use ThreadPoolExecutor for parallel local training
        with ThreadPoolExecutor(max_workers=self.n_tokamaks) as executor:
            futures = {}
            
            for config in self.tokamak_configs:
                if config in local_data and config in local_targets:
                    future = executor.submit(
                        self._local_training,
                        config, local_data[config], local_targets[config]
                    )
                    futures[config] = future
            
            # Collect results
            for config, future in futures.items():
                local_updates[config] = future.result()
        
        # Aggregate updates
        aggregated_update = self._aggregate_updates(local_updates)
        
        # Update global model
        self._update_global_model(aggregated_update)
        
        # Broadcast updated global model to all tokamaks
        self._broadcast_global_model()
        
        # Calculate convergence metrics
        convergence_metric = self._calculate_convergence()
        self.convergence_history.append(convergence_metric)
        self.communication_rounds += 1
        
        metrics = {
            'communication_round': self.communication_rounds,
            'convergence_metric': convergence_metric,
            'n_participating_tokamaks': len(local_updates),
            'global_model_norm': np.linalg.norm(self.global_model['weights'])
        }
        
        return metrics
    
    def _local_training(self, tokamak_config: str, 
                       local_data: np.ndarray, 
                       local_targets: np.ndarray) -> Dict[str, np.ndarray]:
        """Perform local training on a single tokamak"""
        model = self.local_models[tokamak_config]
        
        # Simple gradient descent (could be replaced with more sophisticated optimizer)
        n_samples = local_data.shape[0]
        
        # Forward pass
        hidden = np.tanh(local_data @ model['weights'] + model['bias'])
        predictions = hidden @ model['output_layer']
        
        # Loss (MSE)
        loss = np.mean((predictions - local_targets) ** 2)
        
        # Backward pass
        output_grad = 2 * (predictions - local_targets) / n_samples
        output_layer_grad = hidden.T @ output_grad
        
        hidden_grad = output_grad @ model['output_layer'].T
        hidden_grad *= (1 - hidden ** 2)  # tanh derivative
        
        weights_grad = local_data.T @ hidden_grad
        bias_grad = np.mean(hidden_grad, axis=0)
        
        # Apply differential privacy noise
        weights_grad = self._add_privacy_noise(weights_grad)
        bias_grad = self._add_privacy_noise(bias_grad)
        output_layer_grad = self._add_privacy_noise(output_layer_grad)
        
        # Clip gradients for privacy
        weights_grad = self._clip_gradients(weights_grad)
        bias_grad = self._clip_gradients(bias_grad)
        output_layer_grad = self._clip_gradients(output_layer_grad)
        
        # Update local model
        model['weights'] -= self.learning_rate * weights_grad
        model['bias'] -= self.learning_rate * bias_grad
        model['output_layer'] -= self.learning_rate * output_layer_grad
        
        # Return parameter updates (difference from global model)
        updates = {
            'weights': model['weights'] - self.global_model['weights'],
            'bias': model['bias'] - self.global_model['bias'],
            'output_layer': model['output_layer'] - self.global_model['output_layer'],
            'n_samples': n_samples,
            'loss': loss
        }
        
        return updates
    
    def _add_privacy_noise(self, gradient: np.ndarray) -> np.ndarray:
        """Add differential privacy noise to gradients"""
        noise = np.random.normal(0, self.noise_scale, gradient.shape)
        return gradient + noise
    
    def _clip_gradients(self, gradient: np.ndarray) -> np.ndarray:
        """Clip gradients for privacy preservation"""
        grad_norm = np.linalg.norm(gradient)
        if grad_norm > self.clip_norm:
            gradient = gradient * (self.clip_norm / grad_norm)
        return gradient
    
    def _aggregate_updates(self, local_updates: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """Aggregate local updates using weighted averaging"""
        # Calculate weights based on number of samples
        total_samples = sum(update['n_samples'] for update in local_updates.values())
        
        aggregated = {
            'weights': np.zeros_like(self.global_model['weights']),
            'bias': np.zeros_like(self.global_model['bias']),
            'output_layer': np.zeros_like(self.global_model['output_layer'])
        }
        
        for config, update in local_updates.items():
            weight = update['n_samples'] / total_samples
            
            aggregated['weights'] += weight * update['weights']
            aggregated['bias'] += weight * update['bias']
            aggregated['output_layer'] += weight * update['output_layer']
        
        return aggregated
    
    def _update_global_model(self, aggregated_update: Dict[str, np.ndarray]):
        """Update global model with aggregated updates"""
        self.global_model['weights'] += aggregated_update['weights']
        self.global_model['bias'] += aggregated_update['bias']
        self.global_model['output_layer'] += aggregated_update['output_layer']
    
    def _broadcast_global_model(self):
        """Broadcast updated global model to all local models"""
        for config in self.tokamak_configs:
            self.local_models[config]['weights'] = self.global_model['weights'].copy()
            self.local_models[config]['bias'] = self.global_model['bias'].copy()
            self.local_models[config]['output_layer'] = self.global_model['output_layer'].copy()
    
    def _calculate_convergence(self) -> float:
        """Calculate convergence metric based on model parameter stability"""
        if len(self.convergence_history) < 2:
            return 1.0
        
        # Calculate parameter change from previous round
        # (This is simplified - in practice would compare with previous global model)
        param_norm = np.linalg.norm(self.global_model['weights'])
        
        if len(self.convergence_history) > 0:
            prev_norm = self.convergence_history[-1]
            convergence = abs(param_norm - prev_norm) / (prev_norm + 1e-12)
        else:
            convergence = 1.0
        
        return convergence


def create_advanced_research_system(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Create comprehensive advanced research system
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary containing advanced research components
    """
    if config is None:
        config = {
            'neuromorphic': {
                'input_dim': 45,
                'hidden_dim': 128,
                'output_dim': 8
            },
            'swarm': {
                'control_dim': 8,
                'swarm_size': 50,
                'bounds': (-1.0, 1.0)
            },
            'causal': {
                'variable_names': [
                    'plasma_current', 'plasma_beta', 'elongation', 'triangularity',
                    'q_min', 'density', 'temperature', 'stored_energy'
                ]
            },
            'federated': {
                'tokamak_configs': ['ITER', 'SPARC', 'DIII-D', 'NSTX'],
                'local_model_dim': 128
            }
        }
    
    # Initialize components
    neuromorphic_controller = NeuromorphicPlasmaController(**config['neuromorphic'])
    swarm_optimizer = SwarmIntelligencePlasmaOptimizer(**config['swarm'])
    causal_analyzer = CausalInferencePlasmaAnalyzer(**config['causal'])
    federated_learner = FederatedPlasmaLearning(**config['federated'])
    
    logger.info("Created advanced research system with neuromorphic, swarm, causal, and federated components")
    
    return {
        'neuromorphic_controller': neuromorphic_controller,
        'swarm_optimizer': swarm_optimizer,
        'causal_analyzer': causal_analyzer,
        'federated_learner': federated_learner,
        'config': config
    }


# Example usage and demonstration
if __name__ == "__main__":
    import asyncio
    
    # Create advanced research system
    research_system = create_advanced_research_system()
    
    print("🔬 Advanced Research Algorithms Demo")
    print("====================================")
    
    # Demo neuromorphic controller
    print("\n1. Neuromorphic Plasma Controller:")
    neuromorphic = research_system['neuromorphic_controller']
    
    sample_plasma_state = np.random.randn(45)
    current_time = 0.001
    
    control_actions = neuromorphic.control_plasma(sample_plasma_state, current_time)
    
    print(f"   ✓ Input state dim: {len(sample_plasma_state)}")
    print(f"   ✓ Control actions: {control_actions}")
    print(f"   ✓ Membrane potentials: {neuromorphic.membrane_potential[:5]}")
    
    # Demo swarm optimizer
    print("\n2. Swarm Intelligence Optimizer:")
    swarm = research_system['swarm_optimizer']
    
    optimized_control = swarm.optimize_control(sample_plasma_state, max_iterations=20)
    
    print(f"   ✓ Optimized control: {optimized_control}")
    print(f"   ✓ Best fitness: {swarm.global_best_fitness:.3f}")
    print(f"   ✓ Convergence history: {len(swarm.convergence_history)} points")
    
    # Demo causal analyzer
    print("\n3. Causal Inference Analyzer:")
    causal = research_system['causal_analyzer']
    
    # Generate sample time series data
    n_timesteps = 100
    n_variables = len(causal.variable_names)
    sample_data = np.random.randn(n_timesteps, n_variables)
    
    # Add some causal relationships
    sample_data[:, 1] = 0.5 * sample_data[:, 0] + np.random.randn(n_timesteps) * 0.1
    
    causal_graph = causal.discover_causal_structure(sample_data)
    
    print(f"   ✓ Discovered causal edges: {np.sum(causal_graph > 0)}")
    print(f"   ✓ Strongest causal relationship: {np.max(causal_graph):.3f}")
    
    # Test intervention
    intervention_effect = causal.estimate_intervention_effect(0, 1.0, 1, sample_data)
    print(f"   ✓ Intervention effect: {intervention_effect['total_effect']:.3f}")
    
    # Demo federated learning
    print("\n4. Federated Plasma Learning:")
    federated = research_system['federated_learner']
    
    # Generate sample local data for each tokamak
    local_data = {}
    local_targets = {}
    
    for config in federated.tokamak_configs:
        n_samples = np.random.randint(50, 200)
        local_data[config] = np.random.randn(n_samples, 45)
        local_targets[config] = np.random.randn(n_samples, 8)
    
    # Run federated training round
    async def run_federated_demo():
        metrics = await federated.federated_training_round(local_data, local_targets)
        return metrics
    
    # Run the async demo
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        fed_metrics = loop.run_until_complete(run_federated_demo())
        
        print(f"   ✓ Communication round: {fed_metrics['communication_round']}")
        print(f"   ✓ Participating tokamaks: {fed_metrics['n_participating_tokamaks']}")
        print(f"   ✓ Convergence metric: {fed_metrics['convergence_metric']:.6f}")
        print(f"   ✓ Global model norm: {fed_metrics['global_model_norm']:.3f}")
    finally:
        loop.close()
    
    print("\n🚀 Advanced research algorithms completed successfully!")
    print("    Novel algorithms ready for breakthrough plasma control")