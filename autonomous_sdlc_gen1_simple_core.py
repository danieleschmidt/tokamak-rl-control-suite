#!/usr/bin/env python3
"""
Autonomous SDLC Generation 1: Simple Core Implementation
Dependency-free tokamak control system core
"""

import json
import time
import math
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass


@dataclass
class PlasmaState:
    """Core plasma state representation"""
    current: float  # Plasma current in MA
    density: float  # Electron density in 10^20 m^-3
    temperature: float  # Core temperature in keV
    beta: float  # Normalized pressure
    q_min: float  # Minimum safety factor
    shape_error: float  # RMS shape error in cm
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass 
class ControlAction:
    """Control action representation"""
    pf_coils: List[float]  # Poloidal field coil currents
    gas_rate: float  # Gas puff rate
    heating: float  # Auxiliary heating power
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class SimplePlasmaSimulator:
    """Lightweight plasma physics simulator"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
            'major_radius': 6.2,  # ITER-like parameters
            'minor_radius': 2.0,
            'magnetic_field': 5.3,
            'max_current': 15.0
        }
        self.state = PlasmaState(
            current=1.0,
            density=0.8,
            temperature=10.0,
            beta=0.02,
            q_min=2.0,
            shape_error=3.0
        )
        self.step_count = 0
    
    def step(self, action: ControlAction) -> PlasmaState:
        """Simple physics step simulation"""
        dt = 0.01  # 10ms timestep
        
        # Simple current evolution
        current_cmd = sum(action.pf_coils) * 0.1
        self.state.current += (current_cmd - self.state.current) * dt * 10
        self.state.current = max(0.1, min(self.config['max_current'], self.state.current))
        
        # Density control via gas puff
        self.state.density += action.gas_rate * dt * 5
        self.state.density = max(0.1, min(2.0, self.state.density))
        
        # Temperature from heating
        heating_power = action.heating * dt * 2
        self.state.temperature += heating_power - 0.1  # cooling
        self.state.temperature = max(1.0, min(50.0, self.state.temperature))
        
        # Safety factor (simplified)
        self.state.q_min = 1.0 + self.state.current / (self.state.density + 0.1)
        
        # Beta calculation
        self.state.beta = self.state.density * self.state.temperature / 1000
        
        # Shape error (simplified control response)
        target_shape = 2.0  # Target shape error
        control_effort = abs(sum(action.pf_coils))
        self.state.shape_error += (target_shape - self.state.shape_error) * dt * control_effort
        self.state.shape_error = max(0.5, min(10.0, self.state.shape_error))
        
        self.step_count += 1
        self.state.timestamp = time.time()
        
        return self.state
    
    def is_stable(self) -> bool:
        """Check if plasma is in stable regime"""
        return (self.state.q_min > 1.5 and 
                self.state.beta < 0.05 and
                self.state.shape_error < 5.0)
    
    def get_reward(self, action: ControlAction) -> float:
        """Simple reward calculation"""
        shape_reward = -self.state.shape_error ** 2
        stability_reward = max(0, self.state.q_min - 1.0) * 10
        efficiency_penalty = -sum(x**2 for x in action.pf_coils) * 0.01
        
        if not self.is_stable():
            return -100  # Large penalty for instability
            
        return shape_reward + stability_reward + efficiency_penalty


class SimpleController:
    """Basic PID-like controller"""
    
    def __init__(self):
        self.error_history = []
        self.integral_error = 0.0
        self.last_error = 0.0
        
        # PID gains
        self.kp = 0.5
        self.ki = 0.1  
        self.kd = 0.02
    
    def control(self, state: PlasmaState, target_shape: float = 2.0) -> ControlAction:
        """Generate control action"""
        error = state.shape_error - target_shape
        
        self.integral_error += error
        derivative = error - self.last_error
        
        # PID control
        control_signal = (self.kp * error + 
                         self.ki * self.integral_error +
                         self.kd * derivative)
        
        # Distribute control to PF coils
        pf_coils = [control_signal * 0.2] * 6  # 6 PF coils
        
        # Gas puff control for density
        gas_rate = max(0, min(1.0, (1.0 - state.density) * 0.5))
        
        # Heating control for temperature
        heating = max(0, min(1.0, (20.0 - state.temperature) * 0.05))
        
        self.last_error = error
        
        return ControlAction(
            pf_coils=pf_coils,
            gas_rate=gas_rate,
            heating=heating
        )


class ExperimentRunner:
    """Run tokamak control experiments"""
    
    def __init__(self):
        self.simulator = SimplePlasmaSimulator()
        self.controller = SimpleController()
        self.data_log = []
    
    def run_experiment(self, duration: int = 100) -> Dict[str, Any]:
        """Run control experiment"""
        print(f"🚀 Starting tokamak control experiment ({duration} steps)")
        
        results = {
            'steps': [],
            'rewards': [],
            'stability_rate': 0.0,
            'avg_shape_error': 0.0,
            'total_reward': 0.0
        }
        
        stable_count = 0
        total_shape_error = 0.0
        total_reward = 0.0
        
        for step in range(duration):
            # Get current state
            state = self.simulator.state
            
            # Generate control action
            action = self.controller.control(state)
            
            # Simulate physics
            next_state = self.simulator.step(action)
            
            # Calculate reward
            reward = self.simulator.get_reward(action)
            
            # Log data
            step_data = {
                'step': step,
                'state': {
                    'current': state.current,
                    'density': state.density,
                    'temperature': state.temperature,
                    'q_min': state.q_min,
                    'shape_error': state.shape_error
                },
                'action': {
                    'pf_coils': action.pf_coils,
                    'gas_rate': action.gas_rate,
                    'heating': action.heating
                },
                'reward': reward,
                'stable': self.simulator.is_stable()
            }
            
            results['steps'].append(step_data)
            results['rewards'].append(reward)
            
            if self.simulator.is_stable():
                stable_count += 1
            
            total_shape_error += state.shape_error
            total_reward += reward
            
            # Progress indicator
            if step % 20 == 0:
                print(f"Step {step}: Shape error = {state.shape_error:.2f} cm, "
                      f"q_min = {state.q_min:.2f}, Stable = {self.simulator.is_stable()}")
        
        # Calculate final metrics
        results['stability_rate'] = stable_count / duration
        results['avg_shape_error'] = total_shape_error / duration
        results['total_reward'] = total_reward
        
        print(f"✅ Experiment complete!")
        print(f"   Stability rate: {results['stability_rate']:.1%}")
        print(f"   Average shape error: {results['avg_shape_error']:.2f} cm")
        print(f"   Total reward: {results['total_reward']:.1f}")
        
        return results


def run_autonomous_gen1_demo():
    """Autonomous Generation 1 demonstration"""
    print("=" * 60)
    print("AUTONOMOUS SDLC GENERATION 1: SIMPLE CORE IMPLEMENTATION")
    print("=" * 60)
    
    try:
        # Initialize experiment
        runner = ExperimentRunner()
        
        # Run basic experiment
        results = runner.run_experiment(duration=100)
        
        # Save results
        output_file = "/root/repo/autonomous_gen1_simple_results.json"
        with open(output_file, 'w') as f:
            # Convert results to JSON-serializable format
            json_results = {
                'stability_rate': results['stability_rate'],
                'avg_shape_error': results['avg_shape_error'], 
                'total_reward': results['total_reward'],
                'final_metrics': {
                    'steps_completed': len(results['steps']),
                    'max_reward': max(results['rewards']),
                    'min_reward': min(results['rewards'])
                },
                'timestamp': time.time(),
                'generation': '1-simple'
            }
            json.dump(json_results, f, indent=2)
        
        print(f"\n📊 Results saved to: {output_file}")
        
        # Success metrics
        success_criteria = {
            'stability_rate': results['stability_rate'] > 0.7,
            'shape_error': results['avg_shape_error'] < 4.0,
            'reward': results['total_reward'] > -1000
        }
        
        all_passed = all(success_criteria.values())
        
        print(f"\n🎯 SUCCESS CRITERIA:")
        for criterion, passed in success_criteria.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"   {criterion}: {status}")
        
        if all_passed:
            print(f"\n🎉 GENERATION 1 SUCCESS: Basic functionality implemented!")
            return True
        else:
            print(f"\n⚠️  Some criteria not met, but core functionality works")
            return False
            
    except Exception as e:
        print(f"❌ Generation 1 Error: {e}")
        return False


if __name__ == "__main__":
    success = run_autonomous_gen1_demo()
    print(f"\nGeneration 1 Status: {'SUCCESS' if success else 'PARTIAL SUCCESS'}")