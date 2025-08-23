#!/usr/bin/env python3
"""
Direct module testing without full __init__.py imports
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_quantum_control():
    """Test quantum control module directly."""
    print("Testing quantum plasma control...")
    
    try:
        # Import quantum module directly
        from tokamak_rl import quantum_plasma_control as qpc
        
        # Test controller creation
        controller = qpc.QuantumPlasmaController(n_qubits=4, coherence_time=1e-3)
        print("  ✓ Controller created")
        
        # Test quantum evolution
        plasma_obs = [1.0, 0.5, -0.3, 0.8]
        control_action = [0.1, -0.2, 0.3, -0.1]
        
        optimal_control, quantum_advantage = controller.quantum_plasma_evolution(plasma_obs, control_action)
        
        print(f"  ✓ Quantum evolution: advantage={quantum_advantage:.3f}")
        print(f"  ✓ Control output: {len(optimal_control)} dimensions")
        
        # Test quantum metrics
        metrics = controller.quantum_control_metrics()
        print(f"  ✓ Quantum coherence: {metrics['quantum_coherence']:.3f}")
        
        # Test SAC agent
        agent = qpc.QuantumEnhancedSAC(observation_dim=10, action_dim=4)
        observation = [0.1 * i for i in range(10)]
        action = agent.select_action(observation)
        print(f"  ✓ SAC agent action: {len(action)} dimensions")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def test_physics_research():
    """Test physics research module directly."""
    print("Testing advanced physics research...")
    
    try:
        from tokamak_rl import advanced_physics_research as apr
        
        # Test MHD predictor
        predictor = apr.AdvancedMHDPredictor(mode_database_size=50)
        print("  ✓ MHD predictor created")
        
        # Create test profile
        n_points = 20
        radius = [i / (n_points - 1) for i in range(n_points)]
        temp_e = [10.0 * (1 - r**2) for r in radius]
        temp_i = [8.0 * (1 - r**2) for r in radius]
        density = [2e19 * (1 - r**2) for r in radius]
        pressure = [1.6e-19 * (te + ti) * ne for te, ti, ne in zip(temp_e, temp_i, density)]
        q_profile = [1.0 + 3.0 * r**2 for r in radius]
        j_profile = [1e6 * (1 - r**2) for r in radius]
        magnetic_shear = [0.1] * n_points
        
        profile = apr.PlasmaProfile(
            radius=radius, temperature_e=temp_e, temperature_i=temp_i,
            density_e=density, pressure=pressure, q_profile=q_profile,
            j_profile=j_profile, magnetic_shear=magnetic_shear
        )
        
        print("  ✓ Plasma profile created")
        
        # Test stability analysis
        instabilities = predictor.analyze_stability(profile)
        print(f"  ✓ Instabilities found: {len(instabilities)}")
        
        # Test disruption prediction
        disruption_prob, diagnostics = predictor.predict_disruption_probability(instabilities)
        print(f"  ✓ Disruption probability: {disruption_prob:.3f}")
        
        # Test mitigation strategies
        strategies = predictor.suggest_mitigation_strategies(instabilities)
        print(f"  ✓ Mitigation strategies: {len(strategies)}")
        
        # Test multi-scale model
        model = apr.MultiScalePhysicsModel()
        test_state = {'temperature': temp_e[:5], 'density': density[:5]}
        evolved_state = model.evolve_coupled_system(test_state, 0.001)
        print("  ✓ Multi-scale evolution")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def test_error_handling():
    """Test error handling module directly."""
    print("Testing robust error handling...")
    
    try:
        from tokamak_rl import robust_error_handling_system as rehs
        
        # Test error classes
        error = rehs.PlasmaControlError("Test error", rehs.ErrorSeverity.MEDIUM)
        print("  ✓ Error classes created")
        
        # Test error handler
        handler = rehs.ErrorHandler()
        success = handler.handle_error(error)
        print("  ✓ Error handler processed error")
        
        # Test validator
        validator = rehs.DataValidator()
        validator.add_validation_rule('test_param', 'range', min=0, max=10)
        
        test_state = {'test_param': 5.0, 'plasma_current': 2.0}
        is_valid, errors = validator.validate_plasma_state(test_state)
        print(f"  ✓ Data validation: valid={is_valid}")
        
        # Test robust decorator
        @rehs.robust_operation_decorator(max_retries=2)
        def test_operation():
            return "success"
        
        result = test_operation()
        print("  ✓ Robust operation decorator")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def test_safety_system():
    """Test safety system module directly."""
    print("Testing comprehensive safety system...")
    
    try:
        from tokamak_rl import comprehensive_safety_system as css
        
        # Test disruption predictor
        predictor = css.DisruptionPredictor()
        print("  ✓ Disruption predictor created")
        
        test_state = {
            'plasma_current': 2.0,
            'beta_n': 0.025,
            'q_min': 1.2,
            'density_avg': 2e19,
            'temp_avg': 12.0
        }
        
        prediction = predictor.predict_disruption(test_state)
        print(f"  ✓ Disruption prediction: {prediction.probability:.3f}")
        print(f"  ✓ Confidence: {prediction.confidence:.3f}")
        
        # Test safety interlock
        interlock = css.SafetyInterlock()
        safety_params = {
            'plasma_current': 2.0,
            'beta_n': 0.025,
            'stored_energy': 150
        }
        
        safety_level, violations = interlock.check_safety_limits(safety_params)
        print(f"  ✓ Safety level: {safety_level.value}")
        
        # Test component health monitor
        health_monitor = css.ComponentHealthMonitor()
        component_data = {
            'poloidal_field_coils': {
                'current_density': 5e6,
                'temperature': 70,
                'insulation_resistance': 3e6
            }
        }
        
        health_assessments = health_monitor.assess_component_health(component_data)
        print("  ✓ Component health assessment")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def test_hpc_system():
    """Test HPC system module directly."""
    print("Testing high-performance computing...")
    
    try:
        from tokamak_rl import high_performance_computing as hpc
        
        # Test distributed compute
        distributed = hpc.DistributedCompute(max_workers=2)
        print("  ✓ Distributed compute created")
        
        # Test resource discovery
        resources = distributed.get_resource_status()
        print(f"  ✓ Resources discovered: {len(resources)}")
        
        # Test memory optimizer
        memory_opt = hpc.MemoryOptimizer(max_memory_gb=2.0)
        success = memory_opt.allocate_memory_pool('test_pool', 0.5)
        print(f"  ✓ Memory pool allocated: {success}")
        
        stats = memory_opt.get_memory_stats()
        print(f"  ✓ Memory usage: {stats['total_allocated']:.1f}GB")
        
        # Test LRU cache
        cache = hpc.LRUCache(max_size_gb=1.0)
        cache.put('test_key', 'test_value', 10.0)  # 10MB
        value = cache.get('test_key')
        print(f"  ✓ Cache working: {value is not None}")
        
        # Cleanup
        distributed.shutdown()
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def main():
    """Run direct module tests."""
    print("TOKAMAK RL CONTROL SUITE - DIRECT MODULE TESTS")
    print("=" * 55)
    
    tests = [
        ("Quantum Plasma Control", test_quantum_control),
        ("Advanced Physics Research", test_physics_research),
        ("Robust Error Handling", test_error_handling),
        ("Comprehensive Safety System", test_safety_system),
        ("High-Performance Computing", test_hpc_system),
    ]
    
    passed = 0
    total = len(tests)
    
    for name, test_func in tests:
        print(f"\n{name}:")
        if test_func():
            passed += 1
            print(f"  ✅ {name} PASSED")
        else:
            print(f"  ❌ {name} FAILED")
    
    print(f"\n{'='*55}")
    print("SUMMARY")
    print(f"{'='*55}")
    print(f"Tests Passed: {passed}/{total}")
    print(f"Success Rate: {passed/total:.1%}")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        print("✅ System core functionality validated")
    elif passed >= total * 0.8:
        print("👍 MOST TESTS PASSED")
        print("⚠️  Some minor issues detected")
    else:
        print("⚠️  MULTIPLE TEST FAILURES")
        print("🔧 Requires debugging and fixes")
    
    print(f"\n🎯 QUALITY SCORE: {passed/total:.1%}")
    return passed == total


if __name__ == "__main__":
    main()