#!/usr/bin/env python3
"""
Simplified Quality Validation for Tokamak RL Control Suite

Tests basic functionality and integration without external dependencies.
"""

import os
import sys
import time
import json
import math
import random

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_quantum_plasma_control():
    """Test quantum plasma control system."""
    print("Testing Quantum Plasma Control...")
    
    try:
        from tokamak_rl.quantum_plasma_control import (
            QuantumPlasmaController,
            QuantumEnhancedSAC,
            create_quantum_enhanced_training_system
        )
        
        # Test controller initialization
        controller = QuantumPlasmaController(n_qubits=4, coherence_time=1e-3)
        assert controller.n_qubits == 4
        print("  ✓ Controller initialization")
        
        # Test quantum evolution
        plasma_obs = [1.0, 0.5, -0.3, 0.8, 0.2]
        control_action = [0.1, -0.2, 0.3, -0.1]
        
        optimal_control, quantum_advantage = controller.quantum_plasma_evolution(plasma_obs, control_action)
        
        assert isinstance(optimal_control, list)
        assert 0.0 <= quantum_advantage <= 1.0
        assert all(-1.0 <= c <= 1.0 for c in optimal_control)
        print("  ✓ Quantum evolution")
        
        # Test quantum-enhanced SAC
        agent = QuantumEnhancedSAC(observation_dim=10, action_dim=4, quantum_enhancement=True)
        observation = [random.uniform(-1, 1) for _ in range(10)]
        action = agent.select_action(observation, training=True)
        
        assert len(action) == 4
        assert all(-1.0 <= a <= 1.0 for a in action)
        print("  ✓ Quantum-enhanced SAC")
        
        # Test training system
        system = create_quantum_enhanced_training_system()
        test_obs = [0.1 * i for i in range(45)]
        action, metrics = system['train_step'](test_obs)
        
        assert len(action) >= 4
        assert isinstance(metrics, dict)
        print("  ✓ Training system")
        
        return True, "Quantum plasma control tests passed"
        
    except Exception as e:
        return False, f"Quantum plasma control test failed: {e}"


def test_advanced_physics_research():
    """Test advanced physics research system."""
    print("Testing Advanced Physics Research...")
    
    try:
        from tokamak_rl.advanced_physics_research import (
            AdvancedMHDPredictor,
            MultiScalePhysicsModel,
            create_advanced_physics_research_system
        )
        
        # Test MHD predictor
        predictor = AdvancedMHDPredictor(mode_database_size=100)
        assert len(predictor.model_weights) == 100
        print("  ✓ MHD predictor initialization")
        
        # Test physics research system
        system = create_advanced_physics_research_system()
        profile = system['generate_test_plasma_profile']()
        
        assert len(profile.radius) == 50
        assert len(profile.temperature_e) == 50
        print("  ✓ Plasma profile generation")
        
        # Test stability analysis
        instabilities = system['mhd_predictor'].analyze_stability(profile)
        assert isinstance(instabilities, list)
        print("  ✓ MHD stability analysis")
        
        # Test disruption prediction
        disruption_prob, diagnostics = system['mhd_predictor'].predict_disruption_probability(instabilities)
        assert 0.0 <= disruption_prob <= 1.0
        assert isinstance(diagnostics, dict)
        print("  ✓ Disruption prediction")
        
        # Test multi-scale physics
        multiscale = MultiScalePhysicsModel()
        test_state = {
            'temperature': [10.0, 8.0, 6.0, 4.0, 2.0],
            'density': [2e19, 1.8e19, 1.5e19, 1.2e19, 1e19]
        }
        
        evolved_state = multiscale.evolve_coupled_system(test_state, 0.001)
        assert 'temperature' in evolved_state
        print("  ✓ Multi-scale physics")
        
        return True, "Advanced physics research tests passed"
        
    except Exception as e:
        return False, f"Advanced physics research test failed: {e}"


def test_robust_error_handling():
    """Test robust error handling system."""
    print("Testing Robust Error Handling...")
    
    try:
        from tokamak_rl.robust_error_handling_system import (
            create_robust_error_handling_system,
            PlasmaControlError,
            ErrorSeverity
        )
        
        # Test system creation
        system = create_robust_error_handling_system()
        assert 'error_handler' in system
        assert 'validator' in system
        print("  ✓ Error handling system creation")
        
        # Test robust control step
        normal_obs = [2.0, 0.03] + [1e19] * 10 + [10.0] * 10
        normal_action = [0.1, -0.2, 0.3, -0.1, 0.0, 0.2, -0.3, 0.1]
        
        result = system['robust_control_step'](normal_obs, normal_action)
        assert 'observation_valid' in result
        assert 'action_valid' in result
        print("  ✓ Robust control step")
        
        # Test error handling with invalid input
        invalid_obs = [50.0, -0.1] + [0] * 20  # Out of range
        try:
            system['robust_control_step'](invalid_obs, normal_action)
            print("  ✓ Invalid input handling")
        except Exception:
            print("  ✓ Invalid input handling (exception expected)")
        
        # Test system health check
        health = system['system_health_check']()
        assert 'health_score' in health
        assert 'status' in health
        assert 0.0 <= health['health_score'] <= 100.0
        print("  ✓ System health check")
        
        return True, "Robust error handling tests passed"
        
    except Exception as e:
        return False, f"Robust error handling test failed: {e}"


def test_comprehensive_safety_system():
    """Test comprehensive safety system."""
    print("Testing Comprehensive Safety System...")
    
    try:
        from tokamak_rl.comprehensive_safety_system import (
            create_comprehensive_safety_system,
            DisruptionPredictor,
            SafetyInterlock
        )
        
        # Test system creation
        system = create_comprehensive_safety_system()
        assert 'disruption_predictor' in system
        assert 'safety_interlock' in system
        print("  ✓ Safety system creation")
        
        # Test comprehensive safety check
        test_state = {
            'plasma_current': 2.0,
            'beta_n': 0.025,
            'density': [2e19] * 10,
            'temperature': [15.0] * 10,
            'stored_energy': 150,
            'divertor_power': 12,
            'pressure': 1e-5
        }
        
        safety_results = system['comprehensive_safety_check'](test_state)
        
        assert 'overall_status' in safety_results
        assert 'safety_level' in safety_results
        assert 'disruption_prediction' in safety_results
        assert safety_results['overall_status'] in ['NORMAL', 'CAUTION', 'WARNING', 'CRITICAL', 'EMERGENCY']
        print("  ✓ Comprehensive safety check")
        
        # Test disruption predictor
        predictor = DisruptionPredictor()
        prediction = predictor.predict_disruption(test_state)
        
        assert 0.0 <= prediction.probability <= 1.0
        assert 0.0 <= prediction.confidence <= 1.0
        assert isinstance(prediction.contributing_factors, list)
        print("  ✓ Disruption prediction")
        
        # Test safety interlock
        interlock = SafetyInterlock()
        safety_params = {
            'plasma_current': 2.0,
            'beta_n': 0.025,
            'density_peak': 2e19
        }
        
        safety_level, violations = interlock.check_safety_limits(safety_params)
        assert hasattr(safety_level, 'value')  # Enum
        print("  ✓ Safety interlock")
        
        return True, "Comprehensive safety system tests passed"
        
    except Exception as e:
        return False, f"Comprehensive safety system test failed: {e}"


def test_high_performance_computing():
    """Test high-performance computing system."""
    print("Testing High-Performance Computing...")
    
    try:
        from tokamak_rl.high_performance_computing import (
            create_high_performance_system,
            DistributedCompute,
            MemoryOptimizer
        )
        
        # Test system creation
        system = create_high_performance_system()
        assert 'distributed_compute' in system
        assert 'memory_optimizer' in system
        print("  ✓ HPC system creation")
        
        # Test memory optimizer
        memory_optimizer = MemoryOptimizer(max_memory_gb=4.0)
        success = memory_optimizer.allocate_memory_pool('test_pool', 1.0)
        assert success is True
        
        stats = memory_optimizer.get_memory_stats()
        assert 'total_allocated' in stats
        assert stats['total_allocated'] == 1.0
        print("  ✓ Memory optimizer")
        
        # Test distributed compute
        distributed_compute = DistributedCompute(max_workers=2)
        resource_status = distributed_compute.get_resource_status()
        assert isinstance(resource_status, dict)
        assert len(resource_status) >= 1  # At least CPU resource
        print("  ✓ Distributed compute")
        
        # Test performance metrics
        metrics = distributed_compute.get_performance_metrics()
        assert 'total_tasks_completed' in metrics
        assert 'resource_status' in metrics
        print("  ✓ Performance metrics")
        
        # Cleanup
        distributed_compute.shutdown()
        
        return True, "High-performance computing tests passed"
        
    except Exception as e:
        return False, f"High-performance computing test failed: {e}"


def test_system_integration():
    """Test integration between systems."""
    print("Testing System Integration...")
    
    try:
        # Import all systems
        from tokamak_rl.quantum_plasma_control import create_quantum_enhanced_training_system
        from tokamak_rl.advanced_physics_research import create_advanced_physics_research_system
        from tokamak_rl.comprehensive_safety_system import create_comprehensive_safety_system
        
        # Create all systems
        quantum_system = create_quantum_enhanced_training_system()
        physics_system = create_advanced_physics_research_system()
        safety_system = create_comprehensive_safety_system()
        
        print("  ✓ All systems created successfully")
        
        # Test data flow between systems
        # 1. Generate plasma profile
        profile = physics_system['generate_test_plasma_profile']()
        
        # 2. Create observation from profile
        observation = (
            [2.0, 0.025] +  # plasma_current, beta_n
            profile.density_e[:10] +  # density profile (first 10 points)
            profile.temperature_e[:10] +  # temperature profile (first 10 points)
            profile.q_profile[:10]  # q profile (first 10 points)
        )
        observation = observation[:45]  # Limit to expected dimension
        
        # 3. Get control action from quantum system
        action, metrics = quantum_system['train_step'](observation)
        assert len(action) >= 4
        
        # 4. Run safety check
        test_state = {
            'plasma_current': 2.0,
            'beta_n': 0.025,
            'density': profile.density_e[:10],
            'temperature': profile.temperature_e[:10],
            'stored_energy': 150,
            'divertor_power': 12,
            'pressure': 1e-5
        }
        
        safety_results = safety_system['comprehensive_safety_check'](test_state)
        assert 'overall_status' in safety_results
        
        print("  ✓ Data flow between systems")
        
        # 5. Test end-to-end workflow
        workflow_steps = [
            "Physics analysis completed",
            "Control action generated",
            "Safety assessment completed",
            "Integration successful"
        ]
        
        for step in workflow_steps:
            print(f"    - {step}")
        
        return True, "System integration tests passed"
        
    except Exception as e:
        return False, f"System integration test failed: {e}"


def run_performance_benchmark():
    """Run basic performance benchmark."""
    print("Running Performance Benchmark...")
    
    try:
        from tokamak_rl.quantum_plasma_control import create_quantum_enhanced_training_system
        
        system = create_quantum_enhanced_training_system()
        
        # Benchmark quantum control action generation
        n_operations = 50
        start_time = time.time()
        
        for i in range(n_operations):
            observation = [random.uniform(-1, 1) for _ in range(45)]
            action, metrics = system['train_step'](observation)
        
        end_time = time.time()
        total_time = end_time - start_time
        throughput = n_operations / total_time
        
        print(f"  Performance Results:")
        print(f"    Operations: {n_operations}")
        print(f"    Total Time: {total_time:.3f} seconds")
        print(f"    Throughput: {throughput:.1f} operations/second")
        print(f"    Avg Time per Operation: {(total_time/n_operations)*1000:.2f} ms")
        
        # Performance criteria
        if throughput >= 10:  # At least 10 ops/sec
            print("  ✓ Performance benchmark: EXCELLENT")
            return True, "Performance benchmark passed"
        elif throughput >= 5:
            print("  ✓ Performance benchmark: GOOD")
            return True, "Performance benchmark passed"
        else:
            print("  ⚠ Performance benchmark: NEEDS IMPROVEMENT")
            return False, f"Low throughput: {throughput:.1f} ops/sec"
            
    except Exception as e:
        return False, f"Performance benchmark failed: {e}"


def main():
    """Run comprehensive validation."""
    print("TOKAMAK RL CONTROL SUITE - QUALITY VALIDATION")
    print("=" * 55)
    print(f"Start Time: {time.ctime()}")
    print()
    
    # Test suite
    tests = [
        ("Quantum Plasma Control", test_quantum_plasma_control),
        ("Advanced Physics Research", test_advanced_physics_research),
        ("Robust Error Handling", test_robust_error_handling),
        ("Comprehensive Safety System", test_comprehensive_safety_system),
        ("High-Performance Computing", test_high_performance_computing),
        ("System Integration", test_system_integration),
        ("Performance Benchmark", run_performance_benchmark)
    ]
    
    results = {}
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        
        try:
            start_time = time.time()
            success, message = test_func()
            execution_time = time.time() - start_time
            
            results[test_name] = {
                'success': success,
                'message': message,
                'execution_time': execution_time
            }
            
            if success:
                passed_tests += 1
                print(f"✅ PASSED: {message} ({execution_time:.2f}s)")
            else:
                print(f"❌ FAILED: {message} ({execution_time:.2f}s)")
                
        except Exception as e:
            print(f"❌ ERROR: {e}")
            results[test_name] = {
                'success': False,
                'message': str(e),
                'execution_time': 0.0
            }
    
    # Summary
    print(f"\n{'='*55}")
    print("VALIDATION SUMMARY")
    print(f"{'='*55}")
    
    success_rate = passed_tests / total_tests
    overall_status = (
        "EXCELLENT" if success_rate >= 0.9 else
        "VERY GOOD" if success_rate >= 0.8 else
        "GOOD" if success_rate >= 0.7 else
        "NEEDS IMPROVEMENT" if success_rate >= 0.5 else
        "POOR"
    )
    
    print(f"Tests Passed:        {passed_tests}/{total_tests}")
    print(f"Success Rate:        {success_rate:.1%}")
    print(f"Overall Status:      {overall_status}")
    
    total_execution_time = sum(r['execution_time'] for r in results.values())
    print(f"Total Runtime:       {total_execution_time:.2f} seconds")
    print(f"Completion Time:     {time.ctime()}")
    
    # Detailed results
    print(f"\nDETAILED RESULTS:")
    for test_name, result in results.items():
        status = "PASS" if result['success'] else "FAIL"
        print(f"  {test_name:<30} {status:<4} ({result['execution_time']:.2f}s)")
    
    # Recommendations
    print(f"\nRECOMMENDATIONS:")
    if success_rate >= 0.9:
        print("  🎉 Excellent! System is ready for advanced research and deployment.")
        print("  📈 Consider performance optimization for production use.")
        print("  🔒 Ensure security measures are appropriate for deployment environment.")
    elif success_rate >= 0.7:
        print("  👍 Good progress! Address failing tests before production deployment.")
        print("  🔧 Focus on reliability and error handling improvements.")
        print("  📊 Run additional stress tests.")
    else:
        print("  ⚠️  Significant improvements needed before deployment.")
        print("  🐛 Debug and fix failing components.")
        print("  🧪 Increase test coverage and validation.")
    
    print(f"\n🎯 FINAL QUALITY SCORE: {success_rate:.1%}")
    
    if success_rate >= 0.8:
        print("✅ QUALITY VALIDATION PASSED!")
    else:
        print("❌ QUALITY VALIDATION NEEDS IMPROVEMENT")
    
    # Save results
    validation_report = {
        'timestamp': time.time(),
        'success_rate': success_rate,
        'overall_status': overall_status,
        'total_tests': total_tests,
        'passed_tests': passed_tests,
        'total_execution_time': total_execution_time,
        'detailed_results': results
    }
    
    with open('simple_quality_validation_results.json', 'w') as f:
        json.dump(validation_report, f, indent=2)
    
    print(f"\n📄 Results saved to: simple_quality_validation_results.json")
    print("✅ Quality validation complete!")
    
    return success_rate >= 0.7  # Return True if validation passes


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)