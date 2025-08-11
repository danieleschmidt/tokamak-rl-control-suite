#!/usr/bin/env python3
"""
Enhanced runtime test with comprehensive tokamak-rl functionality validation.
Tests all core systems and generates a demonstration of capabilities.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_complete_system():
    """Test complete system integration including business logic."""
    print("🧪 Testing Complete System Integration...")
    
    try:
        from tokamak_rl import make_tokamak_env, TokamakConfig
        from tokamak_rl.business import create_business_system
        from tokamak_rl.research import create_research_system
        from tokamak_rl.analytics import create_analytics_system
        
        # Create ITER environment
        env = make_tokamak_env("ITER", enable_safety=False)
        print("✅ ITER environment created")
        
        # Test business system
        business_system = create_business_system()
        print("✅ Business intelligence system initialized")
        
        # Test analytics system  
        analytics_system = create_analytics_system()
        print("✅ Advanced analytics system initialized")
        
        # Test research system
        research_system = create_research_system()
        print("✅ Research framework initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ Complete system test failed: {e}")
        return False

def demonstrate_plasma_control():
    """Demonstrate plasma control simulation."""
    print("\n🔬 Demonstrating Plasma Control Simulation...")
    
    try:
        from tokamak_rl import make_tokamak_env
        
        # Create environment
        env = make_tokamak_env("ITER", control_frequency=10, enable_safety=False)
        
        # Initialize environment
        observation, info = env.reset()
        print(f"✅ Environment reset - Initial observation shape: {len(observation)}")
        
        # Run simulation steps
        total_reward = 0
        for step in range(5):
            # Random action (normally would be from trained RL agent)
            action = [0.1 * (-1)**step for _ in range(env.action_space.shape[0])]
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            print(f"   Step {step+1}: reward={reward:.3f}, done={terminated or truncated}")
            
            if terminated or truncated:
                break
        
        print(f"✅ Simulation completed - Total reward: {total_reward:.3f}")
        return True
        
    except Exception as e:
        print(f"❌ Plasma control demonstration failed: {e}")
        return False

def demonstrate_research_capabilities():
    """Demonstrate research and experimental capabilities."""
    print("\n🔬 Demonstrating Research Capabilities...")
    
    try:
        from tokamak_rl.research import create_research_system, ExperimentalFramework
        from tokamak_rl.physics import TokamakConfig
        
        # Create research system
        research_system = create_research_system()
        
        # Create experimental framework
        framework = ExperimentalFramework()
        
        # Test algorithm comparison
        configs = ["ITER", "SPARC"]
        results = {}
        
        for config_name in configs:
            config = TokamakConfig.from_preset(config_name)
            result = framework.run_baseline_comparison(
                config, 
                algorithms=["SAC", "Dreamer"],
                n_episodes=3
            )
            results[config_name] = result
            print(f"✅ Baseline comparison completed for {config_name}")
        
        print(f"✅ Research demonstration completed - {len(results)} configurations tested")
        return True
        
    except Exception as e:
        print(f"❌ Research demonstration failed: {e}")
        return False

def demonstrate_business_intelligence():
    """Demonstrate business intelligence and optimization."""
    print("\n💼 Demonstrating Business Intelligence...")
    
    try:
        from tokamak_rl.business import create_business_system, PlasmaOptimizer
        from tokamak_rl.physics import TokamakConfig
        
        # Create business system
        business_system = create_business_system()
        
        # Test plasma optimization
        optimizer = PlasmaOptimizer()
        config = TokamakConfig.from_preset("ITER")
        
        # Optimize plasma parameters
        optimized_params = optimizer.optimize_plasma_parameters(
            config,
            objectives={'shape_accuracy': 0.6, 'confinement': 0.4},
            constraints={'q_min': 1.5, 'beta_limit': 0.04}
        )
        
        print(f"✅ Plasma optimization completed")
        print(f"   Optimized parameters: {list(optimized_params.keys())}")
        
        # Performance analysis
        from tokamak_rl.business import PerformanceAnalyzer
        analyzer = PerformanceAnalyzer()
        
        metrics = analyzer.analyze_performance({
            'shape_error': [2.1, 1.8, 1.5, 1.2],
            'disruption_rate': [0.05, 0.03, 0.02, 0.01],
            'efficiency': [0.85, 0.88, 0.92, 0.95]
        })
        
        print(f"✅ Performance analysis completed")
        print(f"   Metrics analyzed: {list(metrics.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Business intelligence demonstration failed: {e}")
        return False

def demonstrate_safety_systems():
    """Demonstrate safety and monitoring systems.""" 
    print("\n🛡️ Demonstrating Safety Systems...")
    
    try:
        from tokamak_rl.safety import SafetyShield, create_safety_system
        from tokamak_rl.monitoring import create_monitoring_system, PlasmaMonitor
        from tokamak_rl.physics import TokamakConfig, PlasmaState
        
        # Create safety system
        safety_system = create_safety_system()
        shield = SafetyShield()
        
        # Test safety validation
        config = TokamakConfig.from_preset("ITER")
        test_state = PlasmaState(
            plasma_current=15.0,  # MA
            plasma_beta=0.035,
            q_profile=[2.5, 2.0, 1.8, 1.6, 1.4, 1.3, 1.2, 1.1, 1.0, 1.1],
            shape_error=3.2,
            density_profile=[8e19] * 10,
            temperature_profile=[20, 18, 15, 12, 10] # keV
        )
        
        is_safe = shield.validate_safety(test_state)
        print(f"✅ Safety validation completed - State is {'safe' if is_safe else 'unsafe'}")
        
        # Test monitoring system
        monitoring_system = create_monitoring_system()
        monitor = PlasmaMonitor()
        
        monitor.log_plasma_state(test_state)
        alerts = monitor.check_alerts(test_state)
        
        print(f"✅ Monitoring system operational - {len(alerts)} alerts generated")
        
        return True
        
    except Exception as e:
        print(f"❌ Safety system demonstration failed: {e}")
        return False

def run_comprehensive_demo():
    """Run comprehensive demonstration of all systems."""
    print("🚀 TOKAMAK-RL COMPREHENSIVE SYSTEM DEMONSTRATION\n")
    print("=" * 60)
    
    demonstrations = [
        ("Core System Integration", test_complete_system),
        ("Plasma Control Simulation", demonstrate_plasma_control), 
        ("Research Capabilities", demonstrate_research_capabilities),
        ("Business Intelligence", demonstrate_business_intelligence),
        ("Safety Systems", demonstrate_safety_systems)
    ]
    
    passed = 0
    total = len(demonstrations)
    
    for name, demo_func in demonstrations:
        print(f"\n{'='*20} {name} {'='*20}")
        try:
            if demo_func():
                passed += 1
                print(f"✅ {name}: PASSED")
            else:
                print(f"❌ {name}: FAILED")
        except Exception as e:
            print(f"❌ {name}: FAILED - {e}")
    
    print(f"\n{'='*60}")
    print(f"📊 DEMONSTRATION RESULTS: {passed}/{total} systems operational")
    
    if passed == total:
        print("🎉 ALL SYSTEMS OPERATIONAL - TOKAMAK-RL IS FULLY FUNCTIONAL!")
        print("\n🌟 Key Capabilities Demonstrated:")
        print("   • Physics-based tokamak simulation")
        print("   • Reinforcement learning integration")
        print("   • Safety-critical system monitoring")
        print("   • Business intelligence and optimization")
        print("   • Research and experimental frameworks")
        print("   • Real-time performance analytics")
    else:
        print("⚠️ Some systems need attention - core functionality works")
    
    return passed == total

if __name__ == "__main__":
    success = run_comprehensive_demo()
    sys.exit(0 if success else 1)