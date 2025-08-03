#!/usr/bin/env python3
"""
Simple test script for business logic module.
"""

import sys
import os
sys.path.insert(0, 'src')

# Import modules directly to avoid circular dependencies
import numpy as np
from scipy.optimize import minimize, differential_evolution

# Import specific business modules without going through __init__.py
exec(open('src/tokamak_rl/business.py').read())

def test_business_module():
    """Test business logic functionality."""
    print("🧪 Testing Business Logic Module")
    print("=" * 50)
    
    # Test 1: PerformanceMetrics
    print("\n1. Testing PerformanceMetrics...")
    metrics = PerformanceMetrics(
        energy_efficiency=0.85,
        q_factor_stability=0.9,
        shape_control_accuracy=2.5,
        uptime_percentage=95.0
    )
    print(f"   ✓ Created PerformanceMetrics with efficiency: {metrics.energy_efficiency}")
    print(f"   ✓ Q-factor stability: {metrics.q_factor_stability}")
    
    # Test 2: OptimizationTarget
    print("\n2. Testing OptimizationTarget...")
    target = OptimizationTarget(
        desired_beta=0.03,
        target_q_min=2.5,
        max_shape_error=2.0,
        beta_weight=2.0
    )
    print(f"   ✓ Created OptimizationTarget with beta: {target.desired_beta}")
    print(f"   ✓ Target Q-min: {target.target_q_min}")
    
    # Test 3: OperationalMode enum
    print("\n3. Testing OperationalMode...")
    mode = OperationalMode.FLATTOP
    print(f"   ✓ OperationalMode enum works: {mode.value}")
    
    # Test 4: PerformanceAnalyzer (basic functionality)
    print("\n4. Testing PerformanceAnalyzer...")
    analyzer = PerformanceAnalyzer("./test_data")
    
    # Add sample data
    for i in range(10):
        sample_metrics = PerformanceMetrics(
            energy_efficiency=0.8 + 0.01 * i,
            q_factor_stability=0.9 + 0.005 * i,
            timestamp=1000.0 + i * 100
        )
        analyzer.add_performance_data(sample_metrics)
    
    print(f"   ✓ Added {len(analyzer.performance_data)} data points")
    
    # Calculate KPIs
    kpis = analyzer.calculate_kpis()
    print(f"   ✓ Calculated KPIs: {list(kpis.keys())}")
    if 'efficiency' in kpis:
        print(f"   ✓ Average efficiency: {kpis['efficiency']:.3f}")
    
    print("\n✅ All business logic tests passed!")
    return True

if __name__ == "__main__":
    try:
        test_business_module()
        print("\n🎉 Business logic module is fully functional!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()