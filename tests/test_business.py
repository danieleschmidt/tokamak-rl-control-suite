"""
Tests for business logic and optimization algorithms.
"""

import pytest
import numpy as np
import tempfile
import json
from pathlib import Path
from tokamak_rl.business import (
    PlasmaOptimizer, ScenarioPlanner, PerformanceAnalyzer,
    OperationalMode, PerformanceMetrics, OptimizationTarget,
    create_business_system
)
from tokamak_rl.physics import TokamakConfig, PlasmaState
from tokamak_rl.safety import SafetyLimits


class TestPerformanceMetrics:
    """Test performance metrics data structure."""
    
    def test_performance_metrics_creation(self):
        """Test performance metrics creation."""
        metrics = PerformanceMetrics(
            q_factor_stability=0.95,
            shape_control_accuracy=2.5,
            beta_normalized=0.025,
            energy_efficiency=0.85,
            session_id="test_session"
        )
        
        assert metrics.q_factor_stability == 0.95
        assert metrics.shape_control_accuracy == 2.5
        assert metrics.beta_normalized == 0.025
        assert metrics.energy_efficiency == 0.85
        assert metrics.session_id == "test_session"
    
    def test_performance_metrics_defaults(self):
        """Test performance metrics default values."""
        metrics = PerformanceMetrics()
        
        assert metrics.q_factor_stability == 0.0
        assert metrics.disruption_avoidance_rate == 0.0
        assert metrics.uptime_percentage == 0.0
        assert metrics.timestamp == 0.0


class TestOptimizationTarget:
    """Test optimization target configuration."""
    
    def test_optimization_target_creation(self):
        """Test optimization target creation."""
        target = OptimizationTarget(
            desired_beta=0.03,
            target_q_min=2.5,
            max_shape_error=1.5,
            beta_weight=2.0
        )
        
        assert target.desired_beta == 0.03
        assert target.target_q_min == 2.5
        assert target.max_shape_error == 1.5
        assert target.beta_weight == 2.0
    
    def test_optimization_target_defaults(self):
        """Test optimization target default values."""
        target = OptimizationTarget()
        
        assert target.desired_beta == 0.025
        assert target.target_q_min == 2.0
        assert target.elongation_target == 1.7
        assert target.safety_weight == 3.0


class TestPlasmaOptimizer:
    """Test plasma optimization algorithms."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = TokamakConfig.from_preset("ITER")
        self.safety_limits = SafetyLimits()
        self.optimizer = PlasmaOptimizer(self.config, self.safety_limits)
        self.plasma_state = PlasmaState(self.config)
        
    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        assert self.optimizer.config == self.config
        assert self.optimizer.safety_limits == self.safety_limits
        assert len(self.optimizer.optimization_history) == 0
        assert len(self.optimizer.best_solutions) == 0
    
    def test_plasma_response_simulation(self):
        """Test simplified plasma response simulation."""
        action = np.array([0.1, -0.1, 0.2, -0.2, 0.1, -0.1, 0.5, 10.0])
        
        new_state = self.optimizer._simulate_plasma_response(self.plasma_state, action)
        
        assert isinstance(new_state, PlasmaState)
        assert hasattr(new_state, 'q_min')
        assert hasattr(new_state, 'shape_error')
        assert hasattr(new_state, 'plasma_beta')
    
    def test_plasma_optimization(self):
        """Test plasma shape optimization."""
        target = OptimizationTarget(
            desired_beta=0.025,
            target_q_min=2.0,
            max_shape_error=2.0
        )
        
        optimal_action, optimization_info = self.optimizer.optimize_plasma_shape(
            self.plasma_state, target, method="scipy"
        )
        
        assert isinstance(optimal_action, np.ndarray)
        assert optimal_action.shape == (8,)
        assert isinstance(optimization_info, dict)
        assert 'success' in optimization_info
        assert 'fun' in optimization_info
        assert 'method' in optimization_info
    
    def test_differential_evolution_optimization(self):
        """Test optimization with differential evolution."""
        target = OptimizationTarget()
        
        optimal_action, optimization_info = self.optimizer.optimize_plasma_shape(
            self.plasma_state, target, method="differential_evolution"
        )
        
        assert isinstance(optimal_action, np.ndarray)
        assert optimal_action.shape == (8,)
        assert optimization_info['method'] == "differential_evolution"
        
        # Check that optimization history is updated
        assert len(self.optimizer.optimization_history) == 1
    
    def test_operational_efficiency_analysis(self):
        """Test operational efficiency analysis."""
        # Create sample performance data
        performance_data = []
        for i in range(50):
            metrics = PerformanceMetrics(
                q_factor_stability=0.9 + 0.1 * np.random.random(),
                shape_control_accuracy=2.0 + np.random.random(),
                energy_efficiency=0.8 + 0.2 * np.random.random(),
                disruption_avoidance_rate=0.95 + 0.05 * np.random.random(),
                uptime_percentage=90 + 10 * np.random.random()
            )
            performance_data.append(metrics)
        
        efficiency_analysis = self.optimizer.analyze_operational_efficiency(performance_data)
        
        assert isinstance(efficiency_analysis, dict)
        assert 'mean_q_stability' in efficiency_analysis
        assert 'shape_control_rms' in efficiency_analysis
        assert 'disruption_rate' in efficiency_analysis
        assert 'uptime_factor' in efficiency_analysis
        
        # Values should be reasonable
        assert 0 <= efficiency_analysis['disruption_rate'] <= 1
        assert 0 <= efficiency_analysis['uptime_factor'] <= 100
    
    def test_optimization_summary(self):
        """Test optimization summary generation."""
        # Run a few optimizations
        target = OptimizationTarget()
        for _ in range(3):
            self.optimizer.optimize_plasma_shape(self.plasma_state, target)
        
        summary = self.optimizer.get_optimization_summary()
        
        assert isinstance(summary, dict)
        assert 'total_optimizations' in summary
        assert 'successful_optimizations' in summary
        assert 'success_rate' in summary
        assert summary['total_optimizations'] == 3


class TestScenarioPlanner:
    """Test operational scenario planning."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = TokamakConfig.from_preset("SPARC")
        self.planner = ScenarioPlanner(self.config)
    
    def test_planner_initialization(self):
        """Test planner initialization."""
        assert self.planner.config == self.config
        assert len(self.planner.scenarios) == 0
        assert len(self.planner.schedule) == 0
    
    def test_startup_scenario_creation(self):
        """Test startup scenario creation."""
        scenario = self.planner.create_discharge_scenario(
            "startup_test",
            duration=10.0,
            mode=OperationalMode.STARTUP,
            target_parameters={
                'plasma_current': 12e6,
                'beta': 0.02,
                'elongation': 1.6,
                'q_min': 2.0
            }
        )
        
        assert scenario['name'] == "startup_test"
        assert scenario['mode'] == OperationalMode.STARTUP
        assert scenario['duration'] == 10.0
        assert 'time_points' in scenario
        assert 'targets' in scenario
        
        # Check target trajectories
        assert 'plasma_current' in scenario['targets']
        assert 'beta' in scenario['targets']
        assert 'elongation' in scenario['targets']
        
        # Startup should have ramping behavior
        current_traj = scenario['targets']['plasma_current']
        assert current_traj[0] < current_traj[-1]  # Should increase
    
    def test_flattop_scenario_creation(self):
        """Test flat-top scenario creation."""
        scenario = self.planner.create_discharge_scenario(
            "flattop_test",
            duration=5.0,
            mode=OperationalMode.FLATTOP,
            target_parameters={
                'plasma_current': 15e6,
                'beta': 0.025
            }
        )
        
        assert scenario['mode'] == OperationalMode.FLATTOP
        
        # Flattop should have relatively constant values with small variations
        current_traj = scenario['targets']['plasma_current']
        current_variation = np.std(current_traj) / np.mean(current_traj)
        assert current_variation < 0.1  # Less than 10% variation
    
    def test_rampdown_scenario_creation(self):
        """Test ramp-down scenario creation."""
        scenario = self.planner.create_discharge_scenario(
            "rampdown_test",
            duration=8.0,
            mode=OperationalMode.RAMPDOWN,
            target_parameters={
                'initial_current': 15e6,
                'initial_beta': 0.025,
                'rampdown_time': 5.0
            }
        )
        
        assert scenario['mode'] == OperationalMode.RAMPDOWN
        
        # Rampdown should have decreasing behavior
        current_traj = scenario['targets']['plasma_current']
        assert current_traj[0] > current_traj[-1]  # Should decrease
    
    def test_schedule_optimization(self):
        """Test schedule optimization."""
        # Create multiple scenarios
        scenarios = []
        for i, mode in enumerate([OperationalMode.STARTUP, OperationalMode.FLATTOP, OperationalMode.RAMPDOWN]):
            scenario_name = f"test_scenario_{i}"
            self.planner.create_discharge_scenario(
                scenario_name,
                duration=5.0,
                mode=mode,
                target_parameters={'plasma_current': 12e6}
            )
            scenarios.append(scenario_name)
        
        # Optimize schedule
        constraints = {'preparation_time': 300.0}
        schedule = self.planner.optimize_schedule(scenarios, constraints)
        
        assert len(schedule) == 3
        assert all('start_time' in item for item in schedule)
        assert all('duration' in item for item in schedule)
        assert all('preparation_time' in item for item in schedule)
        
        # Times should be sequential
        for i in range(1, len(schedule)):
            prev_end = schedule[i-1]['start_time'] + schedule[i-1]['duration'] + schedule[i-1]['preparation_time']
            current_start = schedule[i]['start_time']
            assert current_start >= prev_end
    
    def test_current_targets_retrieval(self):
        """Test retrieving current targets for a time point."""
        scenario = self.planner.create_discharge_scenario(
            "test_scenario",
            duration=10.0,
            mode=OperationalMode.STARTUP,
            target_parameters={'plasma_current': 15e6, 'beta': 0.025}
        )
        
        # Get targets at different time points
        targets_start = self.planner.get_current_targets("test_scenario", 0.0)
        targets_mid = self.planner.get_current_targets("test_scenario", 5.0)
        targets_end = self.planner.get_current_targets("test_scenario", 10.0)
        
        assert isinstance(targets_start, dict)
        assert isinstance(targets_mid, dict)
        assert isinstance(targets_end, dict)
        
        # For startup, current should increase over time
        if 'plasma_current' in targets_start and 'plasma_current' in targets_end:
            assert targets_start['plasma_current'] < targets_end['plasma_current']


class TestPerformanceAnalyzer:
    """Test performance analysis and reporting."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.analyzer = PerformanceAnalyzer(self.temp_dir)
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        assert self.analyzer.data_dir == Path(self.temp_dir)
        assert len(self.analyzer.performance_data) == 0
        assert isinstance(self.analyzer.analysis_cache, dict)
    
    def test_performance_data_addition(self):
        """Test adding performance data."""
        metrics = PerformanceMetrics(
            q_factor_stability=0.9,
            energy_efficiency=0.85,
            timestamp=1000.0
        )
        
        self.analyzer.add_performance_data(metrics)
        
        assert len(self.analyzer.performance_data) == 1
        assert self.analyzer.performance_data[0] == metrics
    
    def test_kpi_calculation(self):
        """Test KPI calculation."""
        # Add sample data
        for i in range(50):
            metrics = PerformanceMetrics(
                q_factor_stability=0.9 + 0.1 * np.random.random(),
                shape_control_accuracy=2.0 + np.random.random(),
                energy_efficiency=0.8 + 0.2 * np.random.random(),
                disruption_avoidance_rate=0.95 + 0.05 * np.random.random(),
                uptime_percentage=90 + 10 * np.random.random(),
                operational_cost_per_shot=100 + 50 * np.random.random(),
                timestamp=i * 100.0
            )
            self.analyzer.add_performance_data(metrics)
        
        kpis = self.analyzer.calculate_kpis()
        
        assert isinstance(kpis, dict)
        expected_kpis = [
            'availability', 'reliability', 'efficiency',
            'shape_accuracy', 'plasma_performance', 'stability',
            'cost_efficiency', 'maintenance_efficiency', 'control_quality'
        ]
        
        for kpi in expected_kpis:
            assert kpi in kpis
            assert isinstance(kpis[kpi], (int, float))
    
    def test_kpi_time_window(self):
        """Test KPI calculation with time window."""
        # Add data with different timestamps
        current_time = 5000.0
        for i in range(100):
            metrics = PerformanceMetrics(
                energy_efficiency=0.8 + 0.01 * i,  # Increasing efficiency
                timestamp=current_time - 100 + i
            )
            self.analyzer.add_performance_data(metrics)
        
        # Calculate KPIs for last 50 time units
        kpis_windowed = self.analyzer.calculate_kpis(time_window=50.0)
        kpis_all = self.analyzer.calculate_kpis()
        
        # Windowed KPIs should be different (higher efficiency for recent data)
        assert kpis_windowed['efficiency'] > kpis_all['efficiency']
    
    def test_trend_analysis(self):
        """Test trend analysis for specific metrics."""
        # Add data with clear trend
        for i in range(150):
            efficiency = 0.7 + 0.002 * i  # Clear increasing trend
            metrics = PerformanceMetrics(
                energy_efficiency=efficiency,
                timestamp=i * 10.0
            )
            self.analyzer.add_performance_data(metrics)
        
        trend = self.analyzer.generate_trend_analysis('energy_efficiency')
        
        assert isinstance(trend, dict)
        assert 'metric' in trend
        assert 'current_value' in trend
        assert 'trend_direction' in trend
        assert 'trend_magnitude' in trend
        
        assert trend['metric'] == 'energy_efficiency'
        assert trend['trend_direction'] == 'improving'  # Should detect increasing trend
        assert trend['trend_magnitude'] > 0
    
    def test_performance_report_generation(self):
        """Test performance report generation."""
        # Add sample data
        for i in range(100):
            metrics = PerformanceMetrics(
                q_factor_stability=0.9,
                energy_efficiency=0.85,
                disruption_avoidance_rate=0.95,
                uptime_percentage=95.0,
                timestamp=i * 100.0
            )
            self.analyzer.add_performance_data(metrics)
        
        report = self.analyzer.generate_performance_report()
        
        assert isinstance(report, str)
        assert "TOKAMAK PERFORMANCE REPORT" in report
        assert "OPERATIONAL KPIs" in report
        assert "PLASMA PERFORMANCE" in report
        assert "ECONOMIC METRICS" in report
        assert "TREND ANALYSIS" in report
    
    def test_performance_data_saving(self):
        """Test saving performance data to file."""
        # Add sample data
        for i in range(10):
            metrics = PerformanceMetrics(
                energy_efficiency=0.8,
                session_id=f"test_session_{i}",
                timestamp=i * 100.0
            )
            self.analyzer.add_performance_data(metrics)
        
        filepath = self.analyzer.save_performance_data("test_performance.json")
        
        assert Path(filepath).exists()
        
        # Load and verify data
        with open(filepath, 'r') as f:
            saved_data = json.load(f)
        
        assert len(saved_data) == 10
        assert all('energy_efficiency' in d for d in saved_data)
        assert all('timestamp' in d for d in saved_data)
    
    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


class TestBusinessSystemIntegration:
    """Test business system integration."""
    
    def test_business_system_factory(self):
        """Test business system factory function."""
        config = TokamakConfig.from_preset("NSTX")
        safety_limits = SafetyLimits()
        
        business_system = create_business_system(config, safety_limits)
        
        assert isinstance(business_system, dict)
        assert 'optimizer' in business_system
        assert 'planner' in business_system
        assert 'analyzer' in business_system
        
        assert isinstance(business_system['optimizer'], PlasmaOptimizer)
        assert isinstance(business_system['planner'], ScenarioPlanner)
        assert isinstance(business_system['analyzer'], PerformanceAnalyzer)
    
    def test_integrated_workflow(self):
        """Test integrated business logic workflow."""
        config = TokamakConfig.from_preset("SPARC")
        safety_limits = SafetyLimits()
        business_system = create_business_system(config, safety_limits)
        
        optimizer = business_system['optimizer']
        planner = business_system['planner']
        analyzer = business_system['analyzer']
        
        # 1. Create operational scenario
        scenario = planner.create_discharge_scenario(
            "integrated_test",
            duration=5.0,
            mode=OperationalMode.FLATTOP,
            target_parameters={'plasma_current': 12e6, 'beta': 0.025}
        )
        
        assert scenario['name'] == "integrated_test"
        
        # 2. Run optimization
        plasma_state = PlasmaState(config)
        target = OptimizationTarget()
        
        optimal_action, opt_info = optimizer.optimize_plasma_shape(plasma_state, target)
        
        assert isinstance(optimal_action, np.ndarray)
        assert opt_info['success'] or len(optimizer.optimization_history) > 0
        
        # 3. Analyze performance
        performance_metrics = PerformanceMetrics(
            energy_efficiency=0.85,
            q_factor_stability=0.9,
            timestamp=1000.0
        )
        analyzer.add_performance_data(performance_metrics)
        
        kpis = analyzer.calculate_kpis()
        assert 'efficiency' in kpis
        
        # 4. Generate reports
        opt_summary = optimizer.get_optimization_summary()
        perf_report = analyzer.generate_performance_report()
        
        assert isinstance(opt_summary, dict)
        assert isinstance(perf_report, str)
        assert "PERFORMANCE REPORT" in perf_report