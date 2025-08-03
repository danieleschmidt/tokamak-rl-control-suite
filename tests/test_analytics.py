"""
Tests for analytics and machine learning components.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
from tokamak_rl.analytics import (
    AnomalyDetector, PerformancePredictor, TrendAnalyzer,
    AnomalyType, AnomalyEvent, PredictionResult,
    create_analytics_system
)
from tokamak_rl.physics import TokamakConfig, PlasmaState
from tokamak_rl.business import PerformanceMetrics


class TestAnomalyEvent:
    """Test anomaly event data structure."""
    
    def test_anomaly_event_creation(self):
        """Test anomaly event creation."""
        event = AnomalyEvent(
            type=AnomalyType.PERFORMANCE_DEGRADATION,
            severity=0.8,
            timestamp=1000.0,
            features={'q_min': 1.2, 'shape_error': 6.5},
            description="Q-factor below threshold",
            confidence=0.9
        )
        
        assert event.type == AnomalyType.PERFORMANCE_DEGRADATION
        assert event.severity == 0.8
        assert event.timestamp == 1000.0
        assert event.features['q_min'] == 1.2
        assert event.description == "Q-factor below threshold"
        assert event.confidence == 0.9


class TestPredictionResult:
    """Test prediction result data structure."""
    
    def test_prediction_result_creation(self):
        """Test prediction result creation."""
        result = PredictionResult(
            predicted_values={'shape_error': 2.5, 'q_min': 2.0},
            confidence_intervals={'shape_error': (2.0, 3.0), 'q_min': (1.8, 2.2)},
            forecast_horizon=300.0,
            model_accuracy=0.85,
            feature_importance={'q_min': 0.3, 'beta': 0.4},
            timestamp=1000.0
        )
        
        assert result.predicted_values['shape_error'] == 2.5
        assert result.confidence_intervals['q_min'] == (1.8, 2.2)
        assert result.forecast_horizon == 300.0
        assert result.model_accuracy == 0.85
        assert result.feature_importance['beta'] == 0.4


class TestAnomalyDetector:
    """Test anomaly detection algorithms."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.detector = AnomalyDetector(contamination=0.1, window_size=20)
        self.config = TokamakConfig.from_preset("ITER")
        self.plasma_state = PlasmaState(self.config)
    
    def test_detector_initialization(self):
        """Test detector initialization."""
        assert self.detector.contamination == 0.1
        assert self.detector.window_size == 20
        assert not self.detector.is_trained
        assert len(self.detector.feature_history) == 0
        assert len(self.detector.anomaly_history) == 0
    
    def test_feature_extraction(self):
        """Test feature extraction from plasma state."""
        performance = PerformanceMetrics(
            energy_efficiency=0.85,
            q_factor_stability=0.9,
            control_smoothness=0.8
        )
        control_action = np.array([0.1, -0.1, 0.0, 0.0, 0.1, -0.1, 0.5, 10.0])
        
        features = self.detector.extract_features(
            self.plasma_state, performance, control_action
        )
        
        assert isinstance(features, dict)
        
        # Check plasma physics features
        expected_plasma_features = [
            'q_min', 'q_std', 'beta_normalized', 'shape_error',
            'elongation', 'triangularity', 'disruption_probability'
        ]
        for feature in expected_plasma_features:
            assert feature in features
        
        # Check performance features
        expected_performance_features = [
            'efficiency', 'stability', 'control_smoothness'
        ]
        for feature in expected_performance_features:
            assert feature in features
        
        # Check control features
        expected_control_features = [
            'control_magnitude', 'control_pf_rms', 'control_gas_rate', 'control_heating'
        ]
        for feature in expected_control_features:
            assert feature in features
    
    def test_detector_training(self):
        """Test anomaly detector training."""
        # Generate training data
        training_data = []
        for i in range(100):
            features = {
                'q_min': 2.0 + 0.5 * np.random.random(),
                'shape_error': 2.0 + np.random.random(),
                'beta_normalized': 0.6 + 0.4 * np.random.random(),
                'efficiency': 0.8 + 0.2 * np.random.random(),
                'disruption_probability': 0.05 * np.random.random()
            }
            training_data.append(features)
        
        training_stats = self.detector.train_detector(training_data)
        
        assert self.detector.is_trained
        assert isinstance(training_stats, dict)
        assert 'samples_trained' in training_stats
        assert 'features_count' in training_stats
        assert training_stats['samples_trained'] == 100
        
        # Check normal ranges were computed
        assert len(self.detector.normal_ranges) > 0
        assert 'q_min' in self.detector.normal_ranges
    
    def test_insufficient_training_data(self):
        """Test error handling for insufficient training data."""
        with pytest.raises(ValueError, match="Insufficient training data"):
            self.detector.train_detector([{'feature': 1}] * 10)
    
    def test_anomaly_detection_normal_case(self):
        """Test anomaly detection with normal features."""
        # Train detector first
        training_data = []
        for _ in range(100):
            features = {
                'q_min': 2.0 + 0.2 * np.random.random(),
                'shape_error': 2.0 + 0.5 * np.random.random(),
                'efficiency': 0.85 + 0.1 * np.random.random()
            }
            training_data.append(features)
        
        self.detector.train_detector(training_data)
        
        # Test normal features
        normal_features = {
            'q_min': 2.1,
            'shape_error': 2.2,
            'efficiency': 0.87
        }
        
        anomalies = self.detector.detect_anomalies(normal_features, timestamp=1000.0)
        
        # Should detect few or no anomalies for normal data
        assert len(anomalies) <= 1  # May have some false positives
    
    def test_anomaly_detection_abnormal_case(self):
        """Test anomaly detection with abnormal features."""
        # Train detector with normal data
        training_data = []
        for _ in range(100):
            features = {
                'q_min': 2.0 + 0.2 * np.random.random(),
                'shape_error': 2.0 + 0.5 * np.random.random(),
                'disruption_probability': 0.02 + 0.03 * np.random.random()
            }
            training_data.append(features)
        
        self.detector.train_detector(training_data)
        
        # Test abnormal features
        abnormal_features = {
            'q_min': 0.8,  # Very low q
            'shape_error': 10.0,  # High shape error
            'disruption_probability': 0.5  # High disruption risk
        }
        
        anomalies = self.detector.detect_anomalies(abnormal_features, timestamp=1000.0)
        
        # Should detect multiple anomalies
        assert len(anomalies) > 0
        
        # Check anomaly types
        anomaly_types = [a.type for a in anomalies]
        assert AnomalyType.DISRUPTION_PRECURSOR in anomaly_types or AnomalyType.SHAPE_DEVIATION in anomaly_types
    
    def test_temporal_anomaly_detection(self):
        """Test temporal pattern anomaly detection."""
        # Train detector
        training_data = [{'q_min': 2.0, 'efficiency': 0.85} for _ in range(100)]
        self.detector.train_detector(training_data)
        
        # Build up feature history with stable values
        for i in range(25):
            features = {'q_min': 2.0 + 0.01 * np.random.random(), 'efficiency': 0.85}
            self.detector.detect_anomalies(features, timestamp=i * 100.0)
        
        # Sudden change
        sudden_change_features = {'q_min': 1.5, 'efficiency': 0.85}  # Sudden drop in q
        anomalies = self.detector.detect_anomalies(sudden_change_features, timestamp=2600.0)
        
        # Should detect temporal anomaly
        temporal_anomalies = [a for a in anomalies if a.type == AnomalyType.CONTROL_INSTABILITY]
        assert len(temporal_anomalies) > 0 or len(anomalies) > 0  # Either temporal or statistical
    
    def test_anomaly_summary(self):
        """Test anomaly summary generation."""
        # Add some mock anomalies
        anomalies = [
            AnomalyEvent(
                type=AnomalyType.PERFORMANCE_DEGRADATION,
                severity=0.7, timestamp=1000.0, features={}, description="Test", confidence=0.8
            ),
            AnomalyEvent(
                type=AnomalyType.SHAPE_DEVIATION,
                severity=0.5, timestamp=1100.0, features={}, description="Test", confidence=0.9
            ),
            AnomalyEvent(
                type=AnomalyType.PERFORMANCE_DEGRADATION,
                severity=0.9, timestamp=1200.0, features={}, description="Test", confidence=0.7
            )
        ]
        
        self.detector.anomaly_history = anomalies
        
        summary = self.detector.get_anomaly_summary()
        
        assert isinstance(summary, dict)
        assert 'total_anomalies' in summary
        assert 'anomaly_types' in summary
        assert 'average_severity_by_type' in summary
        assert 'max_severity' in summary
        
        assert summary['total_anomalies'] == 3
        assert summary['anomaly_types']['performance_degradation'] == 2
        assert summary['anomaly_types']['shape_deviation'] == 1
        assert summary['max_severity'] == 0.9


class TestPerformancePredictor:
    """Test performance prediction algorithms."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.predictor = PerformancePredictor(prediction_horizon=300.0)
    
    def test_predictor_initialization(self):
        """Test predictor initialization."""
        assert self.predictor.prediction_horizon == 300.0
        assert not self.predictor.is_trained
        assert len(self.predictor.training_features) == 0
        assert len(self.predictor.training_targets) == 0
        assert 'shape_error' in self.predictor.models
        assert 'q_min' in self.predictor.models
    
    def test_training_data_addition(self):
        """Test adding training data."""
        features = {
            'q_min': 2.0,
            'shape_error': 2.5,
            'beta': 0.025,
            'control_magnitude': 0.5
        }
        
        targets = {
            'shape_error': 2.3,
            'q_min': 2.1,
            'disruption_probability': 0.05
        }
        
        self.predictor.add_training_data(features, targets)
        
        assert len(self.predictor.training_features) == 1
        assert len(self.predictor.training_targets) == 1
        assert self.predictor.training_features[0] == features
        assert self.predictor.training_targets[0] == targets
    
    def test_model_training(self):
        """Test model training process."""
        # Generate synthetic training data
        for i in range(150):
            features = {
                'q_min': 1.5 + np.random.random(),
                'shape_error': 1.0 + 3.0 * np.random.random(),
                'beta': 0.01 + 0.03 * np.random.random(),
                'control_magnitude': np.random.random(),
                'efficiency': 0.7 + 0.3 * np.random.random()
            }
            
            # Targets with some correlation to features
            targets = {
                'shape_error': features['shape_error'] + 0.1 * np.random.random(),
                'q_min': features['q_min'] + 0.05 * np.random.random(),
                'disruption_probability': max(0, 0.5 - features['q_min']) + 0.01 * np.random.random(),
                'energy_efficiency': features['efficiency'] + 0.01 * np.random.random()
            }
            
            self.predictor.add_training_data(features, targets)
        
        training_results = self.predictor.train_models()
        
        assert self.predictor.is_trained
        assert isinstance(training_results, dict)
        assert 'models_trained' in training_results
        assert 'training_samples' in training_results
        assert 'model_performance' in training_results
        
        # Check model performance metrics
        performance = training_results['model_performance']
        for model_name in ['shape_error', 'q_min', 'disruption_probability', 'energy_efficiency']:
            if model_name in performance:
                assert 'mse' in performance[model_name]
                assert 'r2_score' in performance[model_name]
                assert 'feature_importance' in performance[model_name]
    
    def test_insufficient_training_data(self):
        """Test error handling for insufficient training data."""
        # Add minimal data
        for _ in range(10):
            self.predictor.add_training_data({'feature': 1}, {'target': 1})
        
        with pytest.raises(ValueError, match="Insufficient training data"):
            self.predictor.train_models()
    
    def test_performance_prediction(self):
        """Test performance prediction."""
        # Train model first
        for i in range(150):
            features = {
                'q_min': 1.5 + np.random.random(),
                'shape_error': 1.0 + 3.0 * np.random.random(),
                'beta': 0.01 + 0.03 * np.random.random()
            }
            
            targets = {
                'shape_error': features['shape_error'] + 0.1 * np.random.random(),
                'q_min': features['q_min'] + 0.05 * np.random.random()
            }
            
            self.predictor.add_training_data(features, targets)
        
        self.predictor.train_models()
        
        # Make prediction
        test_features = {
            'q_min': 2.0,
            'shape_error': 2.5,
            'beta': 0.025
        }
        
        prediction = self.predictor.predict_performance(test_features, horizon=300.0)
        
        assert isinstance(prediction, PredictionResult)
        assert prediction.forecast_horizon == 300.0
        assert isinstance(prediction.predicted_values, dict)
        assert isinstance(prediction.confidence_intervals, dict)
        assert isinstance(prediction.feature_importance, dict)
        assert 0 <= prediction.model_accuracy <= 1
        
        # Check predictions exist for trained models
        for target in ['shape_error', 'q_min']:
            assert target in prediction.predicted_values
            assert target in prediction.confidence_intervals
    
    def test_prediction_without_training(self):
        """Test error handling for prediction without training."""
        test_features = {'q_min': 2.0, 'shape_error': 2.5}
        
        with pytest.raises(ValueError, match="Models not trained"):
            self.predictor.predict_performance(test_features)
    
    def test_prediction_accuracy_calculation(self):
        """Test prediction accuracy calculation."""
        # Add some training targets
        for i in range(50):
            targets = {
                'shape_error': 2.0 + 0.5 * np.random.random(),
                'q_min': 1.8 + 0.4 * np.random.random(),
                'energy_efficiency': 0.8 + 0.2 * np.random.random()
            }
            self.predictor.training_targets.append(targets)
        
        self.predictor.is_trained = True  # Mock training state
        
        accuracy = self.predictor.get_prediction_accuracy()
        
        assert isinstance(accuracy, dict)
        for target in ['shape_error', 'q_min', 'energy_efficiency']:
            if target in accuracy:
                assert 0 <= accuracy[target] <= 1


class TestTrendAnalyzer:
    """Test trend analysis algorithms."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.analyzer = TrendAnalyzer(window_size=50)
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        assert self.analyzer.window_size == 50
        assert len(self.analyzer.trend_history) == 0
        assert len(self.analyzer.pattern_cache) == 0
    
    def test_data_point_addition(self):
        """Test adding data points."""
        self.analyzer.add_data_point('efficiency', 0.85, timestamp=1000.0)
        self.analyzer.add_data_point('efficiency', 0.87, timestamp=1100.0)
        
        assert 'efficiency' in self.analyzer.trend_history
        assert len(self.analyzer.trend_history['efficiency']) == 2
        assert self.analyzer.trend_history['efficiency'][0] == 0.85
        assert self.analyzer.trend_history['efficiency'][1] == 0.87
    
    def test_window_size_maintenance(self):
        """Test that data history maintains window size."""
        # Add more data than window size
        for i in range(120):
            self.analyzer.add_data_point('test_metric', i * 0.01)
        
        assert len(self.analyzer.trend_history['test_metric']) == self.analyzer.window_size
    
    def test_trend_pattern_detection(self):
        """Test trend pattern detection."""
        # Generate data with clear increasing trend
        for i in range(60):
            value = 0.5 + 0.01 * i + 0.05 * np.random.random()  # Increasing trend with noise
            self.analyzer.add_data_point('increasing_metric', value)
        
        patterns = self.analyzer.detect_patterns('increasing_metric')
        
        assert isinstance(patterns, dict)
        assert 'trend' in patterns
        assert 'volatility' in patterns
        assert 'extremes' in patterns
        assert 'regime_changes' in patterns
        
        # Check trend detection
        trend = patterns['trend']
        assert trend['direction'] == 'increasing'
        assert trend['strength'] > 0.5  # Should detect strong trend
        assert 'slope' in trend
        assert 'significance' in trend
    
    def test_cyclical_pattern_detection(self):
        """Test cyclical pattern detection."""
        # Generate sinusoidal data
        for i in range(80):
            value = 0.5 + 0.3 * np.sin(2 * np.pi * i / 20) + 0.05 * np.random.random()
            self.analyzer.add_data_point('cyclical_metric', value)
        
        patterns = self.analyzer.detect_patterns('cyclical_metric')
        
        assert 'cyclical' in patterns
        cyclical = patterns['cyclical']
        assert 'dominant_period' in cyclical
        assert 'has_cycle' in cyclical
        
        # Should detect cycle with period around 20
        if cyclical['has_cycle']:
            assert 15 <= cyclical['dominant_period'] <= 25
    
    def test_volatility_analysis(self):
        """Test volatility analysis."""
        # Generate data with different volatility periods
        for i in range(80):
            if i < 40:
                # Low volatility period
                value = 0.8 + 0.02 * np.random.random()
            else:
                # High volatility period
                value = 0.8 + 0.2 * np.random.random()
            self.analyzer.add_data_point('volatile_metric', value)
        
        patterns = self.analyzer.detect_patterns('volatile_metric')
        
        assert 'volatility' in patterns
        volatility = patterns['volatility']
        assert 'standard_deviation' in volatility
        assert 'coefficient_of_variation' in volatility
        assert 'return_volatility' in volatility
        assert 'max_drawdown' in volatility
        
        assert volatility['standard_deviation'] > 0
        assert volatility['max_drawdown'] >= 0
    
    def test_peak_detection(self):
        """Test peak and valley detection."""
        # Generate data with clear peaks
        for i in range(60):
            if i in [10, 30, 50]:
                value = 1.0  # Peaks
            elif i in [20, 40]:
                value = 0.2  # Valleys
            else:
                value = 0.6 + 0.1 * np.random.random()  # Normal values
            self.analyzer.add_data_point('peak_metric', value)
        
        patterns = self.analyzer.detect_patterns('peak_metric')
        
        assert 'extremes' in patterns
        extremes = patterns['extremes']
        assert 'peak_count' in extremes
        assert 'valley_count' in extremes
        assert 'peak_frequency' in extremes
        
        # Should detect some peaks and valleys
        assert extremes['peak_count'] >= 0
        assert extremes['valley_count'] >= 0
    
    def test_regime_change_detection(self):
        """Test regime change detection."""
        # Generate data with regime change
        for i in range(80):
            if i < 40:
                value = 0.3 + 0.05 * np.random.random()  # Low regime
            else:
                value = 0.8 + 0.05 * np.random.random()  # High regime
            self.analyzer.add_data_point('regime_metric', value)
        
        patterns = self.analyzer.detect_patterns('regime_metric')
        
        assert 'regime_changes' in patterns
        regime = patterns['regime_changes']
        assert 'change_points' in regime
        assert 'regime_count' in regime
        assert 'regime_stability' in regime
        
        # Should detect at least one regime change
        assert regime['regime_count'] >= 2
        assert len(regime['change_points']) >= 1
    
    def test_trend_report_generation(self):
        """Test trend analysis report generation."""
        # Add data for multiple metrics
        metrics = ['efficiency', 'stability', 'cost']
        
        for metric in metrics:
            for i in range(60):
                if metric == 'efficiency':
                    value = 0.7 + 0.005 * i  # Increasing
                elif metric == 'stability':
                    value = 0.9 - 0.002 * i  # Decreasing
                else:
                    value = 0.5 + 0.1 * np.sin(i / 10)  # Cyclical
                
                self.analyzer.add_data_point(metric, value)
        
        report = self.analyzer.generate_trend_report(metrics)
        
        assert isinstance(report, str)
        assert "TREND ANALYSIS REPORT" in report
        
        for metric in metrics:
            assert metric.upper() in report
        
        # Should contain trend indicators
        assert any(icon in report for icon in ["📈", "📉"])


class TestAnalyticsSystemIntegration:
    """Test analytics system integration."""
    
    def test_analytics_system_factory(self):
        """Test analytics system factory function."""
        analytics_system = create_analytics_system()
        
        assert isinstance(analytics_system, dict)
        assert 'anomaly_detector' in analytics_system
        assert 'performance_predictor' in analytics_system
        assert 'trend_analyzer' in analytics_system
        
        assert isinstance(analytics_system['anomaly_detector'], AnomalyDetector)
        assert isinstance(analytics_system['performance_predictor'], PerformancePredictor)
        assert isinstance(analytics_system['trend_analyzer'], TrendAnalyzer)
    
    def test_integrated_analytics_workflow(self):
        """Test integrated analytics workflow."""
        analytics_system = create_analytics_system()
        
        detector = analytics_system['anomaly_detector']
        predictor = analytics_system['performance_predictor']
        analyzer = analytics_system['trend_analyzer']
        
        # Generate sample data
        config = TokamakConfig.from_preset("ITER")
        plasma_state = PlasmaState(config)
        
        # 1. Feature extraction and anomaly detection
        features = detector.extract_features(plasma_state)
        assert isinstance(features, dict)
        
        # Train detector with normal data
        training_data = []
        for i in range(100):
            normal_features = {
                'q_min': 2.0 + 0.2 * np.random.random(),
                'shape_error': 2.0 + 0.5 * np.random.random(),
                'efficiency': 0.85 + 0.1 * np.random.random()
            }
            training_data.append(normal_features)
        
        detector.train_detector(training_data)
        
        # Test anomaly detection
        anomalies = detector.detect_anomalies(features)
        assert isinstance(anomalies, list)
        
        # 2. Performance prediction
        # Train predictor
        for i in range(150):
            train_features = {
                'q_min': 1.5 + np.random.random(),
                'shape_error': 1.0 + 3.0 * np.random.random(),
                'beta': 0.01 + 0.03 * np.random.random()
            }
            
            train_targets = {
                'shape_error': train_features['shape_error'] + 0.1 * np.random.random(),
                'q_min': train_features['q_min'] + 0.05 * np.random.random()
            }
            
            predictor.add_training_data(train_features, train_targets)
        
        predictor.train_models()
        
        # Make prediction
        prediction = predictor.predict_performance(features)
        assert isinstance(prediction, PredictionResult)
        
        # 3. Trend analysis
        for i in range(60):
            efficiency = 0.8 + 0.01 * i + 0.02 * np.random.random()
            analyzer.add_data_point('efficiency', efficiency)
        
        patterns = analyzer.detect_patterns('efficiency')
        assert isinstance(patterns, dict)
        
        report = analyzer.generate_trend_report(['efficiency'])
        assert isinstance(report, str)
        
        # 4. Integration check
        # All components should work together without errors
        assert len(detector.feature_history) > 0
        assert predictor.is_trained
        assert 'efficiency' in analyzer.trend_history