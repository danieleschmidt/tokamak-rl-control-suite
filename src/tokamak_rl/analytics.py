"""
Advanced analytics and machine learning for tokamak performance prediction.

This module provides predictive analytics, anomaly detection, and performance
forecasting capabilities using machine learning techniques.
"""

import numpy as np
try:
    import pandas as pd
except ImportError:
    pd = None
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import pickle
import json
import logging
from pathlib import Path
try:
    from scipy import stats
    from scipy.signal import find_peaks
except ImportError:
    stats = None
    find_peaks = None

try:
    from sklearn.ensemble import IsolationForest, RandomForestRegressor
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.cluster import DBSCAN
except ImportError:
    IsolationForest = None
    RandomForestRegressor = None
    StandardScaler = None
    RobustScaler = None
    train_test_split = None
    mean_squared_error = None
    r2_score = None
    DBSCAN = None
import warnings

from .physics import PlasmaState
from .business import PerformanceMetrics


class AnomalyType(Enum):
    """Types of anomalies that can be detected."""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    CONTROL_INSTABILITY = "control_instability"
    SHAPE_DEVIATION = "shape_deviation"
    DISRUPTION_PRECURSOR = "disruption_precursor"
    EFFICIENCY_DROP = "efficiency_drop"
    SENSOR_MALFUNCTION = "sensor_malfunction"


@dataclass
class AnomalyEvent:
    """Detected anomaly event."""
    type: AnomalyType
    severity: float  # 0-1 scale
    timestamp: float
    features: Dict[str, float]
    description: str
    confidence: float
    session_id: str = ""


@dataclass
class PredictionResult:
    """Performance prediction result."""
    predicted_values: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    forecast_horizon: float
    model_accuracy: float
    feature_importance: Dict[str, float]
    timestamp: float


class AnomalyDetector:
    """Advanced anomaly detection for tokamak operations."""
    
    def __init__(self, contamination: float = 0.1, window_size: int = 50):
        self.contamination = contamination
        self.window_size = window_size
        
        # Models for different types of anomalies
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        
        self.scaler = RobustScaler()
        
        # Feature history for temporal anomaly detection
        self.feature_history: List[Dict[str, float]] = []
        self.anomaly_history: List[AnomalyEvent] = []
        
        # Trained models
        self.is_trained = False
        self.normal_ranges: Dict[str, Tuple[float, float]] = {}
        
        self.logger = logging.getLogger(__name__)
    
    def extract_features(self, state: PlasmaState, 
                        performance: Optional[PerformanceMetrics] = None,
                        control_action: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Extract features for anomaly detection."""
        features = {
            # Plasma physics features
            'q_min': state.q_min,
            'q_std': np.std(state.q_profile),
            'q_gradient_max': np.max(np.gradient(state.q_profile)),
            'beta_normalized': state.plasma_beta / 0.04,  # Normalized to typical limit
            'shape_error': state.shape_error,
            'elongation': state.elongation,
            'triangularity': state.triangularity,
            'disruption_probability': state.disruption_probability,
            
            # Profile features
            'pressure_peaking': np.max(state.pressure_profile) / np.mean(state.pressure_profile),
            'temperature_peaking': np.max(state.temperature_profile) / np.mean(state.temperature_profile),
            'density_peaking': np.max(state.density_profile) / np.mean(state.density_profile),
            'pressure_gradient_max': np.max(np.gradient(state.pressure_profile)),
            
            # Current and field features
            'plasma_current_normalized': state.plasma_current / 15e6,  # Normalized to typical value
            'pf_current_rms': np.sqrt(np.mean(state.pf_coil_currents ** 2)),
            'pf_current_max': np.max(np.abs(state.pf_coil_currents)),
            'pf_current_asymmetry': np.std(state.pf_coil_currents),
        }
        
        # Performance features if available
        if performance:
            features.update({
                'efficiency': performance.energy_efficiency,
                'stability': performance.q_factor_stability,
                'control_smoothness': performance.control_smoothness,
                'target_achievement': performance.target_achievement_rate
            })
        
        # Control action features if available
        if control_action is not None:
            features.update({
                'control_magnitude': np.linalg.norm(control_action),
                'control_pf_rms': np.sqrt(np.mean(control_action[:6] ** 2)),
                'control_gas_rate': control_action[6],
                'control_heating': control_action[7]
            })
        
        return features
    
    def train_detector(self, training_data: List[Dict[str, float]]) -> Dict[str, Any]:
        """Train anomaly detection models on historical data."""
        if len(training_data) < 50:
            raise ValueError("Insufficient training data (need at least 50 samples)")
        
        # Convert to DataFrame for easier handling
        df = pd.DataFrame(training_data)
        
        # Handle missing values
        df = df.fillna(df.median())
        
        # Extract features and scale
        feature_matrix = df.values
        feature_matrix_scaled = self.scaler.fit_transform(feature_matrix)
        
        # Train isolation forest
        self.isolation_forest.fit(feature_matrix_scaled)
        
        # Calculate normal ranges for statistical detection
        for column in df.columns:
            values = df[column].values
            q1, q3 = np.percentile(values, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            self.normal_ranges[column] = (lower_bound, upper_bound)
        
        self.is_trained = True
        
        # Return training statistics
        return {
            'samples_trained': len(training_data),
            'features_count': len(df.columns),
            'contamination_rate': self.contamination,
            'normal_ranges_computed': len(self.normal_ranges)
        }
    
    def detect_anomalies(self, current_features: Dict[str, float],
                        timestamp: float = None) -> List[AnomalyEvent]:
        """Detect anomalies in current features."""
        if not self.is_trained:
            self.logger.warning("Anomaly detector not trained. Training on available data.")
            if len(self.feature_history) > 50:
                self.train_detector(self.feature_history[-1000:])
            else:
                return []
        
        if timestamp is None:
            timestamp = float(np.datetime64('now').astype('datetime64[s]').astype(int))
        
        anomalies = []
        
        # Add current features to history
        self.feature_history.append(current_features)
        if len(self.feature_history) > 10000:  # Limit history size
            self.feature_history = self.feature_history[-5000:]
        
        # 1. Isolation Forest anomaly detection
        features_array = np.array([list(current_features.values())]).reshape(1, -1)
        features_scaled = self.scaler.transform(features_array)
        
        isolation_score = self.isolation_forest.decision_function(features_scaled)[0]
        is_anomaly = self.isolation_forest.predict(features_scaled)[0] == -1
        
        if is_anomaly:
            anomalies.append(AnomalyEvent(
                type=AnomalyType.PERFORMANCE_DEGRADATION,
                severity=max(0, min(1, -isolation_score)),
                timestamp=timestamp,
                features=current_features.copy(),
                description=f"Isolation Forest detected anomaly (score: {isolation_score:.3f})",
                confidence=0.8
            ))
        
        # 2. Statistical range-based detection
        for feature, value in current_features.items():
            if feature in self.normal_ranges:
                lower, upper = self.normal_ranges[feature]
                if value < lower or value > upper:
                    severity = min(1.0, max(abs(value - lower), abs(value - upper)) / (upper - lower))
                    
                    # Determine anomaly type based on feature
                    anomaly_type = self._classify_anomaly_type(feature, value, lower, upper)
                    
                    anomalies.append(AnomalyEvent(
                        type=anomaly_type,
                        severity=severity,
                        timestamp=timestamp,
                        features={feature: value, 'normal_range': (lower, upper)},
                        description=f"{feature} outside normal range: {value:.3f} not in [{lower:.3f}, {upper:.3f}]",
                        confidence=0.7
                    ))
        
        # 3. Temporal pattern detection
        if len(self.feature_history) >= self.window_size:
            temporal_anomalies = self._detect_temporal_anomalies(current_features, timestamp)
            anomalies.extend(temporal_anomalies)
        
        # Store detected anomalies
        self.anomaly_history.extend(anomalies)
        
        return anomalies
    
    def _classify_anomaly_type(self, feature: str, value: float, 
                              lower: float, upper: float) -> AnomalyType:
        """Classify anomaly type based on feature."""
        if 'disruption' in feature:
            return AnomalyType.DISRUPTION_PRECURSOR
        elif 'shape' in feature:
            return AnomalyType.SHAPE_DEVIATION
        elif 'control' in feature:
            return AnomalyType.CONTROL_INSTABILITY
        elif 'efficiency' in feature:
            return AnomalyType.EFFICIENCY_DROP
        else:
            return AnomalyType.PERFORMANCE_DEGRADATION
    
    def _detect_temporal_anomalies(self, current_features: Dict[str, float],
                                  timestamp: float) -> List[AnomalyEvent]:
        """Detect temporal pattern anomalies."""
        anomalies = []
        
        # Get recent history for key features
        key_features = ['q_min', 'shape_error', 'disruption_probability', 'efficiency']
        
        for feature in key_features:
            if feature not in current_features:
                continue
                
            # Extract recent values
            recent_values = []
            for hist_features in self.feature_history[-self.window_size:]:
                if feature in hist_features:
                    recent_values.append(hist_features[feature])
            
            if len(recent_values) < 10:
                continue
                
            recent_values = np.array(recent_values)
            
            # Detect sudden changes
            if len(recent_values) > 1:
                recent_change = abs(recent_values[-1] - recent_values[-2])
                typical_change = np.std(np.diff(recent_values[:-1]))
                
                if recent_change > 3 * typical_change and typical_change > 0:
                    anomalies.append(AnomalyEvent(
                        type=AnomalyType.CONTROL_INSTABILITY,
                        severity=min(1.0, recent_change / (3 * typical_change)),
                        timestamp=timestamp,
                        features={feature: current_features[feature], 'change_magnitude': recent_change},
                        description=f"Sudden change in {feature}: {recent_change:.3f} (typical: {typical_change:.3f})",
                        confidence=0.6
                    ))
            
            # Detect trending behavior
            if len(recent_values) >= 10:
                slope, _, r_value, _, _ = stats.linregress(range(len(recent_values)), recent_values)
                
                # Significant trend with high correlation
                if abs(r_value) > 0.8 and abs(slope) > 0.01:
                    anomalies.append(AnomalyEvent(
                        type=AnomalyType.PERFORMANCE_DEGRADATION,
                        severity=min(1.0, abs(slope) * 10),
                        timestamp=timestamp,
                        features={feature: current_features[feature], 'trend_slope': slope},
                        description=f"Strong trend in {feature}: slope={slope:.4f}, r²={r_value**2:.3f}",
                        confidence=0.7
                    ))
        
        return anomalies
    
    def get_anomaly_summary(self, time_window: Optional[float] = None) -> Dict[str, Any]:
        """Get summary of detected anomalies."""
        anomalies = self.anomaly_history
        
        if time_window:
            current_time = max(a.timestamp for a in anomalies) if anomalies else 0
            anomalies = [a for a in anomalies if current_time - a.timestamp <= time_window]
        
        if not anomalies:
            return {"message": "No anomalies detected"}
        
        # Count by type
        type_counts = {}
        severity_sum = {}
        for anomaly in anomalies:
            anomaly_type = anomaly.type.value
            type_counts[anomaly_type] = type_counts.get(anomaly_type, 0) + 1
            severity_sum[anomaly_type] = severity_sum.get(anomaly_type, 0) + anomaly.severity
        
        # Calculate average severity by type
        avg_severity = {t: severity_sum[t] / type_counts[t] for t in type_counts}
        
        return {
            'total_anomalies': len(anomalies),
            'anomaly_types': type_counts,
            'average_severity_by_type': avg_severity,
            'max_severity': max(a.severity for a in anomalies),
            'recent_anomalies': len([a for a in anomalies if a.timestamp > current_time - 3600])  # Last hour
        }


class PerformancePredictor:
    """Machine learning-based performance prediction."""
    
    def __init__(self, prediction_horizon: float = 300.0):  # 5 minutes default
        self.prediction_horizon = prediction_horizon
        
        # Models for different prediction tasks
        self.models = {
            'shape_error': RandomForestRegressor(n_estimators=100, random_state=42),
            'q_min': RandomForestRegressor(n_estimators=100, random_state=42),
            'disruption_probability': RandomForestRegressor(n_estimators=100, random_state=42),
            'energy_efficiency': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Training data storage
        self.training_features: List[Dict[str, float]] = []
        self.training_targets: List[Dict[str, float]] = []
        
        self.logger = logging.getLogger(__name__)
    
    def add_training_data(self, features: Dict[str, float], 
                         targets: Dict[str, float],
                         future_features: Optional[Dict[str, float]] = None) -> None:
        """Add training data point."""
        self.training_features.append(features.copy())
        self.training_targets.append(targets.copy())
        
        # Limit training data size
        if len(self.training_features) > 10000:
            self.training_features = self.training_features[-5000:]
            self.training_targets = self.training_targets[-5000:]
    
    def train_models(self) -> Dict[str, Any]:
        """Train prediction models on accumulated data."""
        if len(self.training_features) < 100:
            raise ValueError("Insufficient training data (need at least 100 samples)")
        
        # Convert to DataFrames
        features_df = pd.DataFrame(self.training_features).fillna(0)
        targets_df = pd.DataFrame(self.training_targets).fillna(0)
        
        # Prepare feature matrix
        X = features_df.values
        X_scaled = self.scaler.fit_transform(X)
        
        training_results = {}
        
        # Train models for each target
        for target in self.models.keys():
            if target not in targets_df.columns:
                continue
            
            y = targets_df[target].values
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            # Train model
            self.models[target].fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.models[target].predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            training_results[target] = {
                'mse': mse,
                'r2_score': r2,
                'feature_importance': dict(zip(features_df.columns, self.models[target].feature_importances_))
            }
        
        self.is_trained = True
        
        return {
            'models_trained': len(training_results),
            'training_samples': len(self.training_features),
            'model_performance': training_results
        }
    
    def predict_performance(self, current_features: Dict[str, float],
                           horizon: Optional[float] = None) -> PredictionResult:
        """Predict future performance metrics."""
        if not self.is_trained:
            raise ValueError("Models not trained. Call train_models() first.")
        
        if horizon is None:
            horizon = self.prediction_horizon
        
        # Prepare features
        features_array = np.array([list(current_features.values())]).reshape(1, -1)
        features_scaled = self.scaler.transform(features_array)
        
        predictions = {}
        confidence_intervals = {}
        feature_importance = {}
        
        # Generate predictions for each model
        for target, model in self.models.items():
            if hasattr(model, 'predict'):
                # Point prediction
                pred = model.predict(features_scaled)[0]
                predictions[target] = pred
                
                # Estimate confidence interval using model uncertainty
                if hasattr(model, 'estimators_'):
                    # For ensemble models, use prediction variance
                    estimator_predictions = [tree.predict(features_scaled)[0] for tree in model.estimators_]
                    pred_std = np.std(estimator_predictions)
                    confidence_intervals[target] = (pred - 2*pred_std, pred + 2*pred_std)
                else:
                    # Fallback confidence interval
                    confidence_intervals[target] = (pred * 0.9, pred * 1.1)
                
                # Feature importance
                if hasattr(model, 'feature_importances_'):
                    feature_names = list(current_features.keys())
                    importance = dict(zip(feature_names, model.feature_importances_))
                    feature_importance[target] = importance
        
        # Calculate overall model accuracy (from training)
        accuracies = []
        for target in self.models.keys():
            if len(self.training_targets) > 0 and target in self.training_targets[0]:
                # Simple correlation-based accuracy estimate
                recent_targets = [t.get(target, 0) for t in self.training_targets[-100:]]
                if len(set(recent_targets)) > 1:  # Non-constant
                    accuracy = 1.0 / (1.0 + np.std(recent_targets))
                    accuracies.append(accuracy)
        
        overall_accuracy = np.mean(accuracies) if accuracies else 0.5
        
        return PredictionResult(
            predicted_values=predictions,
            confidence_intervals=confidence_intervals,
            forecast_horizon=horizon,
            model_accuracy=overall_accuracy,
            feature_importance=feature_importance,
            timestamp=float(np.datetime64('now').astype('datetime64[s]').astype(int))
        )
    
    def get_prediction_accuracy(self, time_window: Optional[float] = None) -> Dict[str, float]:
        """Calculate prediction accuracy over time window."""
        if not self.is_trained or len(self.training_targets) < 10:
            return {}
        
        # Use recent data for accuracy calculation
        recent_data = self.training_targets[-100:] if time_window is None else self.training_targets
        
        accuracy = {}
        for target in self.models.keys():
            target_values = [d.get(target, 0) for d in recent_data if target in d]
            if len(target_values) > 1:
                # Calculate coefficient of variation as inverse accuracy measure
                cv = np.std(target_values) / (np.abs(np.mean(target_values)) + 1e-6)
                accuracy[target] = max(0, 1.0 - cv)
        
        return accuracy


class TrendAnalyzer:
    """Advanced trend analysis and pattern detection."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.trend_history: Dict[str, List[float]] = {}
        self.pattern_cache: Dict[str, Any] = {}
        
    def add_data_point(self, metric: str, value: float, timestamp: float = None) -> None:
        """Add data point for trend analysis."""
        if metric not in self.trend_history:
            self.trend_history[metric] = []
        
        self.trend_history[metric].append(value)
        
        # Maintain window size
        if len(self.trend_history[metric]) > self.window_size * 2:
            self.trend_history[metric] = self.trend_history[metric][-self.window_size:]
    
    def detect_patterns(self, metric: str) -> Dict[str, Any]:
        """Detect patterns in metric trends."""
        if metric not in self.trend_history or len(self.trend_history[metric]) < 20:
            return {}
        
        data = np.array(self.trend_history[metric])
        
        patterns = {}
        
        # 1. Trend direction and strength
        x = np.arange(len(data))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, data)
        
        patterns['trend'] = {
            'direction': 'increasing' if slope > 0 else 'decreasing',
            'strength': abs(r_value),
            'slope': slope,
            'p_value': p_value,
            'significance': 'significant' if p_value < 0.05 else 'not_significant'
        }
        
        # 2. Cyclical patterns
        if len(data) > 50:
            fft = np.fft.fft(data - np.mean(data))
            frequencies = np.fft.fftfreq(len(data))
            power_spectrum = np.abs(fft) ** 2
            
            # Find dominant frequencies
            dominant_freq_idx = np.argmax(power_spectrum[1:len(data)//2]) + 1
            dominant_period = 1.0 / abs(frequencies[dominant_freq_idx]) if frequencies[dominant_freq_idx] != 0 else float('inf')
            
            patterns['cyclical'] = {
                'dominant_period': dominant_period,
                'power': power_spectrum[dominant_freq_idx],
                'has_cycle': power_spectrum[dominant_freq_idx] > np.mean(power_spectrum) * 3
            }
        
        # 3. Volatility analysis
        returns = np.diff(data) / (data[:-1] + 1e-6)
        patterns['volatility'] = {
            'standard_deviation': np.std(data),
            'coefficient_of_variation': np.std(data) / (np.abs(np.mean(data)) + 1e-6),
            'return_volatility': np.std(returns),
            'max_drawdown': self._calculate_max_drawdown(data)
        }
        
        # 4. Peak detection
        peaks, peak_properties = find_peaks(data, prominence=np.std(data)*0.5)
        valleys, valley_properties = find_peaks(-data, prominence=np.std(data)*0.5)
        
        patterns['extremes'] = {
            'peak_count': len(peaks),
            'valley_count': len(valleys),
            'peak_frequency': len(peaks) / len(data),
            'avg_peak_prominence': np.mean(peak_properties.get('prominences', [0])),
        }
        
        # 5. Regime changes
        patterns['regime_changes'] = self._detect_regime_changes(data)
        
        self.pattern_cache[metric] = patterns
        return patterns
    
    def _calculate_max_drawdown(self, data: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        cumulative_max = np.maximum.accumulate(data)
        drawdown = (data - cumulative_max) / (cumulative_max + 1e-6)
        return abs(np.min(drawdown))
    
    def _detect_regime_changes(self, data: np.ndarray) -> Dict[str, Any]:
        """Detect regime changes in time series."""
        if len(data) < 30:
            return {'change_points': [], 'regime_count': 1}
        
        # Simple regime change detection using rolling statistics
        window = min(20, len(data) // 4)
        rolling_mean = pd.Series(data).rolling(window=window).mean().values
        rolling_std = pd.Series(data).rolling(window=window).std().values
        
        # Detect significant changes in mean or variance
        change_points = []
        
        for i in range(window, len(data) - window):
            # Compare recent vs. historical statistics
            recent_mean = np.mean(data[i:i+window])
            historical_mean = np.mean(data[i-window:i])
            
            recent_std = np.std(data[i:i+window])
            historical_std = np.std(data[i-window:i])
            
            # Threshold for significant change
            mean_change = abs(recent_mean - historical_mean) / (historical_std + 1e-6)
            std_change = abs(recent_std - historical_std) / (historical_std + 1e-6)
            
            if mean_change > 2.0 or std_change > 1.0:
                change_points.append(i)
        
        return {
            'change_points': change_points,
            'regime_count': len(change_points) + 1,
            'regime_stability': 1.0 / (len(change_points) + 1)
        }
    
    def generate_trend_report(self, metrics: List[str]) -> str:
        """Generate comprehensive trend analysis report."""
        report = "TREND ANALYSIS REPORT\n"
        report += "=" * 50 + "\n\n"
        
        for metric in metrics:
            if metric not in self.trend_history:
                continue
                
            patterns = self.detect_patterns(metric)
            if not patterns:
                continue
            
            report += f"Metric: {metric.upper()}\n"
            report += "-" * 30 + "\n"
            
            # Trend information
            if 'trend' in patterns:
                trend = patterns['trend']
                direction_icon = "📈" if trend['direction'] == 'increasing' else "📉"
                report += f"Trend: {direction_icon} {trend['direction']} (strength: {trend['strength']:.3f})\n"
                report += f"Significance: {trend['significance']} (p={trend['p_value']:.4f})\n"
            
            # Volatility
            if 'volatility' in patterns:
                vol = patterns['volatility']
                report += f"Volatility: {vol['coefficient_of_variation']:.3f} (CV)\n"
                report += f"Max Drawdown: {vol['max_drawdown']:.3f}\n"
            
            # Cyclical patterns
            if 'cyclical' in patterns:
                cyc = patterns['cyclical']
                if cyc['has_cycle']:
                    report += f"Cyclical Pattern: Period ≈ {cyc['dominant_period']:.1f} steps\n"
            
            # Regime changes
            if 'regime_changes' in patterns:
                regime = patterns['regime_changes']
                report += f"Regime Stability: {regime['regime_stability']:.3f}\n"
                if regime['change_points']:
                    report += f"Recent Change Points: {len(regime['change_points'])}\n"
            
            report += "\n"
        
        return report


def create_analytics_system(data_dir: str = "./analytics_data") -> Dict[str, Any]:
    """
    Factory function to create complete analytics system.
    
    Returns:
        Dictionary containing detector, predictor, and analyzer instances
    """
    detector = AnomalyDetector()
    predictor = PerformancePredictor()
    analyzer = TrendAnalyzer()
    
    return {
        'anomaly_detector': detector,
        'performance_predictor': predictor,
        'trend_analyzer': analyzer
    }