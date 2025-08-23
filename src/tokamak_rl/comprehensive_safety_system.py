"""
Comprehensive Safety System for Tokamak Plasma Control

This module implements multi-layered safety systems including:
- Real-time disruption prediction and mitigation
- Hardware safety interlocks and emergency shutdown
- Predictive maintenance and component health monitoring
- Safety-critical control validation
"""

import math
import time
import random
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import threading
import logging

# Configure safety logger
safety_logger = logging.getLogger('tokamak_safety')
safety_logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - SAFETY - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
safety_logger.addHandler(handler)


class SafetyLevel(Enum):
    """Safety protection levels."""
    LEVEL_0 = "NORMAL_OPERATION"
    LEVEL_1 = "ENHANCED_MONITORING"
    LEVEL_2 = "AUTOMATED_INTERVENTION"
    LEVEL_3 = "CONTROLLED_SHUTDOWN"
    LEVEL_4 = "EMERGENCY_SHUTDOWN"


class SafetyStatus(Enum):
    """Safety system status."""
    OPERATIONAL = "OPERATIONAL"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"
    MAINTENANCE = "MAINTENANCE"


@dataclass
class SafetyLimit:
    """Definition of a safety limit with multiple thresholds."""
    parameter_name: str
    normal_max: float
    warning_threshold: float
    alarm_threshold: float
    trip_threshold: float
    unit: str
    description: str
    
    def evaluate(self, value: float) -> Tuple[SafetyLevel, str]:
        """Evaluate safety level based on parameter value."""
        if value <= self.normal_max:
            return SafetyLevel.LEVEL_0, "Normal operation"
        elif value <= self.warning_threshold:
            return SafetyLevel.LEVEL_1, "Warning threshold exceeded"
        elif value <= self.alarm_threshold:
            return SafetyLevel.LEVEL_2, "Alarm threshold exceeded - automated intervention required"
        elif value <= self.trip_threshold:
            return SafetyLevel.LEVEL_3, "Trip threshold exceeded - controlled shutdown initiated"
        else:
            return SafetyLevel.LEVEL_4, "Emergency threshold exceeded - immediate shutdown"


@dataclass
class DisruptionPrediction:
    """Disruption prediction results."""
    probability: float
    time_to_disruption: Optional[float]
    confidence: float
    contributing_factors: List[str]
    mitigation_recommendations: List[str]
    prediction_timestamp: float = field(default_factory=time.time)


@dataclass
class ComponentHealth:
    """Component health assessment."""
    component_name: str
    health_score: float  # 0-100
    remaining_lifetime: Optional[float]  # hours
    failure_probability: float  # 0-1
    maintenance_required: bool
    critical_issues: List[str]
    recommendations: List[str]


class DisruptionPredictor:
    """
    Advanced disruption prediction system using multiple indicators
    and machine learning algorithms.
    """
    
    def __init__(self):
        self.prediction_history = deque(maxlen=1000)
        self.disruption_indicators = self._initialize_indicators()
        self.ml_model_weights = [random.gauss(0, 0.1) for _ in range(50)]
        self.feature_history = deque(maxlen=100)
        
    def _initialize_indicators(self) -> Dict[str, Callable]:
        """Initialize physics-based disruption indicators."""
        return {
            'beta_limit': self._beta_limit_indicator,
            'locked_mode': self._locked_mode_indicator,
            'density_limit': self._density_limit_indicator,
            'impurity_accumulation': self._impurity_indicator,
            'mhd_activity': self._mhd_activity_indicator,
            'error_field': self._error_field_indicator,
            'profile_consistency': self._profile_consistency_indicator
        }
    
    def predict_disruption(self, plasma_state: Dict[str, Any], 
                          mhd_modes: List[Any] = None) -> DisruptionPrediction:
        """
        Predict disruption probability using ensemble of methods.
        """
        # Extract features
        features = self._extract_features(plasma_state, mhd_modes)
        self.feature_history.append(features)
        
        # Physics-based indicators
        physics_predictions = {}
        contributing_factors = []
        
        for indicator_name, indicator_func in self.disruption_indicators.items():
            prob, factors = indicator_func(plasma_state, mhd_modes)
            physics_predictions[indicator_name] = prob
            if prob > 0.3:  # Significant contribution
                contributing_factors.extend(factors)
        
        # Machine learning prediction
        ml_prediction = self._ml_predict(features)
        
        # Ensemble prediction
        physics_avg = sum(physics_predictions.values()) / len(physics_predictions)
        ensemble_probability = 0.6 * physics_avg + 0.4 * ml_prediction
        
        # Time to disruption estimation
        time_to_disruption = self._estimate_time_to_disruption(ensemble_probability, features)
        
        # Confidence assessment
        confidence = self._calculate_confidence(physics_predictions, ml_prediction)
        
        # Generate mitigation recommendations
        mitigation_recommendations = self._generate_mitigation_recommendations(
            ensemble_probability, contributing_factors
        )
        
        prediction = DisruptionPrediction(
            probability=ensemble_probability,
            time_to_disruption=time_to_disruption,
            confidence=confidence,
            contributing_factors=contributing_factors,
            mitigation_recommendations=mitigation_recommendations
        )
        
        self.prediction_history.append(prediction)
        return prediction
    
    def _extract_features(self, plasma_state: Dict[str, Any], 
                         mhd_modes: List[Any] = None) -> List[float]:
        """Extract features for ML prediction."""
        features = []
        
        # Basic plasma parameters
        features.append(plasma_state.get('plasma_current', 1.0))
        features.append(plasma_state.get('beta_n', 0.02))
        features.append(plasma_state.get('q_min', 1.0))
        features.append(plasma_state.get('density_avg', 1e19))
        features.append(plasma_state.get('temp_avg', 10.0))
        
        # Profile characteristics
        if 'q_profile' in plasma_state:
            q_profile = plasma_state['q_profile']
            features.append(min(q_profile))  # q_min
            features.append(max(q_profile))  # q_edge
            features.append(sum((q_profile[i+1] - q_profile[i])**2 for i in range(len(q_profile)-1)))  # q roughness
        else:
            features.extend([1.0, 3.0, 0.1])
        
        # MHD mode features
        if mhd_modes:
            features.append(len(mhd_modes))
            features.append(max(mode.growth_rate for mode in mhd_modes) if mhd_modes else 0.0)
            features.append(sum(mode.amplitude for mode in mhd_modes))
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # Time derivatives (if history available)
        if len(self.feature_history) >= 2:
            prev_features = self.feature_history[-1]
            for i in range(min(len(features), len(prev_features))):
                features.append(features[i] - prev_features[i])  # Derivative
        else:
            features.extend([0.0] * len(features))
        
        return features[:20]  # Limit feature vector size
    
    def _beta_limit_indicator(self, plasma_state: Dict[str, Any], 
                             mhd_modes: List[Any] = None) -> Tuple[float, List[str]]:
        """Beta limit disruption indicator."""
        beta_n = plasma_state.get('beta_n', 0.02)
        beta_limit = 3.0  # Typical beta limit in %
        
        if beta_n > beta_limit * 0.9:
            prob = min(1.0, (beta_n - beta_limit * 0.9) / (beta_limit * 0.1))
            return prob, [f"Beta limit approach: βN = {beta_n:.2f}%"]
        
        return 0.0, []
    
    def _locked_mode_indicator(self, plasma_state: Dict[str, Any], 
                              mhd_modes: List[Any] = None) -> Tuple[float, List[str]]:
        """Locked mode disruption indicator."""
        if not mhd_modes:
            return 0.0, []
        
        locked_modes = [mode for mode in mhd_modes if abs(mode.frequency) < 100]  # Nearly stationary
        
        if locked_modes:
            max_amplitude = max(mode.amplitude for mode in locked_modes)
            if max_amplitude > 0.05:  # 5% amplitude
                prob = min(1.0, max_amplitude / 0.1)
                return prob, [f"Locked mode detected: amplitude = {max_amplitude:.3f}"]
        
        return 0.0, []
    
    def _density_limit_indicator(self, plasma_state: Dict[str, Any], 
                                mhd_modes: List[Any] = None) -> Tuple[float, List[str]:
        """Density limit disruption indicator."""
        density = plasma_state.get('density_avg', 1e19)
        plasma_current = plasma_state.get('plasma_current', 1.0)
        minor_radius = plasma_state.get('minor_radius', 0.5)
        
        # Greenwald limit
        greenwald_limit = plasma_current / (math.pi * minor_radius**2) * 1e20
        
        if density > greenwald_limit * 0.8:
            prob = min(1.0, (density - greenwald_limit * 0.8) / (greenwald_limit * 0.2))
            return prob, [f"Greenwald limit approach: n/nG = {density/greenwald_limit:.2f}"]
        
        return 0.0, []
    
    def _impurity_indicator(self, plasma_state: Dict[str, Any], 
                           mhd_modes: List[Any] = None) -> Tuple[float, List[str]]:
        """Impurity accumulation indicator."""
        z_eff = plasma_state.get('z_effective', 1.5)
        radiated_power_fraction = plasma_state.get('rad_power_fraction', 0.3)
        
        factors = []
        prob = 0.0
        
        if z_eff > 3.0:
            prob = max(prob, min(1.0, (z_eff - 3.0) / 2.0))
            factors.append(f"High Z_eff: {z_eff:.2f}")
        
        if radiated_power_fraction > 0.8:
            prob = max(prob, min(1.0, (radiated_power_fraction - 0.8) / 0.2))
            factors.append(f"High radiation: {radiated_power_fraction:.2f}")
        
        return prob, factors
    
    def _mhd_activity_indicator(self, plasma_state: Dict[str, Any], 
                               mhd_modes: List[Any] = None) -> Tuple[float, List[str]]:
        """MHD activity disruption indicator."""
        if not mhd_modes:
            return 0.0, []
        
        # Count growing modes
        growing_modes = [mode for mode in mhd_modes if mode.growth_rate > 100]
        total_amplitude = sum(mode.amplitude for mode in mhd_modes)
        
        factors = []
        prob = 0.0
        
        if len(growing_modes) > 2:
            prob = max(prob, min(1.0, len(growing_modes) / 5.0))
            factors.append(f"Multiple growing modes: {len(growing_modes)}")
        
        if total_amplitude > 0.1:
            prob = max(prob, min(1.0, total_amplitude / 0.2))
            factors.append(f"High MHD amplitude: {total_amplitude:.3f}")
        
        return prob, factors
    
    def _error_field_indicator(self, plasma_state: Dict[str, Any], 
                              mhd_modes: List[Any] = None) -> Tuple[float, List[str]]:
        """Error field disruption indicator."""
        error_field = plasma_state.get('error_field_amplitude', 0.0)
        
        if error_field > 1e-4:  # Tesla
            prob = min(1.0, error_field / 5e-4)
            return prob, [f"Error field: {error_field:.2e} T"]
        
        return 0.0, []
    
    def _profile_consistency_indicator(self, plasma_state: Dict[str, Any], 
                                     mhd_modes: List[Any] = None) -> Tuple[float, List[str]]:
        """Profile consistency indicator."""
        if 'q_profile' not in plasma_state:
            return 0.0, []
        
        q_profile = plasma_state['q_profile']
        
        # Check for q=1 surface near center (sawtooth instability)
        if len(q_profile) > 2 and q_profile[1] < 1.0:
            return 0.1, ["q < 1 near center"]
        
        # Check for steep gradients
        gradients = [abs(q_profile[i+1] - q_profile[i]) for i in range(len(q_profile)-1)]
        max_gradient = max(gradients) if gradients else 0.0
        
        if max_gradient > 5.0:
            prob = min(1.0, max_gradient / 10.0)
            return prob, [f"Steep q gradient: {max_gradient:.2f}"]
        
        return 0.0, []
    
    def _ml_predict(self, features: List[float]) -> float:
        """Machine learning disruption prediction."""
        # Simple feedforward network
        if len(features) < len(self.ml_model_weights) // 2:
            features.extend([0.0] * (len(self.ml_model_weights) // 2 - len(features)))
        
        # Hidden layer
        hidden = []
        for i in range(10):
            activation = sum(f * self.ml_model_weights[i * len(features[:10]) + j] 
                           for j, f in enumerate(features[:10]))
            hidden.append(max(0, activation))  # ReLU
        
        # Output
        output = sum(h * self.ml_model_weights[40 + i] for i, h in enumerate(hidden))
        return 1.0 / (1.0 + math.exp(-output))  # Sigmoid
    
    def _estimate_time_to_disruption(self, probability: float, features: List[float]) -> Optional[float]:
        """Estimate time to disruption based on growth rates."""
        if probability < 0.5:
            return None
        
        # Simple model based on probability and feature trends
        if len(self.feature_history) >= 2:
            # Estimate growth rate from recent history
            recent_change = sum(abs(features[i] - self.feature_history[-1][i]) 
                              for i in range(min(len(features), len(self.feature_history[-1]))))
            
            if recent_change > 0.1:
                # Fast evolution - short time
                return 0.01 + 0.1 / probability  # 10ms to 100ms
            else:
                # Slow evolution - longer time
                return 0.1 + 1.0 / probability   # 100ms to 1s
        
        return 0.05  # Default 50ms
    
    def _calculate_confidence(self, physics_predictions: Dict[str, float], 
                             ml_prediction: float) -> float:
        """Calculate prediction confidence."""
        # Agreement between physics and ML
        physics_avg = sum(physics_predictions.values()) / len(physics_predictions)
        agreement = 1.0 - abs(physics_avg - ml_prediction)
        
        # Consistency of physics indicators
        physics_std = math.sqrt(sum((p - physics_avg)**2 for p in physics_predictions.values()) / 
                               len(physics_predictions))
        consistency = max(0.0, 1.0 - physics_std)
        
        return 0.6 * agreement + 0.4 * consistency
    
    def _generate_mitigation_recommendations(self, probability: float, 
                                           factors: List[str]) -> List[str]:
        """Generate mitigation recommendations."""
        recommendations = []
        
        if probability > 0.8:
            recommendations.append("URGENT: Initiate emergency shutdown sequence")
        elif probability > 0.6:
            recommendations.append("Prepare for controlled discharge termination")
        elif probability > 0.4:
            recommendations.append("Activate disruption mitigation systems")
        
        # Specific recommendations based on factors
        for factor in factors:
            if "Beta limit" in factor:
                recommendations.append("Reduce auxiliary heating power")
                recommendations.append("Increase plasma current if possible")
            elif "Locked mode" in factor:
                recommendations.append("Apply ECCD at mode location")
                recommendations.append("Activate error field correction")
            elif "Greenwald" in factor:
                recommendations.append("Reduce gas puffing")
                recommendations.append("Increase plasma current if possible")
            elif "radiation" in factor:
                recommendations.append("Reduce impurity sources")
                recommendations.append("Apply central heating")
        
        return list(set(recommendations))  # Remove duplicates


class SafetyInterlock:
    """
    Hardware safety interlock system for emergency protection.
    """
    
    def __init__(self):
        self.safety_limits = self._initialize_safety_limits()
        self.interlock_status = {}
        self.trip_history = deque(maxlen=1000)
        self.emergency_callbacks = []
        
        # Initialize all interlocks as operational
        for limit in self.safety_limits:
            self.interlock_status[limit.parameter_name] = SafetyStatus.OPERATIONAL
    
    def _initialize_safety_limits(self) -> List[SafetyLimit]:
        """Initialize safety limits for all critical parameters."""
        return [
            SafetyLimit(
                parameter_name="plasma_current",
                normal_max=15.0,
                warning_threshold=18.0,
                alarm_threshold=19.5,
                trip_threshold=20.0,
                unit="MA",
                description="Plasma current limit"
            ),
            SafetyLimit(
                parameter_name="beta_n",
                normal_max=2.5,
                warning_threshold=3.0,
                alarm_threshold=3.5,
                trip_threshold=4.0,
                unit="%",
                description="Normalized beta limit"
            ),
            SafetyLimit(
                parameter_name="density_peak",
                normal_max=8e19,
                warning_threshold=1e20,
                alarm_threshold=1.2e20,
                trip_threshold=1.5e20,
                unit="m^-3",
                description="Peak electron density limit"
            ),
            SafetyLimit(
                parameter_name="stored_energy",
                normal_max=400,
                warning_threshold=500,
                alarm_threshold=600,
                trip_threshold=700,
                unit="MJ",
                description="Plasma stored energy limit"
            ),
            SafetyLimit(
                parameter_name="power_load",
                normal_max=15,
                warning_threshold=18,
                alarm_threshold=22,
                trip_threshold=25,
                unit="MW/m^2",
                description="Divertor power load limit"
            ),
            SafetyLimit(
                parameter_name="vessel_pressure",
                normal_max=1e-4,
                warning_threshold=5e-4,
                alarm_threshold=1e-3,
                trip_threshold=2e-3,
                unit="Pa",
                description="Vacuum vessel pressure limit"
            )
        ]
    
    def check_safety_limits(self, parameters: Dict[str, float]) -> Tuple[SafetyLevel, List[str]]:
        """Check all safety limits and return overall safety level."""
        max_safety_level = SafetyLevel.LEVEL_0
        violations = []
        
        for limit in self.safety_limits:
            if limit.parameter_name in parameters:
                value = parameters[limit.parameter_name]
                level, message = limit.evaluate(value)
                
                if level.value > max_safety_level.value:
                    max_safety_level = level
                
                if level != SafetyLevel.LEVEL_0:
                    violations.append(f"{limit.parameter_name}: {message}")
                    
                    # Log the violation
                    safety_logger.warning(f"Safety limit violation: {limit.parameter_name} = {value} {limit.unit}, {message}")
                    
                    # Record trip if at trip level
                    if level in [SafetyLevel.LEVEL_3, SafetyLevel.LEVEL_4]:
                        self._record_trip(limit.parameter_name, value, level)
        
        return max_safety_level, violations
    
    def _record_trip(self, parameter: str, value: float, level: SafetyLevel):
        """Record safety trip event."""
        trip_event = {
            'timestamp': time.time(),
            'parameter': parameter,
            'value': value,
            'level': level,
            'action_required': level == SafetyLevel.LEVEL_4
        }
        
        self.trip_history.append(trip_event)
        safety_logger.critical(f"SAFETY TRIP: {parameter} = {value}, Level: {level.value}")
        
        # Trigger emergency callbacks for level 4
        if level == SafetyLevel.LEVEL_4:
            for callback in self.emergency_callbacks:
                try:
                    callback(trip_event)
                except Exception as e:
                    safety_logger.error(f"Emergency callback failed: {e}")
    
    def register_emergency_callback(self, callback: Callable[[Dict], None]):
        """Register emergency callback for critical trips."""
        self.emergency_callbacks.append(callback)
    
    def get_trip_statistics(self) -> Dict[str, Any]:
        """Get statistics on safety trips."""
        if not self.trip_history:
            return {'total_trips': 0}
        
        total_trips = len(self.trip_history)
        critical_trips = sum(1 for trip in self.trip_history if trip['level'] == SafetyLevel.LEVEL_4)
        
        # Most common trip parameters
        parameter_counts = {}
        for trip in self.trip_history:
            param = trip['parameter']
            parameter_counts[param] = parameter_counts.get(param, 0) + 1
        
        return {
            'total_trips': total_trips,
            'critical_trips': critical_trips,
            'trips_per_day': total_trips / max(1, (time.time() - self.trip_history[0]['timestamp']) / 86400),
            'most_common_parameter': max(parameter_counts, key=parameter_counts.get) if parameter_counts else None,
            'parameter_counts': parameter_counts
        }


class ComponentHealthMonitor:
    """
    Predictive maintenance and component health monitoring system.
    """
    
    def __init__(self):
        self.components = self._initialize_components()
        self.health_history = {comp: deque(maxlen=1000) for comp in self.components}
        self.maintenance_schedule = {}
        
    def _initialize_components(self) -> List[str]:
        """Initialize list of monitored components."""
        return [
            'poloidal_field_coils',
            'toroidal_field_coils',
            'neutral_beam_injectors',
            'electron_cyclotron_heating',
            'cryogenic_system',
            'vacuum_pumping_system',
            'diagnostics_systems',
            'power_supplies',
            'cooling_system',
            'tritium_handling'
        ]
    
    def assess_component_health(self, component_data: Dict[str, Dict[str, float]]) -> Dict[str, ComponentHealth]:
        """Assess health of all components."""
        health_assessments = {}
        
        for component in self.components:
            if component in component_data:
                health = self._assess_single_component(component, component_data[component])
                health_assessments[component] = health
                self.health_history[component].append(health)
        
        return health_assessments
    
    def _assess_single_component(self, component: str, data: Dict[str, float]) -> ComponentHealth:
        """Assess health of a single component."""
        # Basic health scoring
        health_score = 100.0
        critical_issues = []
        recommendations = []
        
        # Component-specific health assessment
        if component == 'poloidal_field_coils':
            health_score, issues, recs = self._assess_pf_coils(data)
            critical_issues.extend(issues)
            recommendations.extend(recs)
        elif component == 'toroidal_field_coils':
            health_score, issues, recs = self._assess_tf_coils(data)
            critical_issues.extend(issues)
            recommendations.extend(recs)
        elif component == 'cryogenic_system':
            health_score, issues, recs = self._assess_cryogenic(data)
            critical_issues.extend(issues)
            recommendations.extend(recs)
        else:
            # Generic assessment
            health_score, issues, recs = self._generic_health_assessment(data)
            critical_issues.extend(issues)
            recommendations.extend(recs)
        
        # Calculate remaining lifetime
        remaining_lifetime = self._estimate_remaining_lifetime(component, health_score, data)
        
        # Calculate failure probability
        failure_probability = (100 - health_score) / 100.0
        
        # Determine maintenance requirements
        maintenance_required = health_score < 80 or len(critical_issues) > 0
        
        return ComponentHealth(
            component_name=component,
            health_score=health_score,
            remaining_lifetime=remaining_lifetime,
            failure_probability=failure_probability,
            maintenance_required=maintenance_required,
            critical_issues=critical_issues,
            recommendations=recommendations
        )
    
    def _assess_pf_coils(self, data: Dict[str, float]) -> Tuple[float, List[str], List[str]]:
        """Assess poloidal field coils health."""
        health_score = 100.0
        issues = []
        recommendations = []
        
        # Check current density
        if 'current_density' in data and data['current_density'] > 10e6:  # A/m^2
            health_score -= 20
            issues.append("High current density stress")
            recommendations.append("Consider current redistribution")
        
        # Check temperature
        if 'temperature' in data and data['temperature'] > 80:  # Kelvin
            health_score -= 15
            issues.append("Elevated coil temperature")
            recommendations.append("Check cryogenic cooling system")
        
        # Check insulation resistance
        if 'insulation_resistance' in data and data['insulation_resistance'] < 1e6:  # Ohms
            health_score -= 30
            issues.append("Insulation degradation detected")
            recommendations.append("Schedule insulation inspection")
        
        return max(0, health_score), issues, recommendations
    
    def _assess_tf_coils(self, data: Dict[str, float]) -> Tuple[float, List[str], List[str]]:
        """Assess toroidal field coils health."""
        health_score = 100.0
        issues = []
        recommendations = []
        
        # Check magnetic field uniformity
        if 'field_error' in data and data['field_error'] > 1e-4:
            health_score -= 25
            issues.append("Magnetic field error detected")
            recommendations.append("Perform field mapping and correction")
        
        # Check superconductor temperature
        if 'sc_temperature' in data and data['sc_temperature'] > 4.5:  # Kelvin
            health_score -= 20
            issues.append("Superconductor temperature high")
            recommendations.append("Check helium cooling system")
        
        # Check quench history
        if 'quench_count' in data and data['quench_count'] > 5:
            health_score -= 15
            issues.append("Multiple quenches detected")
            recommendations.append("Investigate quench causes")
        
        return max(0, health_score), issues, recommendations
    
    def _assess_cryogenic(self, data: Dict[str, float]) -> Tuple[float, List[str], List[str]]:
        """Assess cryogenic system health."""
        health_score = 100.0
        issues = []
        recommendations = []
        
        # Check helium temperature
        if 'helium_temp' in data and data['helium_temp'] > 4.5:
            health_score -= 30
            issues.append("Helium temperature too high")
            recommendations.append("Check refrigerator capacity")
        
        # Check system efficiency
        if 'efficiency' in data and data['efficiency'] < 0.3:
            health_score -= 20
            issues.append("Low system efficiency")
            recommendations.append("Schedule maintenance inspection")
        
        # Check pressure stability
        if 'pressure_variation' in data and data['pressure_variation'] > 0.1:
            health_score -= 15
            issues.append("Pressure instability detected")
            recommendations.append("Check pressure control system")
        
        return max(0, health_score), issues, recommendations
    
    def _generic_health_assessment(self, data: Dict[str, float]) -> Tuple[float, List[str], List[str]]:
        """Generic health assessment for components."""
        health_score = 100.0
        issues = []
        recommendations = []
        
        # Check for any anomalous values (simple heuristic)
        for param, value in data.items():
            if abs(value) > 1000:  # Arbitrarily large value
                health_score -= 10
                issues.append(f"Anomalous {param} reading")
                recommendations.append(f"Calibrate {param} sensor")
        
        return max(0, health_score), issues, recommendations
    
    def _estimate_remaining_lifetime(self, component: str, health_score: float, 
                                   data: Dict[str, float]) -> Optional[float]:
        """Estimate remaining component lifetime in hours."""
        if health_score > 90:
            return 10000.0  # ~1 year
        elif health_score > 70:
            return 5000.0   # ~6 months
        elif health_score > 50:
            return 1000.0   # ~1 month
        elif health_score > 20:
            return 100.0    # ~1 week
        else:
            return 10.0     # Critical - immediate attention
    
    def generate_maintenance_schedule(self, health_assessments: Dict[str, ComponentHealth]) -> Dict[str, Dict[str, Any]]:
        """Generate predictive maintenance schedule."""
        schedule = {}
        
        for component, health in health_assessments.items():
            if health.maintenance_required:
                urgency = "CRITICAL" if health.health_score < 50 else "HIGH" if health.health_score < 70 else "MEDIUM"
                
                schedule[component] = {
                    'urgency': urgency,
                    'estimated_time': health.remaining_lifetime,
                    'maintenance_type': 'CORRECTIVE' if health.critical_issues else 'PREVENTIVE',
                    'critical_issues': health.critical_issues,
                    'recommendations': health.recommendations
                }
        
        return schedule


def create_comprehensive_safety_system() -> Dict[str, Any]:
    """Create comprehensive safety system."""
    
    disruption_predictor = DisruptionPredictor()
    safety_interlock = SafetyInterlock()
    health_monitor = ComponentHealthMonitor()
    
    # Register emergency callback
    def emergency_shutdown_callback(trip_event: Dict):
        safety_logger.critical(f"EMERGENCY SHUTDOWN INITIATED: {trip_event}")
        # Implementation would trigger actual shutdown hardware
    
    safety_interlock.register_emergency_callback(emergency_shutdown_callback)
    
    def comprehensive_safety_check(plasma_state: Dict[str, Any], 
                                  mhd_modes: List[Any] = None,
                                  component_data: Dict[str, Dict[str, float]] = None) -> Dict[str, Any]:
        """Perform comprehensive safety assessment."""
        
        # Disruption prediction
        disruption_pred = disruption_predictor.predict_disruption(plasma_state, mhd_modes)
        
        # Safety limit checking
        safety_params = {
            'plasma_current': plasma_state.get('plasma_current', 1.0),
            'beta_n': plasma_state.get('beta_n', 0.02),
            'density_peak': max(plasma_state.get('density', [1e19])) if hasattr(plasma_state.get('density', 1e19), '__iter__') else plasma_state.get('density', 1e19),
            'stored_energy': plasma_state.get('stored_energy', 100),
            'power_load': plasma_state.get('divertor_power', 10),
            'vessel_pressure': plasma_state.get('pressure', 1e-6)
        }
        
        safety_level, violations = safety_interlock.check_safety_limits(safety_params)
        
        # Component health assessment
        health_assessments = {}
        maintenance_schedule = {}
        if component_data:
            health_assessments = health_monitor.assess_component_health(component_data)
            maintenance_schedule = health_monitor.generate_maintenance_schedule(health_assessments)
        
        # Overall safety status
        if safety_level == SafetyLevel.LEVEL_4 or disruption_pred.probability > 0.9:
            overall_status = "EMERGENCY"
        elif safety_level == SafetyLevel.LEVEL_3 or disruption_pred.probability > 0.7:
            overall_status = "CRITICAL"
        elif safety_level == SafetyLevel.LEVEL_2 or disruption_pred.probability > 0.5:
            overall_status = "WARNING"
        elif safety_level == SafetyLevel.LEVEL_1 or disruption_pred.probability > 0.3:
            overall_status = "CAUTION"
        else:
            overall_status = "NORMAL"
        
        return {
            'overall_status': overall_status,
            'safety_level': safety_level,
            'disruption_prediction': disruption_pred,
            'safety_violations': violations,
            'health_assessments': health_assessments,
            'maintenance_schedule': maintenance_schedule,
            'recommendations': _generate_safety_recommendations(overall_status, disruption_pred, violations, maintenance_schedule)
        }
    
    def _generate_safety_recommendations(status: str, disruption_pred: DisruptionPrediction,
                                       violations: List[str], maintenance: Dict) -> List[str]:
        """Generate safety recommendations."""
        recommendations = []
        
        if status == "EMERGENCY":
            recommendations.append("IMMEDIATE ACTION REQUIRED: Initiate emergency shutdown")
        elif status == "CRITICAL":
            recommendations.append("URGENT: Prepare for controlled shutdown")
        
        # Add disruption mitigation recommendations
        recommendations.extend(disruption_pred.mitigation_recommendations)
        
        # Add maintenance recommendations
        for component, schedule in maintenance.items():
            if schedule['urgency'] == 'CRITICAL':
                recommendations.append(f"CRITICAL MAINTENANCE: {component}")
        
        return recommendations
    
    return {
        'disruption_predictor': disruption_predictor,
        'safety_interlock': safety_interlock,
        'health_monitor': health_monitor,
        'comprehensive_safety_check': comprehensive_safety_check,
        'system_type': 'comprehensive_safety'
    }


if __name__ == "__main__":
    # Demonstration of comprehensive safety system
    print("Comprehensive Safety System for Tokamak Control")
    print("=" * 55)
    
    safety_system = create_comprehensive_safety_system()
    
    # Test scenario
    test_plasma_state = {
        'plasma_current': 2.0,
        'beta_n': 0.025,
        'density': [2e19, 1.8e19, 1.5e19, 1.2e19, 1e19] * 2,
        'temperature': [15, 12, 10, 8, 6] * 2,
        'q_profile': [1.2, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5],
        'stored_energy': 150,
        'divertor_power': 12,
        'pressure': 5e-5,
        'z_effective': 2.0,
        'rad_power_fraction': 0.4
    }
    
    test_component_data = {
        'poloidal_field_coils': {'current_density': 8e6, 'temperature': 75, 'insulation_resistance': 2e6},
        'cryogenic_system': {'helium_temp': 4.2, 'efficiency': 0.35, 'pressure_variation': 0.05}
    }
    
    # Run comprehensive safety check
    safety_results = safety_system['comprehensive_safety_check'](
        test_plasma_state, component_data=test_component_data
    )
    
    print(f"\n🛡️  Safety Assessment Results:")
    print(f"  Overall Status: {safety_results['overall_status']}")
    print(f"  Safety Level: {safety_results['safety_level'].value}")
    print(f"  Disruption Probability: {safety_results['disruption_prediction'].probability:.3f}")
    print(f"  Disruption Confidence: {safety_results['disruption_prediction'].confidence:.3f}")
    
    if safety_results['safety_violations']:
        print(f"\n⚠️  Safety Violations:")
        for violation in safety_results['safety_violations']:
            print(f"    - {violation}")
    
    if safety_results['health_assessments']:
        print(f"\n🔧 Component Health:")
        for component, health in safety_results['health_assessments'].items():
            print(f"  {component}: {health.health_score:.1f}/100")
            if health.critical_issues:
                print(f"    Issues: {', '.join(health.critical_issues)}")
    
    if safety_results['recommendations']:
        print(f"\n💡 Recommendations:")
        for rec in safety_results['recommendations'][:5]:  # Show first 5
            print(f"    - {rec}")
    
    print("\n✅ Comprehensive Safety System Ready!")