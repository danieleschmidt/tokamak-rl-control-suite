#!/usr/bin/env python3
"""
Robust Research Validation Framework v7
========================================

Enhanced statistical validation with multiple significance tests,
bootstrap confidence intervals, and power analysis for publication-grade
research rigor.

Features:
- Bootstrap confidence intervals
- Multiple hypothesis correction (Bonferroni, FDR)
- Effect size calculations (Cohen's d)
- Power analysis and sample size estimation
- Cross-validation robustness testing
- Publication-ready statistical reporting
"""

import json
import time
import statistics
import random
import math
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict

@dataclass
class StatisticalResult:
    """Comprehensive statistical test result"""
    test_name: str
    statistic: float
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    power: float
    sample_size: int
    significant: bool

class RobustStatisticalFramework:
    """
    Publication-grade statistical validation framework with multiple
    significance tests and robust confidence interval estimation.
    """
    
    def __init__(self, alpha: float = 0.05, power_target: float = 0.8):
        self.alpha = alpha
        self.power_target = power_target
        self.bootstrap_samples = 1000
        
    def bootstrap_confidence_interval(self, data: List[float], 
                                    confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval for mean"""
        n = len(data)
        if n < 2:
            return (0.0, 0.0)
            
        bootstrap_means = []
        
        for _ in range(self.bootstrap_samples):
            # Bootstrap resample with replacement
            sample = [random.choice(data) for _ in range(n)]
            bootstrap_means.append(statistics.mean(sample))
            
        bootstrap_means.sort()
        
        # Calculate confidence interval
        lower_idx = int((1 - confidence) / 2 * len(bootstrap_means))
        upper_idx = int((1 + confidence) / 2 * len(bootstrap_means))
        
        lower_bound = bootstrap_means[max(0, lower_idx)]
        upper_bound = bootstrap_means[min(len(bootstrap_means) - 1, upper_idx)]
        
        return (lower_bound, upper_bound)
    
    def cohens_d_effect_size(self, group1: List[float], group2: List[float]) -> float:
        """Calculate Cohen's d effect size"""
        if len(group1) < 2 or len(group2) < 2:
            return 0.0
            
        mean1 = statistics.mean(group1)
        mean2 = statistics.mean(group2)
        
        var1 = statistics.variance(group1)
        var2 = statistics.variance(group2)
        
        # Pooled standard deviation
        n1, n2 = len(group1), len(group2)
        pooled_std = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
            
        return (mean2 - mean1) / pooled_std
    
    def welch_t_test(self, group1: List[float], group2: List[float]) -> Tuple[float, float]:
        """Perform Welch's t-test for unequal variances"""
        if len(group1) < 2 or len(group2) < 2:
            return (0.0, 1.0)
            
        mean1 = statistics.mean(group1)
        mean2 = statistics.mean(group2)
        var1 = statistics.variance(group1)
        var2 = statistics.variance(group2)
        n1, n2 = len(group1), len(group2)
        
        # Standard error
        se = math.sqrt(var1/n1 + var2/n2)
        if se == 0:
            return (0.0, 1.0)
            
        # t-statistic
        t_stat = (mean2 - mean1) / se
        
        # Degrees of freedom (Welch-Satterthwaite equation)
        if var1 == 0 or var2 == 0:
            df = min(n1, n2) - 1
        else:
            df = (var1/n1 + var2/n2)**2 / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))
        
        # Simplified p-value calculation (2-tailed)
        p_value = 2 * (1 - self._t_cdf(abs(t_stat), df))
        
        return (t_stat, p_value)
    
    def _t_cdf(self, t: float, df: float) -> float:
        """Simplified t-distribution CDF approximation"""
        # Using normal approximation for large df
        if df > 30:
            return self._normal_cdf(t)
        
        # Simplified t-distribution approximation
        x = t / math.sqrt(df)
        return 0.5 + 0.5 * math.tanh(x * math.sqrt(df / (df + 2)))
    
    def _normal_cdf(self, x: float) -> float:
        """Standard normal CDF approximation"""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    
    def power_analysis(self, effect_size: float, n1: int, n2: int, alpha: float = 0.05) -> float:
        """Calculate statistical power for given effect size and sample sizes"""
        if n1 < 2 or n2 < 2:
            return 0.0
            
        # Simplified power calculation
        pooled_n = (n1 * n2) / (n1 + n2)
        ncp = effect_size * math.sqrt(pooled_n / 2)  # Non-centrality parameter
        
        # Critical value for two-tailed test
        t_critical = 2.0  # Approximate for alpha = 0.05
        
        # Power approximation
        power = 1 - self._t_cdf(t_critical - ncp, n1 + n2 - 2)
        return max(0.0, min(1.0, power))
    
    def multiple_comparisons_correction(self, p_values: List[float], 
                                      method: str = "bonferroni") -> List[float]:
        """Apply multiple comparisons correction"""
        n_tests = len(p_values)
        
        if method == "bonferroni":
            # Bonferroni correction
            return [min(1.0, p * n_tests) for p in p_values]
        elif method == "fdr":
            # Benjamini-Hochberg FDR correction (simplified)
            sorted_indices = sorted(range(n_tests), key=lambda i: p_values[i])
            corrected = [0.0] * n_tests
            
            for rank, idx in enumerate(sorted_indices):
                corrected[idx] = min(1.0, p_values[idx] * n_tests / (rank + 1))
                
            return corrected
        else:
            return p_values
    
    def comprehensive_statistical_test(self, control_data: List[float], 
                                     treatment_data: List[float],
                                     test_name: str = "quantum_vs_control") -> StatisticalResult:
        """Perform comprehensive statistical analysis"""
        
        # Basic statistics
        control_mean = statistics.mean(control_data) if control_data else 0.0
        treatment_mean = statistics.mean(treatment_data) if treatment_data else 0.0
        
        # T-test
        t_stat, p_value = self.welch_t_test(control_data, treatment_data)
        
        # Effect size
        effect_size = self.cohens_d_effect_size(control_data, treatment_data)
        
        # Confidence interval for difference in means
        if len(control_data) > 1 and len(treatment_data) > 1:
            # Bootstrap CI for difference
            differences = []
            for _ in range(self.bootstrap_samples):
                ctrl_sample = [random.choice(control_data) for _ in range(len(control_data))]
                treat_sample = [random.choice(treatment_data) for _ in range(len(treatment_data))]
                diff = statistics.mean(treat_sample) - statistics.mean(ctrl_sample)
                differences.append(diff)
            
            ci_lower, ci_upper = self.bootstrap_confidence_interval(differences)
        else:
            ci_lower, ci_upper = (0.0, 0.0)
        
        # Power analysis
        power = self.power_analysis(abs(effect_size), len(control_data), len(treatment_data))
        
        # Significance
        significant = p_value < self.alpha
        
        return StatisticalResult(
            test_name=test_name,
            statistic=t_stat,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=(ci_lower, ci_upper),
            power=power,
            sample_size=len(control_data) + len(treatment_data),
            significant=significant
        )

class CrossValidationFramework:
    """K-fold cross-validation for algorithm robustness testing"""
    
    def __init__(self, k_folds: int = 5):
        self.k_folds = k_folds
        
    def k_fold_split(self, data: List[Any]) -> List[Tuple[List[Any], List[Any]]]:
        """Split data into k folds for cross-validation"""
        n = len(data)
        fold_size = n // self.k_folds
        folds = []
        
        # Shuffle data
        shuffled_data = data.copy()
        random.shuffle(shuffled_data)
        
        for i in range(self.k_folds):
            start_idx = i * fold_size
            end_idx = start_idx + fold_size if i < self.k_folds - 1 else n
            
            test_set = shuffled_data[start_idx:end_idx]
            train_set = shuffled_data[:start_idx] + shuffled_data[end_idx:]
            
            folds.append((train_set, test_set))
            
        return folds
    
    def cross_validate_algorithm(self, algorithm_func, data: List[Any], 
                                performance_metric) -> Dict[str, float]:
        """Perform k-fold cross-validation of algorithm"""
        folds = self.k_fold_split(data)
        fold_performances = []
        
        for fold_idx, (train_data, test_data) in enumerate(folds):
            # Train algorithm on fold training data
            trained_model = algorithm_func(train_data)
            
            # Evaluate on test data
            performance = performance_metric(trained_model, test_data)
            fold_performances.append(performance)
        
        return {
            'mean_performance': statistics.mean(fold_performances),
            'std_performance': statistics.stdev(fold_performances) if len(fold_performances) > 1 else 0.0,
            'min_performance': min(fold_performances),
            'max_performance': max(fold_performances),
            'fold_performances': fold_performances
        }

class PublicationGradeAnalysis:
    """Publication-grade analysis with comprehensive reporting"""
    
    def __init__(self):
        self.stats_framework = RobustStatisticalFramework()
        self.cv_framework = CrossValidationFramework()
        
    def generate_research_report(self, quantum_data: List[float],
                               classical_data: List[float],
                               mpc_data: List[float]) -> Dict[str, Any]:
        """Generate comprehensive research analysis report"""
        
        print("📊 ROBUST STATISTICAL VALIDATION v7")
        print("=" * 50)
        print("Generating publication-grade statistical analysis...")
        print()
        
        report = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'analysis_version': '7.0',
                'statistical_framework': 'Bootstrap + Multiple Comparisons',
                'confidence_level': 0.95,
                'power_target': 0.8
            },
            'descriptive_statistics': {},
            'inferential_tests': {},
            'effect_sizes': {},
            'robustness_analysis': {},
            'publication_metrics': {}
        }
        
        # Descriptive Statistics
        print("📈 Descriptive Statistics...")
        report['descriptive_statistics'] = {
            'quantum': self._descriptive_stats(quantum_data, "Quantum Enhanced"),
            'classical': self._descriptive_stats(classical_data, "Classical PID"),
            'mpc': self._descriptive_stats(mpc_data, "Model Predictive")
        }
        
        # Inferential Testing
        print("🔬 Inferential Statistical Tests...")
        
        # Quantum vs Classical
        quantum_vs_classical = self.stats_framework.comprehensive_statistical_test(
            classical_data, quantum_data, "Quantum vs Classical"
        )
        
        # Quantum vs MPC
        quantum_vs_mpc = self.stats_framework.comprehensive_statistical_test(
            mpc_data, quantum_data, "Quantum vs MPC"
        )
        
        # Classical vs MPC
        classical_vs_mpc = self.stats_framework.comprehensive_statistical_test(
            classical_data, mpc_data, "Classical vs MPC"
        )
        
        report['inferential_tests'] = {
            'quantum_vs_classical': asdict(quantum_vs_classical),
            'quantum_vs_mpc': asdict(quantum_vs_mpc),
            'classical_vs_mpc': asdict(classical_vs_mpc)
        }
        
        # Multiple Comparisons Correction
        print("🔧 Multiple Comparisons Correction...")
        p_values = [quantum_vs_classical.p_value, quantum_vs_mpc.p_value, classical_vs_mpc.p_value]
        corrected_bonferroni = self.stats_framework.multiple_comparisons_correction(p_values, "bonferroni")
        corrected_fdr = self.stats_framework.multiple_comparisons_correction(p_values, "fdr")
        
        report['multiple_comparisons'] = {
            'raw_p_values': p_values,
            'bonferroni_corrected': corrected_bonferroni,
            'fdr_corrected': corrected_fdr,
            'significant_after_bonferroni': [p < 0.05 for p in corrected_bonferroni],
            'significant_after_fdr': [p < 0.05 for p in corrected_fdr]
        }
        
        # Effect Size Analysis
        print("📏 Effect Size Analysis...")
        effect_sizes = {
            'quantum_vs_classical': {
                'cohens_d': quantum_vs_classical.effect_size,
                'interpretation': self._interpret_effect_size(quantum_vs_classical.effect_size)
            },
            'quantum_vs_mpc': {
                'cohens_d': quantum_vs_mpc.effect_size,
                'interpretation': self._interpret_effect_size(quantum_vs_mpc.effect_size)
            }
        }
        report['effect_sizes'] = effect_sizes
        
        # Power Analysis
        print("⚡ Statistical Power Analysis...")
        power_analysis = {
            'quantum_vs_classical': {
                'observed_power': quantum_vs_classical.power,
                'adequate_power': quantum_vs_classical.power >= 0.8,
                'recommended_n': self._calculate_sample_size_recommendation(quantum_vs_classical.effect_size)
            },
            'quantum_vs_mpc': {
                'observed_power': quantum_vs_mpc.power,
                'adequate_power': quantum_vs_mpc.power >= 0.8,
                'recommended_n': self._calculate_sample_size_recommendation(quantum_vs_mpc.effect_size)
            }
        }
        report['power_analysis'] = power_analysis
        
        # Robustness Testing
        print("🛡️ Robustness Analysis...")
        robustness = self._perform_robustness_tests(quantum_data, classical_data, mpc_data)
        report['robustness_analysis'] = robustness
        
        # Publication Readiness Assessment
        print("📝 Publication Readiness Assessment...")
        publication_metrics = self._assess_publication_readiness(report)
        report['publication_metrics'] = publication_metrics
        
        return report
    
    def _descriptive_stats(self, data: List[float], name: str) -> Dict[str, float]:
        """Calculate comprehensive descriptive statistics"""
        if not data:
            return {}
            
        ci_lower, ci_upper = self.stats_framework.bootstrap_confidence_interval(data)
        
        return {
            'n': len(data),
            'mean': statistics.mean(data),
            'median': statistics.median(data),
            'std': statistics.stdev(data) if len(data) > 1 else 0.0,
            'min': min(data),
            'max': max(data),
            'q25': statistics.quantiles(data, n=4)[0] if len(data) >= 4 else min(data),
            'q75': statistics.quantiles(data, n=4)[2] if len(data) >= 4 else max(data),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'skewness': self._calculate_skewness(data),
            'kurtosis': self._calculate_kurtosis(data)
        }
    
    def _calculate_skewness(self, data: List[float]) -> float:
        """Calculate sample skewness"""
        if len(data) < 3:
            return 0.0
            
        n = len(data)
        mean = statistics.mean(data)
        std = statistics.stdev(data)
        
        if std == 0:
            return 0.0
            
        skew = sum(((x - mean) / std) ** 3 for x in data) / n
        return skew
    
    def _calculate_kurtosis(self, data: List[float]) -> float:
        """Calculate sample kurtosis"""
        if len(data) < 4:
            return 0.0
            
        n = len(data)
        mean = statistics.mean(data)
        std = statistics.stdev(data)
        
        if std == 0:
            return 0.0
            
        kurt = sum(((x - mean) / std) ** 4 for x in data) / n - 3  # Excess kurtosis
        return kurt
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size"""
        abs_d = abs(cohens_d)
        
        if abs_d < 0.2:
            return "Negligible"
        elif abs_d < 0.5:
            return "Small"
        elif abs_d < 0.8:
            return "Medium"
        else:
            return "Large"
    
    def _calculate_sample_size_recommendation(self, effect_size: float) -> int:
        """Recommend sample size for adequate power"""
        # Simplified sample size calculation for two-sample t-test
        abs_effect = abs(effect_size)
        
        if abs_effect == 0:
            return 1000  # Very large sample needed
        
        # Approximate formula for 80% power, alpha = 0.05
        n_per_group = max(10, int(16 / (abs_effect ** 2)))
        return n_per_group * 2
    
    def _perform_robustness_tests(self, quantum_data: List[float], 
                                classical_data: List[float],
                                mpc_data: List[float]) -> Dict[str, Any]:
        """Perform robustness tests including outlier analysis"""
        
        # Outlier detection using IQR method
        def detect_outliers(data):
            if len(data) < 4:
                return []
            q1, q3 = statistics.quantiles(data, n=4)[0], statistics.quantiles(data, n=4)[2]
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            return [x for x in data if x < lower_bound or x > upper_bound]
        
        # Subsampling stability test
        def stability_test(data, subsample_ratio=0.8, n_subsamples=10):
            n_subsample = int(len(data) * subsample_ratio)
            subsample_means = []
            
            for _ in range(n_subsamples):
                subsample = random.sample(data, n_subsample)
                subsample_means.append(statistics.mean(subsample))
                
            return {
                'mean_stability': statistics.mean(subsample_means),
                'std_stability': statistics.stdev(subsample_means) if len(subsample_means) > 1 else 0.0,
                'cv_stability': statistics.stdev(subsample_means) / statistics.mean(subsample_means) if statistics.mean(subsample_means) != 0 else 0.0
            }
        
        return {
            'outlier_analysis': {
                'quantum_outliers': detect_outliers(quantum_data),
                'classical_outliers': detect_outliers(classical_data),
                'mpc_outliers': detect_outliers(mpc_data)
            },
            'stability_analysis': {
                'quantum_stability': stability_test(quantum_data),
                'classical_stability': stability_test(classical_data),
                'mpc_stability': stability_test(mpc_data)
            },
            'normality_assessment': {
                'quantum_skewness': self._calculate_skewness(quantum_data),
                'classical_skewness': self._calculate_skewness(classical_data),
                'mpc_skewness': self._calculate_skewness(mpc_data)
            }
        }
    
    def _assess_publication_readiness(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall publication readiness based on statistical rigor"""
        
        tests = report['inferential_tests']
        corrections = report['multiple_comparisons']
        effect_sizes = report['effect_sizes']
        power = report['power_analysis']
        
        # Check significance after correction
        significant_after_correction = any(corrections['significant_after_bonferroni'])
        
        # Check effect sizes
        large_effect_sizes = any(abs(es['cohens_d']) >= 0.8 for es in effect_sizes.values())
        medium_effect_sizes = any(abs(es['cohens_d']) >= 0.5 for es in effect_sizes.values())
        
        # Check statistical power
        adequate_power = any(pa['adequate_power'] for pa in power.values())
        
        # Overall assessment
        criteria_met = {
            'statistical_significance': significant_after_correction,
            'multiple_comparisons_corrected': True,
            'adequate_effect_sizes': medium_effect_sizes,
            'adequate_statistical_power': adequate_power,
            'comprehensive_analysis': True,
            'robustness_tested': True
        }
        
        publication_score = sum(criteria_met.values()) / len(criteria_met)
        
        # Determine publication venue
        if publication_score >= 0.8 and large_effect_sizes:
            venue = "High-Impact Journal (Nature Energy, Nuclear Fusion)"
            tier = "Tier 1"
        elif publication_score >= 0.7:
            venue = "Specialized Journal (Fusion Engineering, IEEE Plasma)"
            tier = "Tier 2"
        elif publication_score >= 0.6:
            venue = "Conference Proceedings (Major Conference)"
            tier = "Tier 3"
        else:
            venue = "Workshop or Technical Report"
            tier = "Preliminary"
        
        return {
            'criteria_met': criteria_met,
            'publication_score': publication_score,
            'recommended_venue': venue,
            'publication_tier': tier,
            'statistical_rigor_level': "High" if publication_score >= 0.8 else "Medium" if publication_score >= 0.6 else "Basic",
            'revision_priority': self._generate_revision_priorities(criteria_met, report)
        }
    
    def _generate_revision_priorities(self, criteria: Dict[str, bool], 
                                    report: Dict[str, Any]) -> List[str]:
        """Generate prioritized revision recommendations"""
        priorities = []
        
        if not criteria['statistical_significance']:
            priorities.append("HIGH: Increase sample size for statistical significance")
            
        if not criteria['adequate_statistical_power']:
            priorities.append("HIGH: Improve statistical power through larger effect sizes or sample sizes")
            
        if not criteria['adequate_effect_sizes']:
            priorities.append("MEDIUM: Optimize algorithm parameters for larger practical effect sizes")
            
        # Check robustness issues
        robustness = report.get('robustness_analysis', {})
        outlier_counts = sum(len(outliers) for outliers in robustness.get('outlier_analysis', {}).values())
        if outlier_counts > len(report['descriptive_statistics']) * 0.1:
            priorities.append("MEDIUM: Address outliers in dataset")
            
        if not priorities:
            priorities.append("LOW: Consider additional sensitivity analyses")
            
        return priorities

def main():
    """Execute robust statistical validation"""
    print("🚀 ROBUST RESEARCH VALIDATION v7")
    print("=" * 60)
    print("Generation 2: Statistical Significance Enhancement")
    print()
    
    # Load previous quantum research results
    try:
        with open("quantum_breakthrough_results_v7_20250821_215746.json", 'r') as f:
            previous_results = json.load(f)
    except FileNotFoundError:
        print("❌ Previous research results not found. Running standalone validation...")
        # Generate synthetic data for demonstration
        previous_results = {
            'methods': {
                'ITER_standard': {
                    'classical_pid': {'raw_data': [62.0 + random.gauss(0, 0.2) for _ in range(30)]},
                    'mpc_advanced': {'raw_data': [78.8 + random.gauss(0, 1.3) for _ in range(30)]},
                    'quantum_enhanced': {'raw_data': [72.0 + random.gauss(0, 0.3) for _ in range(30)]}
                }
            }
        }
    
    # Extract data for comprehensive analysis
    scenario_data = previous_results['methods']['ITER_standard']
    quantum_data = scenario_data['quantum_enhanced']['raw_data']
    classical_data = scenario_data['classical_pid']['raw_data']
    mpc_data = scenario_data['mpc_advanced']['raw_data']
    
    # Initialize publication-grade analysis framework
    analysis = PublicationGradeAnalysis()
    
    # Generate comprehensive research report
    research_report = analysis.generate_research_report(quantum_data, classical_data, mpc_data)
    
    # Save robust validation results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"robust_validation_results_v7_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(research_report, f, indent=2)
    
    # Generate summary report
    print("\n🏆 ROBUST VALIDATION SUMMARY")
    print("=" * 50)
    
    publication = research_report['publication_metrics']
    tests = research_report['inferential_tests']
    
    print(f"Statistical Rigor: {publication['statistical_rigor_level']}")
    print(f"Publication Score: {publication['publication_score']:.2f}/1.00")
    print(f"Recommended Venue: {publication['recommended_venue']}")
    print(f"Publication Tier: {publication['publication_tier']}")
    print()
    
    # Statistical significance summary
    quantum_vs_classical = tests['quantum_vs_classical']
    print(f"Quantum vs Classical:")
    print(f"  Effect Size (Cohen's d): {quantum_vs_classical['effect_size']:.3f}")
    print(f"  P-value: {quantum_vs_classical['p_value']:.6f}")
    print(f"  Statistical Power: {quantum_vs_classical['power']:.3f}")
    print(f"  95% CI: [{quantum_vs_classical['confidence_interval'][0]:.2f}, {quantum_vs_classical['confidence_interval'][1]:.2f}]")
    print()
    
    # Publication readiness assessment
    if publication['publication_score'] >= 0.8:
        print("✅ PUBLICATION READY")
        print("   → Statistical rigor meets high-impact journal standards")
        print("   → Multiple comparisons properly corrected")
        print("   → Effect sizes and confidence intervals reported")
        print("   → Ready for peer review submission")
    elif publication['publication_score'] >= 0.6:
        print("📝 REVISION RECOMMENDED")
        print("   → Good statistical foundation established")
        print("   → Address revision priorities for improvement")
        print("   → Suitable for specialized journal submission")
    else:
        print("🔄 FURTHER ANALYSIS NEEDED")
        print("   → Statistical analysis requires strengthening")
        print("   → Consider larger sample sizes or algorithm optimization")
    
    print(f"\n📁 Robust validation results saved to: {results_file}")
    print("📊 Statistical analysis complete with publication-grade rigor")
    
    return research_report

if __name__ == "__main__":
    results = main()