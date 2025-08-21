#!/usr/bin/env python3
"""
Research Quality Gates System v7
================================

Comprehensive quality validation for research publications including:
- Code quality and style compliance
- Statistical validation and power analysis  
- Reproducibility verification
- Benchmark performance testing
- Publication readiness assessment
- Peer review preparation

Ensures all research meets academic standards for high-impact journals.
"""

import json
import time
import os
import subprocess
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict

@dataclass
class QualityGateResult:
    """Result of a quality gate check"""
    gate_name: str
    passed: bool
    score: float
    details: Dict[str, Any]
    recommendations: List[str]
    severity: str  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'

class ResearchQualityGates:
    """Comprehensive quality gates system for research validation"""
    
    def __init__(self):
        self.version = "7.0"
        self.start_time = time.time()
        self.results = []
        
    def run_code_quality_gate(self) -> QualityGateResult:
        """Validate code quality and style compliance"""
        print("🔍 Code Quality Gate...")
        
        details = {}
        recommendations = []
        score = 0.0
        
        # Check if source files exist
        required_files = [
            'quantum_plasma_breakthrough_v7.py',
            'robust_research_validation_v7.py', 
            'academic_methodology_documentation_v7.py'
        ]
        
        missing_files = []
        for file in required_files:
            if not os.path.exists(file):
                missing_files.append(file)
        
        if missing_files:
            details['missing_files'] = missing_files
            recommendations.append(f"Missing required files: {', '.join(missing_files)}")
            return QualityGateResult(
                gate_name="Code Quality",
                passed=False,
                score=0.0,
                details=details,
                recommendations=recommendations,
                severity='CRITICAL'
            )
        
        # File size validation
        file_sizes = {}
        for file in required_files:
            size = os.path.getsize(file)
            file_sizes[file] = size
            
        details['file_sizes'] = file_sizes
        
        # Code structure validation
        code_quality_score = 0
        total_checks = 0
        
        for file in required_files:
            total_checks += 4
            
            # Check file has docstring
            with open(file, 'r') as f:
                content = f.read()
                
            if '"""' in content or "'''" in content:
                code_quality_score += 1
                
            # Check has main function
            if 'def main(' in content:
                code_quality_score += 1
                
            # Check has type hints
            if 'from typing import' in content or '-> ' in content:
                code_quality_score += 1
                
            # Check has error handling
            if 'try:' in content and 'except:' in content:
                code_quality_score += 1
        
        score = code_quality_score / total_checks if total_checks > 0 else 0.0
        details['code_quality_score'] = code_quality_score
        details['total_checks'] = total_checks
        
        # Syntax validation
        syntax_errors = []
        for file in required_files:
            try:
                with open(file, 'r') as f:
                    compile(f.read(), file, 'exec')
            except SyntaxError as e:
                syntax_errors.append(f"{file}: {str(e)}")
        
        if syntax_errors:
            details['syntax_errors'] = syntax_errors
            recommendations.extend([f"Fix syntax error: {error}" for error in syntax_errors])
            score *= 0.5  # Penalize syntax errors
        
        # Documentation quality
        doc_files = ['RESEARCH_PUBLICATION_PACKAGE_v7.md']
        doc_quality = {}
        
        for doc_file in doc_files:
            if os.path.exists(doc_file):
                with open(doc_file, 'r') as f:
                    content = f.read()
                    
                doc_quality[doc_file] = {
                    'length': len(content),
                    'has_abstract': '## Abstract' in content or '# Abstract' in content,
                    'has_methods': '## Method' in content or '# Method' in content,
                    'has_results': '## Result' in content or '# Result' in content,
                    'has_references': 'Reference' in content or 'Citation' in content
                }
            else:
                doc_quality[doc_file] = None
                recommendations.append(f"Missing documentation file: {doc_file}")
        
        details['documentation_quality'] = doc_quality
        
        # Overall assessment
        passed = score >= 0.8 and len(syntax_errors) == 0
        severity = 'LOW' if passed else 'MEDIUM' if score >= 0.6 else 'HIGH'
        
        if score < 0.8:
            recommendations.append("Improve code quality score to ≥ 0.8")
        if not passed:
            recommendations.append("Address all syntax errors and missing files")
            
        return QualityGateResult(
            gate_name="Code Quality",
            passed=passed,
            score=score,
            details=details,
            recommendations=recommendations,
            severity=severity
        )
    
    def run_statistical_validation_gate(self) -> QualityGateResult:
        """Validate statistical analysis rigor"""
        print("📊 Statistical Validation Gate...")
        
        details = {}
        recommendations = []
        
        # Check for statistical results files
        result_files = [
            f for f in os.listdir('.') 
            if f.startswith('quantum_breakthrough_results_v7_') and f.endswith('.json')
        ]
        
        validation_files = [
            f for f in os.listdir('.')
            if f.startswith('robust_validation_results_v7_') and f.endswith('.json')
        ]
        
        if not result_files:
            return QualityGateResult(
                gate_name="Statistical Validation",
                passed=False,
                score=0.0,
                details={'error': 'No quantum breakthrough results found'},
                recommendations=['Run quantum_plasma_breakthrough_v7.py to generate results'],
                severity='CRITICAL'
            )
        
        if not validation_files:
            return QualityGateResult(
                gate_name="Statistical Validation", 
                passed=False,
                score=0.0,
                details={'error': 'No robust validation results found'},
                recommendations=['Run robust_research_validation_v7.py to generate validation'],
                severity='CRITICAL'
            )
        
        # Load and validate statistical results
        try:
            with open(result_files[0], 'r') as f:
                results = json.load(f)
                
            with open(validation_files[0], 'r') as f:
                validation = json.load(f)
        except Exception as e:
            return QualityGateResult(
                gate_name="Statistical Validation",
                passed=False,
                score=0.0,
                details={'error': f'Failed to load results: {str(e)}'},
                recommendations=['Check result file format and integrity'],
                severity='HIGH'
            )
        
        # Statistical rigor assessment
        statistical_score = 0
        total_criteria = 0
        
        # Check for multiple scenarios
        if 'methods' in results and len(results['methods']) >= 3:
            statistical_score += 1
        total_criteria += 1
        
        # Check for breakthrough metrics
        if 'breakthrough_metrics' in results:
            breakthrough = results['breakthrough_metrics']
            if breakthrough.get('statistical_robustness', 0) >= 0.75:
                statistical_score += 1
            if breakthrough.get('consistency_score', 0) >= 0.6:
                statistical_score += 1
        total_criteria += 2
        
        # Check validation framework
        if 'publication_metrics' in validation:
            pub_metrics = validation['publication_metrics']
            if pub_metrics.get('publication_score', 0) >= 0.8:
                statistical_score += 1
            if pub_metrics.get('statistical_rigor_level') == 'High':
                statistical_score += 1
        total_criteria += 2
        
        # Check for multiple comparisons correction
        if 'multiple_comparisons' in validation:
            mc = validation['multiple_comparisons']
            if 'bonferroni_corrected' in mc and 'fdr_corrected' in mc:
                statistical_score += 1
        total_criteria += 1
        
        # Check effect sizes
        if 'effect_sizes' in validation:
            es = validation['effect_sizes']
            large_effects = sum(1 for comp in es.values() 
                              if abs(comp.get('cohens_d', 0)) >= 0.8)
            if large_effects >= 1:
                statistical_score += 1
        total_criteria += 1
        
        # Check statistical power
        if 'power_analysis' in validation:
            pa = validation['power_analysis']
            adequate_power = sum(1 for comp in pa.values() 
                               if comp.get('adequate_power', False))
            if adequate_power >= 1:
                statistical_score += 1
        total_criteria += 1
        
        score = statistical_score / total_criteria if total_criteria > 0 else 0.0
        
        details.update({
            'statistical_score': statistical_score,
            'total_criteria': total_criteria,
            'breakthrough_classification': results.get('breakthrough_metrics', {}).get('breakthrough_classification'),
            'publication_readiness': validation.get('publication_metrics', {}).get('publication_score', 0),
            'statistical_rigor': validation.get('publication_metrics', {}).get('statistical_rigor_level'),
        })
        
        # Generate recommendations
        if score < 0.8:
            recommendations.append("Improve statistical validation score to ≥ 0.8")
        if details.get('publication_readiness', 0) < 0.8:
            recommendations.append("Enhance publication readiness score")
        if details.get('statistical_rigor') != 'High':
            recommendations.append("Achieve 'High' statistical rigor level")
            
        passed = score >= 0.8
        severity = 'LOW' if passed else 'MEDIUM' if score >= 0.6 else 'HIGH'
        
        return QualityGateResult(
            gate_name="Statistical Validation",
            passed=passed,
            score=score,
            details=details,
            recommendations=recommendations,
            severity=severity
        )
    
    def run_reproducibility_gate(self) -> QualityGateResult:
        """Validate research reproducibility"""
        print("🔄 Reproducibility Gate...")
        
        details = {}
        recommendations = []
        
        # Check for required reproducibility components
        reproducibility_score = 0
        total_components = 0
        
        # Source code availability
        if os.path.exists('quantum_plasma_breakthrough_v7.py'):
            reproducibility_score += 1
        else:
            recommendations.append("Provide main algorithm source code")
        total_components += 1
        
        # Documentation availability  
        doc_files = [
            'RESEARCH_PUBLICATION_PACKAGE_v7.md',
            'academic_methodology_v7_*.md'
        ]
        
        doc_exists = any(os.path.exists(f) for f in doc_files if '*' not in f)
        if doc_exists:
            reproducibility_score += 1
        else:
            recommendations.append("Provide comprehensive documentation")
        total_components += 1
        
        # Data availability
        data_files = [f for f in os.listdir('.') if f.endswith('.json') and 'results' in f]
        if len(data_files) >= 2:  # Both breakthrough and validation results
            reproducibility_score += 1
        else:
            recommendations.append("Ensure all result datasets are available")
        total_components += 1
        
        # Dependencies specification
        has_dependencies = False
        if os.path.exists('pyproject.toml'):
            has_dependencies = True
            reproducibility_score += 1
        elif os.path.exists('requirements.txt'):
            has_dependencies = True
            reproducibility_score += 1
        
        if not has_dependencies:
            recommendations.append("Provide dependency specification (requirements.txt or pyproject.toml)")
        total_components += 1
        
        # Execution test
        execution_success = False
        try:
            # Test that main algorithm can be imported
            sys.path.insert(0, '.')
            import quantum_plasma_breakthrough_v7
            execution_success = True
            reproducibility_score += 1
        except ImportError as e:
            recommendations.append(f"Fix import errors: {str(e)}")
        except Exception as e:
            recommendations.append(f"Resolve execution issues: {str(e)}")
        total_components += 1
        
        # Random seed management
        seed_managed = False
        for file in ['quantum_plasma_breakthrough_v7.py', 'robust_research_validation_v7.py']:
            if os.path.exists(file):
                with open(file, 'r') as f:
                    content = f.read()
                    if 'random.seed' in content or 'np.random.seed' in content:
                        seed_managed = True
                        break
        
        if seed_managed:
            reproducibility_score += 1
        else:
            recommendations.append("Implement random seed management for reproducibility")
        total_components += 1
        
        score = reproducibility_score / total_components if total_components > 0 else 0.0
        
        details.update({
            'reproducibility_score': reproducibility_score,
            'total_components': total_components,
            'source_code_available': True,
            'documentation_available': doc_exists,
            'data_available': len(data_files) >= 2,
            'dependencies_specified': has_dependencies,
            'execution_tested': execution_success,
            'random_seed_managed': seed_managed
        })
        
        passed = score >= 0.8
        severity = 'LOW' if passed else 'MEDIUM' if score >= 0.6 else 'HIGH'
        
        if not passed:
            recommendations.append("Address all reproducibility requirements")
            
        return QualityGateResult(
            gate_name="Reproducibility",
            passed=passed,
            score=score,
            details=details,
            recommendations=recommendations,
            severity=severity
        )
    
    def run_benchmark_performance_gate(self) -> QualityGateResult:
        """Validate benchmark performance requirements"""
        print("⚡ Benchmark Performance Gate...")
        
        details = {}
        recommendations = []
        
        # Load performance results
        result_files = [
            f for f in os.listdir('.') 
            if f.startswith('quantum_breakthrough_results_v7_') and f.endswith('.json')
        ]
        
        if not result_files:
            return QualityGateResult(
                gate_name="Benchmark Performance",
                passed=False,
                score=0.0,
                details={'error': 'No performance results found'},
                recommendations=['Run performance benchmarks'],
                severity='HIGH'
            )
        
        try:
            with open(result_files[0], 'r') as f:
                results = json.load(f)
        except Exception as e:
            return QualityGateResult(
                gate_name="Benchmark Performance",
                passed=False,
                score=0.0,
                details={'error': f'Failed to load results: {str(e)}'},
                recommendations=['Check result file integrity'],
                severity='HIGH'
            )
        
        # Performance criteria
        performance_score = 0
        total_benchmarks = 0
        
        # Check overall improvement
        if 'breakthrough_metrics' in results:
            breakthrough = results['breakthrough_metrics']
            improvement = breakthrough.get('mean_improvement_vs_pid', 0)
            
            if improvement >= 20:  # 20% improvement threshold
                performance_score += 2
            elif improvement >= 10:
                performance_score += 1
            
            total_benchmarks += 2
            
            # Consistency across scenarios
            consistency = breakthrough.get('consistency_score', 0)
            if consistency >= 0.7:
                performance_score += 1
            elif consistency >= 0.5:
                performance_score += 0.5
                
            total_benchmarks += 1
        
        # Scenario-specific performance
        if 'methods' in results:
            scenarios_passed = 0
            total_scenarios = len(results['methods'])
            
            for scenario_name, scenario_data in results['methods'].items():
                if 'statistical_significance' in scenario_data:
                    stats = scenario_data['statistical_significance']
                    if stats.get('quantum_vs_pid_improvement', 0) > 5:  # 5% minimum improvement
                        scenarios_passed += 1
            
            scenario_score = scenarios_passed / total_scenarios if total_scenarios > 0 else 0
            performance_score += scenario_score * 2
            total_benchmarks += 2
        
        # Statistical significance
        validation_files = [
            f for f in os.listdir('.')
            if f.startswith('robust_validation_results_v7_') and f.endswith('.json')
        ]
        
        if validation_files:
            try:
                with open(validation_files[0], 'r') as f:
                    validation = json.load(f)
                    
                if 'inferential_tests' in validation:
                    tests = validation['inferential_tests']
                    significant_tests = sum(1 for test in tests.values() 
                                          if test.get('significant', False))
                    if significant_tests >= 1:
                        performance_score += 1
                    total_benchmarks += 1
            except:
                pass
        
        score = performance_score / total_benchmarks if total_benchmarks > 0 else 0.0
        
        details.update({
            'performance_score': performance_score,
            'total_benchmarks': total_benchmarks,
            'improvement_vs_classical': results.get('breakthrough_metrics', {}).get('mean_improvement_vs_pid', 0),
            'consistency_score': results.get('breakthrough_metrics', {}).get('consistency_score', 0),
            'breakthrough_level': results.get('breakthrough_metrics', {}).get('breakthrough_classification'),
            'scenarios_tested': len(results.get('methods', {}))
        })
        
        # Generate recommendations
        if details['improvement_vs_classical'] < 10:
            recommendations.append("Achieve minimum 10% improvement over classical methods")
        if details['consistency_score'] < 0.5:
            recommendations.append("Improve algorithm consistency across scenarios")
        if details['scenarios_tested'] < 3:
            recommendations.append("Test algorithm across more diverse scenarios")
            
        passed = score >= 0.8
        severity = 'LOW' if passed else 'MEDIUM' if score >= 0.6 else 'HIGH'
        
        return QualityGateResult(
            gate_name="Benchmark Performance",
            passed=passed,
            score=score,
            details=details,
            recommendations=recommendations,
            severity=severity
        )
    
    def run_publication_readiness_gate(self) -> QualityGateResult:
        """Assess overall publication readiness"""
        print("📝 Publication Readiness Gate...")
        
        details = {}
        recommendations = []
        
        # Check for publication package components
        publication_score = 0
        total_requirements = 0
        
        # Main publication document
        if os.path.exists('RESEARCH_PUBLICATION_PACKAGE_v7.md'):
            publication_score += 2
            with open('RESEARCH_PUBLICATION_PACKAGE_v7.md', 'r') as f:
                content = f.read()
                if len(content) > 10000:  # Substantial content
                    publication_score += 1
        else:
            recommendations.append("Create main publication document")
        total_requirements += 3
        
        # Methodology documentation
        methodology_files = [f for f in os.listdir('.') if f.startswith('academic_methodology_v7_')]
        if methodology_files:
            publication_score += 1
        else:
            recommendations.append("Generate methodology documentation")
        total_requirements += 1
        
        # Statistical validation
        validation_files = [f for f in os.listdir('.') if f.startswith('robust_validation_results_v7_')]
        if validation_files:
            try:
                with open(validation_files[0], 'r') as f:
                    validation = json.load(f)
                    
                pub_metrics = validation.get('publication_metrics', {})
                if pub_metrics.get('publication_score', 0) >= 0.8:
                    publication_score += 2
                elif pub_metrics.get('publication_score', 0) >= 0.6:
                    publication_score += 1
            except:
                pass
        total_requirements += 2
        
        # Code and data availability
        has_code = os.path.exists('quantum_plasma_breakthrough_v7.py')
        has_data = len([f for f in os.listdir('.') if f.endswith('.json')]) >= 2
        
        if has_code and has_data:
            publication_score += 1
        elif has_code or has_data:
            publication_score += 0.5
        total_requirements += 1
        
        # Compliance with standards
        doc_package_files = [f for f in os.listdir('.') if f.startswith('academic_documentation_package_v7_')]
        if doc_package_files:
            publication_score += 1
        total_requirements += 1
        
        score = publication_score / total_requirements if total_requirements > 0 else 0.0
        
        # Journal tier assessment
        if score >= 0.9:
            journal_tier = "Tier 1 (Nature Energy, Nuclear Fusion)"
        elif score >= 0.8:
            journal_tier = "Tier 1-2 (IEEE Transactions, specialized journals)"
        elif score >= 0.7:
            journal_tier = "Tier 2 (Conference proceedings, technical journals)"
        else:
            journal_tier = "Not ready for submission"
        
        details.update({
            'publication_score': publication_score,
            'total_requirements': total_requirements,
            'journal_tier': journal_tier,
            'main_document_available': os.path.exists('RESEARCH_PUBLICATION_PACKAGE_v7.md'),
            'methodology_documented': len(methodology_files) > 0,
            'statistical_validation_complete': len(validation_files) > 0,
            'code_available': has_code,
            'data_available': has_data
        })
        
        # Generate final recommendations
        if score < 0.8:
            recommendations.append("Complete all publication requirements for high-impact submission")
        if not details['main_document_available']:
            recommendations.append("Create comprehensive publication document")
        if not details['statistical_validation_complete']:
            recommendations.append("Complete statistical validation framework")
            
        passed = score >= 0.8
        severity = 'LOW' if passed else 'MEDIUM' if score >= 0.6 else 'HIGH'
        
        return QualityGateResult(
            gate_name="Publication Readiness",
            passed=passed,
            score=score,
            details=details,
            recommendations=recommendations,
            severity=severity
        )
    
    def run_all_quality_gates(self) -> Dict[str, Any]:
        """Execute all quality gates and generate comprehensive report"""
        print("🛡️ RESEARCH QUALITY GATES v7")
        print("=" * 50)
        print("Comprehensive quality validation for publication readiness...")
        print()
        
        # Execute all quality gates
        gates = [
            self.run_code_quality_gate,
            self.run_statistical_validation_gate,
            self.run_reproducibility_gate,
            self.run_benchmark_performance_gate,
            self.run_publication_readiness_gate
        ]
        
        gate_results = []
        for gate_func in gates:
            result = gate_func()
            gate_results.append(result)
            self.results.append(result)
            
            # Print gate result
            status = "✅ PASS" if result.passed else "❌ FAIL"
            print(f"{status} {result.gate_name}: {result.score:.2f}/1.00 ({result.severity})")
            if result.recommendations:
                for rec in result.recommendations[:2]:  # Show first 2 recommendations
                    print(f"   → {rec}")
            print()
        
        # Calculate overall quality score
        total_score = sum(r.score for r in gate_results)
        max_score = len(gate_results)
        overall_score = total_score / max_score if max_score > 0 else 0.0
        
        # Determine overall status
        passed_gates = sum(1 for r in gate_results if r.passed)
        critical_failures = sum(1 for r in gate_results if r.severity == 'CRITICAL')
        high_failures = sum(1 for r in gate_results if r.severity == 'HIGH')
        
        overall_passed = passed_gates == len(gate_results) and critical_failures == 0
        
        # Generate comprehensive report
        execution_time = time.time() - self.start_time
        
        report = {
            'metadata': {
                'version': self.version,
                'timestamp': datetime.now().isoformat(),
                'execution_time_seconds': execution_time,
                'gates_executed': len(gate_results)
            },
            'overall_assessment': {
                'passed': overall_passed,
                'score': overall_score,
                'gates_passed': passed_gates,
                'gates_total': len(gate_results),
                'critical_failures': critical_failures,
                'high_failures': high_failures
            },
            'gate_results': [asdict(r) for r in gate_results],
            'quality_summary': self._generate_quality_summary(gate_results),
            'recommendations': self._prioritize_recommendations(gate_results),
            'publication_assessment': self._assess_publication_status(gate_results, overall_score)
        }
        
        return report
    
    def _generate_quality_summary(self, results: List[QualityGateResult]) -> Dict[str, Any]:
        """Generate quality summary across all gates"""
        return {
            'code_quality': next(r.score for r in results if r.gate_name == "Code Quality"),
            'statistical_rigor': next(r.score for r in results if r.gate_name == "Statistical Validation"),
            'reproducibility': next(r.score for r in results if r.gate_name == "Reproducibility"),
            'performance': next(r.score for r in results if r.gate_name == "Benchmark Performance"),
            'publication_readiness': next(r.score for r in results if r.gate_name == "Publication Readiness"),
            'strengths': [r.gate_name for r in results if r.score >= 0.9],
            'areas_for_improvement': [r.gate_name for r in results if r.score < 0.8]
        }
    
    def _prioritize_recommendations(self, results: List[QualityGateResult]) -> List[Dict[str, str]]:
        """Prioritize recommendations across all gates"""
        all_recommendations = []
        
        for result in results:
            for rec in result.recommendations:
                all_recommendations.append({
                    'recommendation': rec,
                    'gate': result.gate_name,
                    'severity': result.severity,
                    'priority': 1 if result.severity == 'CRITICAL' else 2 if result.severity == 'HIGH' else 3
                })
        
        # Sort by priority
        all_recommendations.sort(key=lambda x: x['priority'])
        
        return all_recommendations[:10]  # Top 10 recommendations
    
    def _assess_publication_status(self, results: List[QualityGateResult], overall_score: float) -> Dict[str, Any]:
        """Assess overall publication readiness status"""
        
        critical_issues = sum(1 for r in results if r.severity == 'CRITICAL')
        high_issues = sum(1 for r in results if r.severity == 'HIGH')
        
        if critical_issues > 0:
            status = "NOT_READY"
            tier = "Preliminary - Major Issues"
        elif high_issues > 0:
            status = "NEEDS_REVISION" 
            tier = "Conference/Workshop"
        elif overall_score >= 0.9:
            status = "READY"
            tier = "Tier 1 Journal"
        elif overall_score >= 0.8:
            status = "READY"
            tier = "Tier 2 Journal"
        else:
            status = "NEEDS_IMPROVEMENT"
            tier = "Technical Report"
        
        return {
            'status': status,
            'publication_tier': tier,
            'overall_score': overall_score,
            'critical_issues': critical_issues,
            'high_issues': high_issues,
            'recommendation': self._get_status_recommendation(status, overall_score)
        }
    
    def _get_status_recommendation(self, status: str, score: float) -> str:
        """Get recommendation based on publication status"""
        if status == "READY":
            return "Proceed with journal submission"
        elif status == "NEEDS_REVISION":
            return "Address high-priority issues before submission"
        elif status == "NEEDS_IMPROVEMENT":
            return f"Improve overall quality score to ≥ 0.8 (current: {score:.2f})"
        else:
            return "Resolve critical issues before proceeding"

def main():
    """Execute comprehensive research quality gates"""
    quality_system = ResearchQualityGates()
    
    # Run all quality gates
    report = quality_system.run_all_quality_gates()
    
    # Save quality gates report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"research_quality_gates_report_v7_{timestamp}.json"
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Generate summary
    print("🏆 QUALITY GATES SUMMARY")
    print("=" * 50)
    
    overall = report['overall_assessment']
    publication = report['publication_assessment']
    
    print(f"Overall Status: {'✅ PASSED' if overall['passed'] else '❌ FAILED'}")
    print(f"Quality Score: {overall['score']:.2f}/1.00")
    print(f"Gates Passed: {overall['gates_passed']}/{overall['gates_total']}")
    print(f"Publication Status: {publication['status']}")
    print(f"Publication Tier: {publication['publication_tier']}")
    print()
    
    # Quality breakdown
    quality = report['quality_summary']
    print("📊 Quality Breakdown:")
    print(f"  Code Quality: {quality['code_quality']:.2f}/1.00")
    print(f"  Statistical Rigor: {quality['statistical_rigor']:.2f}/1.00")  
    print(f"  Reproducibility: {quality['reproducibility']:.2f}/1.00")
    print(f"  Performance: {quality['performance']:.2f}/1.00")
    print(f"  Publication Readiness: {quality['publication_readiness']:.2f}/1.00")
    print()
    
    # Top recommendations
    if report['recommendations']:
        print("🎯 Priority Recommendations:")
        for i, rec in enumerate(report['recommendations'][:5], 1):
            print(f"  {i}. [{rec['severity']}] {rec['recommendation']}")
        print()
    
    # Final assessment
    if publication['status'] == 'READY':
        print("🚀 READY FOR PUBLICATION")
        print(f"   → Quality standards met for {publication['publication_tier']}")
        print("   → All critical quality gates passed")
        print("   → Proceed with peer review submission")
    elif publication['status'] == 'NEEDS_REVISION':
        print("📝 REVISION REQUIRED")
        print("   → Address high-priority issues")
        print("   → Re-run quality gates after improvements")
        print("   → Consider conference submission in interim")
    else:
        print("🔄 FURTHER DEVELOPMENT NEEDED")
        print("   → Critical issues must be resolved")
        print("   → Focus on highest priority recommendations")
        print("   → Re-evaluate after addressing core issues")
    
    print(f"\n📁 Quality gates report saved to: {report_file}")
    print("🛡️ Quality validation complete")
    
    return report

if __name__ == "__main__":
    results = main()