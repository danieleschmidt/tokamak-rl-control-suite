#!/usr/bin/env python3
"""
Quality Gates Execution: Comprehensive Testing and Validation
Running security, performance, compliance, and documentation checks.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import time
import subprocess
import importlib
import ast
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple

class SecurityScanner:
    """Security vulnerability scanner for the codebase."""
    
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.vulnerabilities = []
        
    def scan_imports(self) -> List[Dict[str, Any]]:
        """Scan for potentially dangerous imports."""
        dangerous_imports = [
            'eval', 'exec', 'compile', 'open', '__import__',
            'subprocess.call', 'subprocess.run', 'os.system'
        ]
        
        vulnerabilities = []
        python_files = list(self.project_root.rglob("*.py"))
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for dangerous in dangerous_imports:
                    if dangerous in content:
                        # More detailed analysis
                        lines = content.split('\n')
                        for line_num, line in enumerate(lines, 1):
                            if dangerous in line and not line.strip().startswith('#'):
                                vulnerabilities.append({
                                    'file': str(py_file.relative_to(self.project_root)),
                                    'line': line_num,
                                    'issue': f"Potentially dangerous: {dangerous}",
                                    'severity': 'MEDIUM',
                                    'line_content': line.strip()
                                })
            except Exception as e:
                continue
                
        return vulnerabilities
    
    def check_hardcoded_secrets(self) -> List[Dict[str, Any]]:
        """Check for hardcoded secrets and credentials."""
        secret_patterns = [
            (r'password\s*=\s*["\'][^"\']+["\']', 'Hardcoded password'),
            (r'api_key\s*=\s*["\'][^"\']+["\']', 'Hardcoded API key'),
            (r'secret\s*=\s*["\'][^"\']+["\']', 'Hardcoded secret'),
            (r'token\s*=\s*["\'][^"\']{20,}["\']', 'Hardcoded token'),
        ]
        
        vulnerabilities = []
        python_files = list(self.project_root.rglob("*.py"))
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for pattern, description in secret_patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        vulnerabilities.append({
                            'file': str(py_file.relative_to(self.project_root)),
                            'line': line_num,
                            'issue': description,
                            'severity': 'HIGH',
                            'line_content': match.group(0)
                        })
            except Exception:
                continue
                
        return vulnerabilities
    
    def validate_input_handling(self) -> List[Dict[str, Any]]:
        """Check for proper input validation."""
        issues = []
        python_files = list(self.project_root.rglob("*.py"))
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Look for functions that should have input validation
                lines = content.split('\n')
                for line_num, line in enumerate(lines, 1):
                    # Check for user input without validation
                    if 'input(' in line and 'validate' not in line.lower():
                        issues.append({
                            'file': str(py_file.relative_to(self.project_root)),
                            'line': line_num,
                            'issue': 'User input without validation',
                            'severity': 'LOW',
                            'line_content': line.strip()
                        })
            except Exception:
                continue
                
        return issues

class PerformanceBenchmark:
    """Performance benchmarking and validation."""
    
    def __init__(self):
        self.benchmarks = {}
        
    def benchmark_environment_performance(self) -> Dict[str, Any]:
        """Benchmark environment step performance."""
        from tokamak_rl.physics import TokamakConfig
        from tokamak_rl.environment import TokamakEnv
        
        config = TokamakConfig.from_preset("ITER")
        env_config = {'tokamak_config': config, 'enable_safety': False}
        env = TokamakEnv(env_config)
        
        # Benchmark reset performance
        reset_times = []
        for _ in range(10):
            start_time = time.time()
            obs, info = env.reset()
            reset_times.append(time.time() - start_time)
        
        # Benchmark step performance
        step_times = []
        obs, info = env.reset()
        
        for _ in range(100):
            action = np.random.uniform(-0.5, 0.5, 8)
            start_time = time.time()
            obs, reward, terminated, truncated, info = env.step(action)
            step_times.append(time.time() - start_time)
            
            if terminated or truncated:
                obs, info = env.reset()
        
        return {
            'reset_avg_ms': np.mean(reset_times) * 1000,
            'reset_std_ms': np.std(reset_times) * 1000,
            'step_avg_ms': np.mean(step_times) * 1000,
            'step_std_ms': np.std(step_times) * 1000,
            'steps_per_second': 1.0 / np.mean(step_times),
            'total_steps_tested': len(step_times)
        }
    
    def benchmark_physics_solver(self) -> Dict[str, Any]:
        """Benchmark physics solver performance."""
        from tokamak_rl.physics import TokamakConfig, PlasmaState, GradShafranovSolver
        
        config = TokamakConfig.from_preset("SPARC")
        solver = GradShafranovSolver(config)
        state = PlasmaState(config)
        
        # Benchmark equilibrium solving
        solve_times = []
        for _ in range(50):
            pf_currents = np.random.uniform(-1, 1, 6)
            start_time = time.time()
            result_state = solver.solve_equilibrium(state, pf_currents)
            solve_times.append(time.time() - start_time)
        
        return {
            'solve_avg_ms': np.mean(solve_times) * 1000,
            'solve_std_ms': np.std(solve_times) * 1000,
            'solves_per_second': 1.0 / np.mean(solve_times),
            'total_solves_tested': len(solve_times)
        }

class CodeQualityChecker:
    """Code quality and standards checker."""
    
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        
    def check_docstring_coverage(self) -> Dict[str, Any]:
        """Check docstring coverage for functions and classes."""
        python_files = list(self.project_root.rglob("*.py"))
        
        total_functions = 0
        documented_functions = 0
        total_classes = 0 
        documented_classes = 0
        
        for py_file in python_files:
            if 'test_' in py_file.name or py_file.name.startswith('test'):
                continue  # Skip test files
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        if not node.name.startswith('_'):  # Skip private functions
                            total_functions += 1
                            if ast.get_docstring(node):
                                documented_functions += 1
                    
                    elif isinstance(node, ast.ClassDef):
                        total_classes += 1
                        if ast.get_docstring(node):
                            documented_classes += 1
                            
            except Exception as e:
                continue
        
        function_coverage = documented_functions / total_functions if total_functions > 0 else 0
        class_coverage = documented_classes / total_classes if total_classes > 0 else 0
        
        return {
            'total_functions': total_functions,
            'documented_functions': documented_functions,
            'function_docstring_coverage': function_coverage,
            'total_classes': total_classes,
            'documented_classes': documented_classes,
            'class_docstring_coverage': class_coverage,
            'overall_coverage': (documented_functions + documented_classes) / (total_functions + total_classes) if (total_functions + total_classes) > 0 else 0
        }
    
    def check_code_complexity(self) -> Dict[str, Any]:
        """Check code complexity metrics."""
        python_files = list(self.project_root.rglob("*.py"))
        
        total_lines = 0
        total_files = 0
        long_functions = []
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    total_lines += len(lines)
                    total_files += 1
                    
                # Simple complexity check - look for very long functions
                content = ''.join(lines)
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        if hasattr(node, 'end_lineno') and hasattr(node, 'lineno'):
                            func_length = node.end_lineno - node.lineno
                            if func_length > 50:  # Functions longer than 50 lines
                                long_functions.append({
                                    'file': str(py_file.relative_to(self.project_root)),
                                    'function': node.name,
                                    'lines': func_length
                                })
                                
            except Exception:
                continue
        
        avg_file_length = total_lines / total_files if total_files > 0 else 0
        
        return {
            'total_files': total_files,
            'total_lines': total_lines,
            'avg_file_length': avg_file_length,
            'long_functions': long_functions,
            'complexity_issues': len(long_functions)
        }

def run_security_scan():
    """Run comprehensive security scan."""
    print("🔒 Running Security Scan...")
    
    scanner = SecurityScanner(Path.cwd())
    
    # Scan for dangerous imports
    import_vulns = scanner.scan_imports()
    print(f"  ✓ Import vulnerabilities: {len(import_vulns)}")
    
    # Check for hardcoded secrets
    secret_vulns = scanner.check_hardcoded_secrets()
    print(f"  ✓ Hardcoded secrets check: {len(secret_vulns)} potential issues")
    
    # Input validation check
    input_issues = scanner.validate_input_handling()
    print(f"  ✓ Input validation check: {len(input_issues)} potential issues")
    
    total_vulns = len(import_vulns) + len(secret_vulns) + len(input_issues)
    high_severity = len([v for v in import_vulns + secret_vulns + input_issues if v.get('severity') == 'HIGH'])
    
    if high_severity > 0:
        print(f"  ⚠ {high_severity} HIGH severity issues found")
    else:
        print(f"  ✅ No high-severity security issues found")
    
    return total_vulns == 0 or high_severity == 0

def run_performance_benchmarks():
    """Run performance benchmarks and validate against thresholds."""
    print("\n⚡ Running Performance Benchmarks...")
    
    benchmark = PerformanceBenchmark()
    
    # Environment performance
    env_perf = benchmark.benchmark_environment_performance()
    print(f"  ✓ Environment step time: {env_perf['step_avg_ms']:.2f}±{env_perf['step_std_ms']:.2f} ms")
    print(f"  ✓ Environment throughput: {env_perf['steps_per_second']:.0f} steps/second")
    
    # Physics solver performance
    physics_perf = benchmark.benchmark_physics_solver()
    print(f"  ✓ Physics solver time: {physics_perf['solve_avg_ms']:.2f}±{physics_perf['solve_std_ms']:.2f} ms")
    print(f"  ✓ Physics solver throughput: {physics_perf['solves_per_second']:.0f} solves/second")
    
    # Performance thresholds
    step_threshold = 50.0  # 50ms max per step
    solve_threshold = 10.0  # 10ms max per solve
    
    step_performance_ok = env_perf['step_avg_ms'] < step_threshold
    solve_performance_ok = physics_perf['solve_avg_ms'] < solve_threshold
    
    if step_performance_ok and solve_performance_ok:
        print("  ✅ All performance benchmarks passed")
        return True
    else:
        print("  ⚠ Some performance benchmarks failed")
        return False

def run_code_quality_checks():
    """Run code quality and standards checks."""
    print("\n📋 Running Code Quality Checks...")
    
    checker = CodeQualityChecker(Path.cwd())
    
    # Documentation coverage
    doc_coverage = checker.check_docstring_coverage()
    print(f"  ✓ Function documentation: {doc_coverage['function_docstring_coverage']:.1%}")
    print(f"  ✓ Class documentation: {doc_coverage['class_docstring_coverage']:.1%}")
    print(f"  ✓ Overall documentation: {doc_coverage['overall_coverage']:.1%}")
    
    # Code complexity
    complexity = checker.check_code_complexity()
    print(f"  ✓ Average file length: {complexity['avg_file_length']:.0f} lines")
    print(f"  ✓ Long functions: {complexity['complexity_issues']}")
    
    # Quality thresholds
    doc_threshold = 0.7  # 70% documentation coverage
    complexity_threshold = 5  # Max 5 overly long functions
    
    doc_quality_ok = doc_coverage['overall_coverage'] >= doc_threshold
    complexity_ok = complexity['complexity_issues'] <= complexity_threshold
    
    if doc_quality_ok and complexity_ok:
        print("  ✅ Code quality checks passed")
        return True
    else:
        if not doc_quality_ok:
            print(f"  ⚠ Documentation coverage below threshold: {doc_coverage['overall_coverage']:.1%} < {doc_threshold:.1%}")
        if not complexity_ok:
            print(f"  ⚠ Too many complex functions: {complexity['complexity_issues']} > {complexity_threshold}")
        return False

def run_integration_tests():
    """Run integration tests for key workflows."""
    print("\n🔧 Running Integration Tests...")
    
    try:
        # Test full workflow
        from tokamak_rl.physics import TokamakConfig
        from tokamak_rl.environment import TokamakEnv
        
        # Test multiple configurations
        configs = ["ITER", "SPARC", "DIII-D"]
        successful_configs = 0
        
        for config_name in configs:
            try:
                config = TokamakConfig.from_preset(config_name)
                env_config = {'tokamak_config': config, 'enable_safety': False}
                env = TokamakEnv(env_config)
                
                # Test basic workflow
                obs, info = env.reset()
                action = np.random.uniform(-0.2, 0.2, 8)
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Validate results
                assert obs is not None, f"Observation is None for {config_name}"
                assert isinstance(reward, (int, float)), f"Invalid reward type for {config_name}"
                assert info is not None, f"Info is None for {config_name}"
                
                successful_configs += 1
                print(f"  ✓ {config_name} configuration test passed")
                
            except Exception as e:
                print(f"  ❌ {config_name} configuration test failed: {e}")
        
        if successful_configs >= len(configs) * 0.8:  # 80% success rate
            print("  ✅ Integration tests passed")
            return True
        else:
            print("  ⚠ Integration tests failed - insufficient success rate")
            return False
            
    except Exception as e:
        print(f"  ❌ Integration tests failed: {e}")
        return False

def generate_quality_report():
    """Generate comprehensive quality report."""
    print("\n📊 Generating Quality Report...")
    
    report = {
        'timestamp': time.time(),
        'security_passed': False,
        'performance_passed': False,
        'quality_passed': False,
        'integration_passed': False,
        'overall_passed': False
    }
    
    # Run all checks
    report['security_passed'] = run_security_scan()
    report['performance_passed'] = run_performance_benchmarks()
    report['quality_passed'] = run_code_quality_checks()
    report['integration_passed'] = run_integration_tests()
    
    # Overall assessment
    report['overall_passed'] = all([
        report['security_passed'],
        report['performance_passed'], 
        report['quality_passed'],
        report['integration_passed']
    ])
    
    return report

if __name__ == "__main__":
    print("TOKAMAK RL CONTROL SUITE - QUALITY GATES EXECUTION")
    print("=" * 70)
    
    report = generate_quality_report()
    
    print(f"\n📈 QUALITY GATES SUMMARY:")
    print(f"🔒 Security Scan: {'✅ PASS' if report['security_passed'] else '❌ FAIL'}")
    print(f"⚡ Performance: {'✅ PASS' if report['performance_passed'] else '❌ FAIL'}")
    print(f"📋 Code Quality: {'✅ PASS' if report['quality_passed'] else '❌ FAIL'}")
    print(f"🔧 Integration: {'✅ PASS' if report['integration_passed'] else '❌ FAIL'}")
    
    if report['overall_passed']:
        print("\n🎉 ALL QUALITY GATES PASSED!")
        print("✅ System ready for production deployment")
        print("✅ Security validated")
        print("✅ Performance benchmarks met")
        print("✅ Code quality standards achieved")
        print("✅ Integration tests successful")
        
        print(f"\n🚀 AUTONOMOUS SDLC EXECUTION COMPLETE!")
        print("Ready to proceed to deployment phase.")
        sys.exit(0)
    else:
        print("\n⚠️ QUALITY GATES FAILED")
        print("Some quality checks did not pass. Review above for details.")
        sys.exit(1)