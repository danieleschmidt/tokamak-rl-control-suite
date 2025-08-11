#!/usr/bin/env python3
"""
Comprehensive quality gates validation system for tokamak-rl.
Implements security scanning, performance testing, code quality, and compliance checks.
"""

import sys
import os
import time
import json
import subprocess
import hashlib
import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import ast
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class QualityMetrics:
    """Track quality assurance metrics."""
    code_coverage: float = 0.0
    security_score: float = 0.0
    performance_score: float = 0.0
    maintainability_score: float = 0.0
    reliability_score: float = 0.0
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    security_issues: int = 0
    performance_issues: int = 0
    code_smells: int = 0

@dataclass
class SecurityIssue:
    """Security issue report."""
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    category: str
    description: str
    file_path: str
    line_number: int
    recommendation: str

class SecurityScanner:
    """Security vulnerability scanner."""
    
    def __init__(self):
        self.logger = logging.getLogger("SecurityScanner")
        
        # Security patterns to detect
        self.security_patterns = {
            "hardcoded_secrets": [
                r"(?i)(password|passwd|pwd|secret|key|token)\s*=\s*[\"'][^\"']+[\"']",
                r"(?i)(api_key|apikey|access_key)\s*=\s*[\"'][^\"']+[\"']",
                r"(?i)(private_key|secret_key)\s*=\s*[\"'][^\"']+[\"']"
            ],
            "sql_injection": [
                r"execute\s*\(\s*[\"'].*%s.*[\"']",
                r"cursor\.execute\s*\(\s*[\"'].*\+.*[\"']",
                r"query\s*=.*\+.*input"
            ],
            "command_injection": [
                r"os\.system\s*\(\s*[\"'].*\+",
                r"subprocess\.(call|run|Popen)\s*\(\s*[\"'].*\+",
                r"exec\s*\(\s*[\"'].*\+.*[\"']"
            ],
            "insecure_random": [
                r"random\.random\(\)",
                r"random\.choice\(",
                r"random\.randint\("
            ],
            "unsafe_deserialization": [
                r"pickle\.loads\s*\(",
                r"yaml\.load\s*\(\s*[^,]",  # yaml.load without Loader
                r"eval\s*\("
            ]
        }
    
    def scan_file(self, file_path: str) -> List[SecurityIssue]:
        """Scan a file for security issues."""
        issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
                
            for line_num, line in enumerate(lines, 1):
                for category, patterns in self.security_patterns.items():
                    for pattern in patterns:
                        if re.search(pattern, line):
                            severity = self._assess_severity(category, line)
                            issues.append(SecurityIssue(
                                severity=severity,
                                category=category,
                                description=f"Potential {category.replace('_', ' ')} vulnerability",
                                file_path=file_path,
                                line_number=line_num,
                                recommendation=self._get_recommendation(category)
                            ))
                            
        except Exception as e:
            self.logger.error(f"Error scanning {file_path}: {e}")
        
        return issues
    
    def _assess_severity(self, category: str, line: str) -> str:
        """Assess severity of security issue."""
        severity_map = {
            "hardcoded_secrets": "HIGH",
            "sql_injection": "CRITICAL", 
            "command_injection": "CRITICAL",
            "insecure_random": "MEDIUM",
            "unsafe_deserialization": "HIGH"
        }
        return severity_map.get(category, "MEDIUM")
    
    def _get_recommendation(self, category: str) -> str:
        """Get security recommendation."""
        recommendations = {
            "hardcoded_secrets": "Use environment variables or secure key management",
            "sql_injection": "Use parameterized queries or ORM",
            "command_injection": "Validate and sanitize all inputs",
            "insecure_random": "Use cryptographically secure random (secrets module)",
            "unsafe_deserialization": "Use safe serialization formats like JSON"
        }
        return recommendations.get(category, "Review and fix security issue")
    
    def scan_directory(self, directory: str) -> List[SecurityIssue]:
        """Scan entire directory for security issues."""
        all_issues = []
        
        for root, dirs, files in os.walk(directory):
            # Skip common non-source directories
            dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', '.pytest_cache', 'node_modules']]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    issues = self.scan_file(file_path)
                    all_issues.extend(issues)
        
        return all_issues

class PerformanceTester:
    """Performance testing and benchmarking."""
    
    def __init__(self):
        self.logger = logging.getLogger("PerformanceTester")
        
    def test_system_performance(self) -> Dict[str, Any]:
        """Test overall system performance."""
        self.logger.info("Starting performance tests...")
        
        # Import our dependency-free system for testing
        sys.path.insert(0, os.path.dirname(__file__))
        from dependency_free_core import DependencyFreeTokamakSystem
        
        performance_results = {}
        
        # Test 1: Basic operation speed
        start_time = time.time()
        system = DependencyFreeTokamakSystem("ITER")
        system.reset()
        
        for i in range(100):
            action = [0.1 * ((-1) ** i) for _ in range(8)]
            system.step(action)
        
        basic_time = time.time() - start_time
        performance_results["basic_ops_time"] = basic_time
        performance_results["steps_per_second"] = 100 / basic_time
        
        # Test 2: Memory usage (simplified)
        performance_results["memory_efficient"] = True  # Assume efficient since we use basic types
        
        # Test 3: Startup time
        startup_start = time.time()
        test_system = DependencyFreeTokamakSystem("SPARC")
        startup_time = time.time() - startup_start
        performance_results["startup_time"] = startup_time
        
        # Test 4: Large action processing
        large_action_start = time.time()
        for _ in range(10):
            large_action = [0.1] * 8
            test_system.step(large_action)
        large_action_time = time.time() - large_action_start
        performance_results["large_action_processing"] = large_action_time
        
        # Calculate performance score
        score = 100.0
        if basic_time > 1.0:  # If 100 steps take more than 1 second
            score -= 20
        if startup_time > 0.1:  # If startup takes more than 100ms
            score -= 10
        if large_action_time > 0.1:  # If processing is slow
            score -= 10
            
        performance_results["performance_score"] = max(0.0, score)
        
        self.logger.info(f"Performance test completed: {score:.1f}/100")
        return performance_results

class CodeQualityAnalyzer:
    """Code quality and maintainability analyzer."""
    
    def __init__(self):
        self.logger = logging.getLogger("CodeQualityAnalyzer")
        
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze code quality of a single file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Parse AST
            try:
                tree = ast.parse(content)
            except SyntaxError as e:
                return {"parse_error": str(e), "quality_score": 0.0}
            
            analyzer = CodeMetricsVisitor()
            analyzer.visit(tree)
            
            metrics = {
                "lines_of_code": len(content.split('\n')),
                "functions": analyzer.functions,
                "classes": analyzer.classes,
                "complexity": analyzer.complexity,
                "max_line_length": max(len(line) for line in content.split('\n')),
                "docstring_coverage": analyzer.docstring_coverage
            }
            
            # Calculate quality score
            score = 100.0
            
            # Penalize high complexity
            if metrics["complexity"] > 10:
                score -= min(20, (metrics["complexity"] - 10) * 2)
            
            # Penalize long lines
            if metrics["max_line_length"] > 100:
                score -= 10
                
            # Reward good documentation
            if metrics["docstring_coverage"] > 0.8:
                score += 5
            elif metrics["docstring_coverage"] < 0.5:
                score -= 10
            
            metrics["quality_score"] = max(0.0, score)
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error analyzing {file_path}: {e}")
            return {"error": str(e), "quality_score": 0.0}
    
    def analyze_directory(self, directory: str) -> Dict[str, Any]:
        """Analyze code quality for entire directory."""
        all_metrics = []
        total_score = 0.0
        file_count = 0
        
        for root, dirs, files in os.walk(directory):
            dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', '.pytest_cache']]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    metrics = self.analyze_file(file_path)
                    if "quality_score" in metrics:
                        all_metrics.append({"file": file_path, "metrics": metrics})
                        total_score += metrics["quality_score"]
                        file_count += 1
        
        average_score = total_score / file_count if file_count > 0 else 0.0
        
        return {
            "files_analyzed": file_count,
            "average_quality_score": average_score,
            "file_metrics": all_metrics,
            "total_lines": sum(m["metrics"].get("lines_of_code", 0) for m in all_metrics),
            "total_functions": sum(m["metrics"].get("functions", 0) for m in all_metrics),
            "total_classes": sum(m["metrics"].get("classes", 0) for m in all_metrics)
        }

class CodeMetricsVisitor(ast.NodeVisitor):
    """AST visitor to collect code metrics."""
    
    def __init__(self):
        self.functions = 0
        self.classes = 0
        self.complexity = 0
        self.docstrings = 0
        self.total_defs = 0
    
    def visit_FunctionDef(self, node):
        self.functions += 1
        self.total_defs += 1
        
        # Check for docstring
        if (node.body and isinstance(node.body[0], ast.Expr) and 
            isinstance(node.body[0].value, ast.Constant) and
            isinstance(node.body[0].value.value, str)):
            self.docstrings += 1
        
        # Calculate complexity (simplified)
        complexity_visitor = ComplexityVisitor()
        complexity_visitor.visit(node)
        self.complexity += complexity_visitor.complexity
        
        self.generic_visit(node)
    
    def visit_ClassDef(self, node):
        self.classes += 1
        self.total_defs += 1
        
        # Check for docstring
        if (node.body and isinstance(node.body[0], ast.Expr) and 
            isinstance(node.body[0].value, ast.Constant) and
            isinstance(node.body[0].value.value, str)):
            self.docstrings += 1
            
        self.generic_visit(node)
    
    @property
    def docstring_coverage(self):
        return self.docstrings / self.total_defs if self.total_defs > 0 else 0.0

class ComplexityVisitor(ast.NodeVisitor):
    """Calculate cyclomatic complexity."""
    
    def __init__(self):
        self.complexity = 1  # Base complexity
    
    def visit_If(self, node):
        self.complexity += 1
        self.generic_visit(node)
    
    def visit_While(self, node):
        self.complexity += 1
        self.generic_visit(node)
    
    def visit_For(self, node):
        self.complexity += 1
        self.generic_visit(node)
    
    def visit_ExceptHandler(self, node):
        self.complexity += 1
        self.generic_visit(node)

class ComprehensiveQualityGate:
    """Comprehensive quality gate system."""
    
    def __init__(self):
        self.logger = logging.getLogger("ComprehensiveQualityGate")
        self.security_scanner = SecurityScanner()
        self.performance_tester = PerformanceTester()
        self.code_analyzer = CodeQualityAnalyzer()
        
    def run_all_checks(self, project_dir: str = ".") -> Dict[str, Any]:
        """Run all quality gate checks."""
        self.logger.info("Starting comprehensive quality gate checks...")
        
        results = {
            "timestamp": time.time(),
            "project_dir": project_dir,
            "checks": {}
        }
        
        # 1. Security Scan
        self.logger.info("Running security scan...")
        security_issues = self.security_scanner.scan_directory(project_dir)
        security_score = max(0, 100 - len(security_issues) * 10)  # Penalize issues
        
        results["checks"]["security"] = {
            "score": security_score,
            "issues": [asdict(issue) for issue in security_issues],
            "critical_issues": len([i for i in security_issues if i.severity == "CRITICAL"]),
            "high_issues": len([i for i in security_issues if i.severity == "HIGH"]),
            "passed": len(security_issues) == 0
        }
        
        # 2. Performance Tests
        self.logger.info("Running performance tests...")
        try:
            perf_results = self.performance_tester.test_system_performance()
            results["checks"]["performance"] = {
                "score": perf_results.get("performance_score", 0.0),
                "metrics": perf_results,
                "passed": perf_results.get("performance_score", 0.0) >= 70.0
            }
        except Exception as e:
            self.logger.error(f"Performance test failed: {e}")
            results["checks"]["performance"] = {
                "score": 0.0,
                "error": str(e),
                "passed": False
            }
        
        # 3. Code Quality Analysis  
        self.logger.info("Running code quality analysis...")
        try:
            quality_results = self.code_analyzer.analyze_directory(project_dir)
            results["checks"]["code_quality"] = {
                "score": quality_results.get("average_quality_score", 0.0),
                "metrics": quality_results,
                "passed": quality_results.get("average_quality_score", 0.0) >= 70.0
            }
        except Exception as e:
            self.logger.error(f"Code quality analysis failed: {e}")
            results["checks"]["code_quality"] = {
                "score": 0.0,
                "error": str(e),
                "passed": False
            }
        
        # 4. Basic Functionality Tests
        self.logger.info("Running functionality tests...")
        functionality_passed = self._test_basic_functionality()
        results["checks"]["functionality"] = {
            "score": 100.0 if functionality_passed else 0.0,
            "passed": functionality_passed
        }
        
        # 5. Compliance Checks
        self.logger.info("Running compliance checks...")
        compliance_score = self._check_compliance(project_dir)
        results["checks"]["compliance"] = {
            "score": compliance_score,
            "passed": compliance_score >= 80.0
        }
        
        # Calculate overall quality score
        check_scores = [check["score"] for check in results["checks"].values() if "score" in check]
        overall_score = sum(check_scores) / len(check_scores) if check_scores else 0.0
        results["overall_score"] = overall_score
        results["passed"] = all(check.get("passed", False) for check in results["checks"].values())
        
        # Generate report
        self._generate_quality_report(results)
        
        return results
    
    def _test_basic_functionality(self) -> bool:
        """Test basic system functionality."""
        try:
            # Test dependency-free core
            sys.path.insert(0, os.path.dirname(__file__))
            from dependency_free_core import DependencyFreeTokamakSystem
            
            system = DependencyFreeTokamakSystem("ITER")
            obs, info = system.reset()
            
            # Test a few steps
            for i in range(5):
                action = [0.1] * 8
                obs, reward, done, truncated, info = system.step(action)
                
                # Basic sanity checks
                if not isinstance(obs, list) or len(obs) != 8:
                    return False
                if not isinstance(reward, (int, float)):
                    return False
                    
            return True
            
        except Exception as e:
            self.logger.error(f"Functionality test failed: {e}")
            return False
    
    def _check_compliance(self, project_dir: str) -> float:
        """Check project compliance (licensing, documentation, etc.)."""
        score = 100.0
        
        # Check for LICENSE file
        license_files = ["LICENSE", "LICENSE.txt", "LICENSE.md"]
        has_license = any(os.path.exists(os.path.join(project_dir, f)) for f in license_files)
        if not has_license:
            score -= 20
        
        # Check for README
        readme_files = ["README.md", "README.txt", "README.rst"]
        has_readme = any(os.path.exists(os.path.join(project_dir, f)) for f in readme_files)
        if not has_readme:
            score -= 20
        
        # Check for pyproject.toml or setup.py
        has_setup = any(os.path.exists(os.path.join(project_dir, f)) 
                       for f in ["pyproject.toml", "setup.py"])
        if not has_setup:
            score -= 15
        
        # Check for requirements or dependencies
        has_deps = any(os.path.exists(os.path.join(project_dir, f)) 
                      for f in ["requirements.txt", "pyproject.toml", "Pipfile"])
        if not has_deps:
            score -= 10
        
        return max(0.0, score)
    
    def _generate_quality_report(self, results: Dict[str, Any]):
        """Generate quality gate report."""
        report = []
        report.append("=" * 60)
        report.append("TOKAMAK-RL COMPREHENSIVE QUALITY GATE REPORT")
        report.append("=" * 60)
        report.append(f"Overall Score: {results['overall_score']:.1f}/100")
        report.append(f"Overall Status: {'✅ PASSED' if results['passed'] else '❌ FAILED'}")
        report.append("")
        
        # Individual check results
        for check_name, check_result in results["checks"].items():
            status = "✅ PASSED" if check_result.get("passed", False) else "❌ FAILED"
            score = check_result.get("score", 0.0)
            report.append(f"{check_name.upper()}: {score:.1f}/100 {status}")
            
            if check_name == "security" and "issues" in check_result:
                issues = check_result["issues"]
                if issues:
                    report.append(f"  Security Issues: {len(issues)} found")
                    for issue in issues[:3]:  # Show first 3 issues
                        report.append(f"    - {issue['severity']}: {issue['description']}")
            
            if "error" in check_result:
                report.append(f"  Error: {check_result['error']}")
        
        report.append("")
        report.append("Quality Gate Summary:")
        if results["passed"]:
            report.append("🎉 ALL QUALITY GATES PASSED - READY FOR PRODUCTION")
        else:
            report.append("⚠️ QUALITY GATES FAILED - REVIEW AND FIX ISSUES")
        
        # Write report to file
        report_content = "\n".join(report)
        with open("quality_gate_report.txt", "w") as f:
            f.write(report_content)
        
        # Also print to console
        print("\n" + report_content)

def main():
    """Run comprehensive quality gates."""
    print("🛡️ COMPREHENSIVE QUALITY GATES VALIDATION")
    print("="*60)
    
    quality_gate = ComprehensiveQualityGate()
    results = quality_gate.run_all_checks(".")
    
    return results["passed"]

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)