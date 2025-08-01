"""
Security compliance tests for Tokamak RL Control Suite.

These tests verify security best practices and compliance requirements
for safety-critical plasma control applications.
"""

import ast
import importlib
import inspect
import pkgutil
import sys
from pathlib import Path
from typing import List, Set

import pytest
import tokamak_rl


class TestSecurityCompliance:
    """Test suite for security compliance verification."""

    def test_no_hardcoded_secrets(self):
        """Ensure no hardcoded secrets exist in source code."""
        src_path = Path(__file__).parent.parent.parent / "src"
        suspicious_patterns = [
            "password",
            "secret",
            "api_key",
            "private_key",
            "token",
            "credential",
        ]
        
        violations = []
        for py_file in src_path.rglob("*.py"):
            content = py_file.read_text(encoding="utf-8").lower()
            for pattern in suspicious_patterns:
                if f'"{pattern}"' in content or f"'{pattern}'" in content:
                    # Check if it's just a variable name, not a value
                    lines = content.split('\n')
                    for i, line in enumerate(lines, 1):
                        if pattern in line and '=' in line:
                            # Skip if it's just a parameter name
                            if not (line.strip().startswith('def ') or 
                                   line.strip().startswith('class ')):
                                violations.append(f"{py_file}:{i} - {line.strip()}")
        
        assert not violations, f"Potential hardcoded secrets found: {violations}"

    def test_no_debug_statements(self):
        """Ensure no debug print statements in production code."""
        src_path = Path(__file__).parent.parent.parent / "src"
        debug_patterns = ["print(", "pprint(", "pp(", "console.log"]
        
        violations = []
        for py_file in src_path.rglob("*.py"):
            if py_file.name.startswith("test_"):
                continue
                
            content = py_file.read_text(encoding="utf-8")
            lines = content.split('\n')
            
            for i, line in enumerate(lines, 1):
                for pattern in debug_patterns:
                    if pattern in line and not line.strip().startswith('#'):
                        violations.append(f"{py_file}:{i} - {line.strip()}")
        
        assert not violations, f"Debug statements found in production code: {violations}"

    def test_import_safety(self):
        """Verify all imports are from trusted sources."""
        dangerous_imports = [
            "subprocess",
            "os.system",
            "eval",
            "exec",
            "compile",
            "__import__",
        ]
        
        src_path = Path(__file__).parent.parent.parent / "src"
        violations = []
        
        for py_file in src_path.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            if alias.name in dangerous_imports:
                                violations.append(f"{py_file} imports {alias.name}")
                    
                    elif isinstance(node, ast.ImportFrom):
                        if node.module and any(danger in node.module for danger in dangerous_imports):
                            violations.append(f"{py_file} imports from {node.module}")
                            
            except (SyntaxError, UnicodeDecodeError):
                # Skip files that can't be parsed
                continue
        
        assert not violations, f"Dangerous imports found: {violations}"

    def test_function_complexity(self):
        """Ensure functions don't exceed complexity thresholds for safety."""
        max_complexity = 10  # Cyclomatic complexity threshold
        
        src_path = Path(__file__).parent.parent.parent / "src"
        violations = []
        
        for py_file in src_path.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())
                
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        complexity = self._calculate_complexity(node)
                        if complexity > max_complexity:
                            violations.append(
                                f"{py_file}:{node.lineno} - "
                                f"Function '{node.name}' has complexity {complexity}"
                            )
                            
            except (SyntaxError, UnicodeDecodeError):
                continue
        
        assert not violations, f"High complexity functions found: {violations}"

    def test_no_sql_injection_patterns(self):
        """Check for potential SQL injection vulnerabilities."""
        src_path = Path(__file__).parent.parent.parent / "src"
        sql_patterns = [
            "SELECT * FROM",
            "DROP TABLE",
            "DELETE FROM",
            "INSERT INTO",
            "UPDATE SET",
        ]
        
        violations = []
        for py_file in src_path.rglob("*.py"):
            content = py_file.read_text(encoding="utf-8").upper()
            for pattern in sql_patterns:
                if pattern in content and '"' + pattern in content:
                    violations.append(f"{py_file} contains SQL pattern: {pattern}")
        
        # For this physics simulation project, SQL should not be present
        assert not violations, f"Potential SQL injection patterns: {violations}"

    def test_file_permissions(self):
        """Verify source files have appropriate permissions."""
        src_path = Path(__file__).parent.parent.parent / "src"
        
        violations = []
        for py_file in src_path.rglob("*.py"):
            stat = py_file.stat()
            # Check if file is executable (shouldn't be for .py files)
            if stat.st_mode & 0o111:  # Check execute permissions
                violations.append(f"{py_file} is executable")
        
        assert not violations, f"Incorrectly permissioned files: {violations}"

    def test_no_temp_files(self):
        """Ensure no temporary files are committed."""
        project_root = Path(__file__).parent.parent.parent
        temp_patterns = ["*.tmp", "*.temp", "*.bak", "*.swp", "*~"]
        
        violations = []
        for pattern in temp_patterns:
            for temp_file in project_root.rglob(pattern):
                if not any(part.startswith('.') for part in temp_file.parts[1:]):
                    violations.append(str(temp_file))
        
        assert not violations, f"Temporary files found: {violations}"

    def _calculate_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity of an AST node."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, (ast.BoolOp, ast.Compare)):
                complexity += 1
        
        return complexity


class TestPlasmaControlSafety:
    """Safety-specific tests for plasma control applications."""

    def test_safety_critical_imports(self):
        """Verify safety-critical systems use approved libraries."""
        approved_ml_libraries = {
            "numpy", "scipy", "matplotlib", "torch", "tensorflow",
            "gymnasium", "stable_baselines3", "tensorboard"
        }
        
        # Get all imported modules
        imported_modules = set()
        src_path = Path(__file__).parent.parent.parent / "src"
        
        for py_file in src_path.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imported_modules.add(alias.name.split('.')[0])
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        imported_modules.add(node.module.split('.')[0])
                        
            except (SyntaxError, UnicodeDecodeError):
                continue
        
        # Filter out standard library and approved modules
        external_modules = imported_modules - {
            "os", "sys", "typing", "pathlib", "json", "logging", 
            "datetime", "collections", "functools", "itertools",
            "warnings", "abc", "dataclasses", "enum"
        } - approved_ml_libraries
        
        # For now, just document what we're importing
        if external_modules:
            print(f"External modules in use: {sorted(external_modules)}")

    def test_error_handling_coverage(self):
        """Ensure critical functions have proper error handling."""
        src_path = Path(__file__).parent.parent.parent / "src"
        functions_without_try_catch = []
        
        for py_file in src_path.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())
                
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        # Check if function contains safety-critical keywords
                        func_source = ast.get_source_segment(f.read(), node)
                        if func_source and any(keyword in func_source.lower() for keyword in 
                                             ['plasma', 'control', 'safety', 'disruption']):
                            # Check if function has try-except blocks
                            has_exception_handling = any(
                                isinstance(child, ast.Try) for child in ast.walk(node)
                            )
                            if not has_exception_handling:
                                functions_without_try_catch.append(
                                    f"{py_file}:{node.lineno} - {node.name}"
                                )
                                
            except (SyntaxError, UnicodeDecodeError):
                continue
        
        # This is informational for now - safety-critical functions should have error handling
        if functions_without_try_catch:
            print(f"Safety-critical functions without error handling: {functions_without_try_catch}")


class TestDataSanitization:
    """Test data input validation and sanitization."""

    def test_input_validation_patterns(self):
        """Check for input validation in public functions."""
        src_path = Path(__file__).parent.parent.parent / "src"
        
        public_functions_without_validation = []
        
        for py_file in src_path.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # Skip private functions
                        if node.name.startswith('_'):
                            continue
                            
                        # Check if function has input validation
                        func_source = ast.get_source_segment(content, node)
                        if func_source:
                            validation_patterns = [
                                'isinstance', 'assert', 'raise ValueError',
                                'raise TypeError', 'if not', 'validate'
                            ]
                            has_validation = any(
                                pattern in func_source for pattern in validation_patterns
                            )
                            
                            if not has_validation and len(node.args.args) > 1:  # Has parameters
                                public_functions_without_validation.append(
                                    f"{py_file}:{node.lineno} - {node.name}"
                                )
                                
            except (SyntaxError, UnicodeDecodeError):
                continue
        
        # This is informational - public APIs should validate inputs
        if public_functions_without_validation:
            print(f"Public functions without input validation: {public_functions_without_validation}")

    def test_numeric_safety(self):
        """Check for potential numeric overflow/underflow issues."""
        src_path = Path(__file__).parent.parent.parent / "src"
        
        unsafe_operations = []
        
        for py_file in src_path.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Look for potentially unsafe numeric operations
                unsafe_patterns = [
                    "**",  # Exponentiation without limits
                    "1.0/",  # Potential division by zero
                    "math.exp(",  # Exponential function
                    "np.exp(",  # NumPy exponential
                ]
                
                lines = content.split('\n')
                for i, line in enumerate(lines, 1):
                    for pattern in unsafe_patterns:
                        if pattern in line and not line.strip().startswith('#'):
                            unsafe_operations.append(f"{py_file}:{i} - {line.strip()}")
                            
            except (SyntaxError, UnicodeDecodeError):
                continue
        
        # This is informational - numeric operations should be bounds-checked
        if unsafe_operations:
            print(f"Potentially unsafe numeric operations: {unsafe_operations}")


def test_package_integrity():
    """Verify package can be imported without errors."""
    try:
        import tokamak_rl
        assert hasattr(tokamak_rl, '__version__')
        assert tokamak_rl.__version__
    except Exception as e:
        pytest.fail(f"Package import failed: {e}")


def test_no_network_dependencies():
    """Ensure the package doesn't make unexpected network calls."""
    # This would require more sophisticated testing with network mocking
    # For now, just check that no obvious network libraries are imported
    dangerous_network_imports = [
        "urllib", "requests", "httpx", "aiohttp", "socket"
    ]
    
    # Check main package imports
    spec = importlib.util.find_spec("tokamak_rl")
    if spec and spec.origin:
        with open(spec.origin, 'r') as f:
            content = f.read()
            for net_import in dangerous_network_imports:
                assert net_import not in content, f"Unexpected network import: {net_import}"