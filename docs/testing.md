# 🧪 Testing Guide

This guide covers the comprehensive testing framework for tokamak-rl-control-suite, including unit tests, integration tests, performance benchmarks, and safety validation.

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── pytest.ini              # Pytest configuration
├── unit/                    # Fast, isolated unit tests
├── integration/             # Component interaction tests
├── performance/             # Performance benchmarks
├── security/                # Security and safety tests
├── e2e/                     # End-to-end workflow tests
└── fixtures/                # Test data and utilities
```

## Test Categories

### 🔬 Unit Tests (`tests/unit/`)

**Purpose**: Test individual components in isolation
**Speed**: Fast (< 1s per test)
**Dependencies**: None (fully mocked)

**Run Command**:
```bash
pytest tests/unit/ -v
make test-unit
```

**Examples**:
- Physics solver mathematical correctness
- Environment observation/action space validation
- Safety constraint enforcement
- Reward function calculations

### 🔗 Integration Tests (`tests/integration/`)

**Purpose**: Test component interactions and data flow
**Speed**: Medium (1-10s per test)
**Dependencies**: Some external dependencies

**Run Command**:
```bash
pytest tests/integration/ -v
make test-integration
```

**Examples**:
- Environment-agent interaction
- Physics solver + safety system integration
- Data pipeline end-to-end
- Configuration loading and validation

### ⚡ Performance Tests (`tests/performance/`)

**Purpose**: Measure performance and detect regressions
**Speed**: Variable (marked with `@pytest.mark.performance`)
**Dependencies**: Full system

**Run Command**:
```bash
pytest tests/performance/ -v --benchmark-only
make test-performance
```

**Benchmarks**:
- Environment step execution time
- Physics solver convergence speed
- Memory usage patterns
- GPU utilization efficiency

### 🛡️ Security Tests (`tests/security/`)

**Purpose**: Validate safety-critical systems
**Speed**: Medium (extensive validation)
**Dependencies**: Safety systems

**Run Command**:
```bash
pytest tests/security/ -v
make test-security
```

**Examples**:
- Disruption prediction accuracy
- Safety constraint enforcement
- Emergency fallback systems
- Input validation and sanitization

### 🌐 End-to-End Tests (`tests/e2e/`)

**Purpose**: Complete workflow validation
**Speed**: Slow (marked with `@pytest.mark.slow`)
**Dependencies**: Full system

**Run Command**:
```bash
pytest tests/e2e/ -v
```

**Workflows**:
- Complete training pipelines
- Model evaluation and benchmarking
- Real-time deployment simulation
- Multi-tokamak compatibility

## Test Markers

Use pytest markers to categorize and filter tests:

```python
@pytest.mark.unit          # Unit tests
@pytest.mark.integration   # Integration tests
@pytest.mark.physics       # Physics validation
@pytest.mark.performance   # Performance benchmarks
@pytest.mark.safety        # Safety-critical tests
@pytest.mark.slow          # Long-running tests
@pytest.mark.gpu           # Requires GPU
@pytest.mark.experimental  # Experimental features
@pytest.mark.regression    # Regression tests
```

**Filter Examples**:
```bash
# Run only fast tests
pytest -m "not slow"

# Run only physics tests
pytest -m "physics"

# Run unit and integration tests
pytest -m "unit or integration"

# Skip GPU tests if no GPU available
pytest -m "not gpu"
```

## Test Fixtures

### Common Fixtures (from `conftest.py`)

```python
# Basic configurations
basic_tokamak_config        # Standard ITER configuration
sample_observation_space    # Environment observation space
sample_action_space         # Environment action space

# Mock objects
mock_plasma_state          # Mock plasma state data
mock_physics_solver        # Mock physics solver
mock_safety_system         # Mock safety system

# Test data
physics_test_tolerances    # Physics validation tolerances
benchmark_config           # Performance benchmark settings
integration_test_scenarios # Integration test cases
```

### Using Fixtures

```python
def test_environment_initialization(basic_tokamak_config):
    """Test environment initializes correctly."""
    # env = TokamakEnvironment(basic_tokamak_config)
    assert basic_tokamak_config["major_radius"] == 6.2

def test_physics_solver(mock_physics_solver, physics_test_tolerances):
    """Test physics solver with validation."""
    result = mock_physics_solver.solve_equilibrium()
    assert result["converged"] is True
    # Additional physics validation...
```

## Running Tests

### Basic Commands

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_physics_solver.py

# Run specific test function
pytest tests/unit/test_physics_solver.py::test_equilibrium_solving

# Run with verbose output
pytest -v

# Run in parallel (faster)
pytest -n auto
```

### Using Makefile

```bash
# Run all tests with coverage
make test

# Run fast tests only
make test-fast

# Run specific test categories
make test-unit
make test-integration
make test-performance
make test-security
```

### Development Workflow

```bash
# Quick test during development
pytest -x --ff  # Fail fast, failed first

# Test with coverage
make test

# Run performance benchmarks
make benchmark

# Run all quality checks
make qa  # lint + typecheck + test + security
```

## Performance Benchmarking

### Benchmark Configuration

Performance targets are defined in `benchmark_config` fixture:

```python
{
    "num_episodes": 10,
    "max_steps_per_episode": 1000,
    "time_limit_seconds": 60,
    "memory_limit_mb": 512,
    "acceptable_step_time_ms": 50,
}
```

### Running Benchmarks

```bash
# Run performance tests
pytest tests/performance/ --benchmark-only

# Generate benchmark report
pytest tests/performance/ --benchmark-json=benchmark.json

# Compare with baseline
pytest-benchmark compare baseline.json benchmark.json
```

### Performance Metrics

- **Environment step time**: < 50ms per step
- **Physics solver convergence**: < 100ms per solve
- **Memory usage**: < 512MB for standard training
- **Training throughput**: > 100 steps/second

## Physics Validation

### Validation Tests

Physics tests validate mathematical correctness:

```python
@pytest.mark.physics
def test_pressure_profile_consistency(physics_test_tolerances):
    """Test pressure profile physical consistency."""
    # Pressure should decrease monotonically outward
    # Check boundary conditions
    # Validate gradient magnitudes
```

### Tolerance Configuration

```python
physics_test_tolerances = {
    "equilibrium_residual": 1e-6,
    "shape_error": 0.1,  # cm
    "q_profile_error": 0.05,
    "pressure_profile_error": 0.02,
    "energy_conservation": 1e-4,
}
```

### Experimental Data Validation

Tests against experimental tokamak data:

```python
# Standard test cases from real tokamaks
STANDARD_TEST_CASES = {
    "iter_baseline": {...},
    "sparc_baseline": {...},
    "nstx_baseline": {...},
}
```

## Safety Testing

### Safety Test Categories

1. **Disruption Prevention**: Test disruption prediction and avoidance
2. **Constraint Enforcement**: Validate safety constraint enforcement
3. **Emergency Response**: Test emergency fallback systems
4. **Input Validation**: Ensure robust input handling

### Critical Safety Tests

```python
@pytest.mark.safety
def test_disruption_avoidance_scenarios(integration_test_scenarios):
    """Test disruption avoidance in various scenarios."""
    for scenario in integration_test_scenarios:
        if scenario["name"] == "disruption_avoidance_scenario":
            # Run scenario and verify no disruption occurs
            assert scenario["expected_disruption"] is False
```

### Safety Metrics

- **Disruption rate**: < 0.1% in all test scenarios
- **Response time**: < 100ms for emergency actions
- **Safety coverage**: 100% of constraint violations detected
- **False positive rate**: < 1% for disruption predictions

## Test Data Management

### Generated Test Data

The `TestDataGenerator` class provides consistent test data:

```python
generator = TestDataGenerator(seed=42)  # Reproducible

# Generate synthetic equilibrium data
eq_data = generator.generate_equilibrium_data(grid_size=64)

# Generate control trajectories
control_data = generator.generate_control_trajectory(num_steps=1000)

# Generate plasma states
plasma_states = generator.generate_plasma_states(num_states=100)

# Generate disruption scenarios
disruption_scenarios = generator.generate_disruption_scenarios()
```

### Benchmark Datasets

Standardized datasets for performance testing:

```python
# Different sizes for scalability testing
small_dataset = BenchmarkDatasets.small_dataset()    # Quick tests
medium_dataset = BenchmarkDatasets.medium_dataset()  # Standard benchmarks
large_dataset = BenchmarkDatasets.large_dataset()    # Stress tests
```

## Continuous Integration

### GitHub Actions Integration

Tests run automatically on:
- Push to main branch
- Pull request creation
- Scheduled nightly runs

### Test Matrix

Tests run across multiple configurations:
- Python versions: 3.8, 3.9, 3.10, 3.11, 3.12
- Operating systems: Ubuntu, macOS, Windows
- Dependencies: Minimal vs. full installation

### Coverage Requirements

- **Minimum coverage**: 80% overall
- **Critical components**: 95% coverage required
- **New code**: 90% coverage required

## Test Development Guidelines

### Writing Good Tests

1. **Test One Thing**: Each test should verify one specific behavior
2. **Clear Names**: Use descriptive test function names
3. **Arrange-Act-Assert**: Structure tests clearly
4. **Use Fixtures**: Leverage shared fixtures for consistency
5. **Mock External Dependencies**: Keep tests isolated

### Example Test Structure

```python
def test_specific_behavior(fixture_name):
    """Test description explaining what is being tested."""
    # Arrange: Set up test data and mocks
    test_input = create_test_input()
    expected_output = calculate_expected_output()
    
    # Act: Execute the functionality being tested
    actual_output = function_under_test(test_input)
    
    # Assert: Verify the results
    assert actual_output == expected_output
    assert additional_condition_is_met()
```

### Test Data Best Practices

1. **Use Fixtures**: Define reusable test data in `conftest.py`
2. **Mock External Systems**: Don't depend on external services
3. **Parameterize Tests**: Test multiple scenarios efficiently
4. **Reproducible**: Use fixed random seeds for consistent results

```python
@pytest.mark.parametrize("tokamak_config", [
    STANDARD_TEST_CASES["iter_baseline"],
    STANDARD_TEST_CASES["sparc_baseline"],
    STANDARD_TEST_CASES["nstx_baseline"],
])
def test_cross_tokamak_compatibility(tokamak_config):
    """Test functionality across different tokamak configurations."""
    # Test implementation
```

## Debugging Tests

### Common Issues

1. **Flaky Tests**: Use fixed random seeds, avoid timing dependencies
2. **Slow Tests**: Profile and optimize, consider mocking
3. **Memory Issues**: Clean up resources, monitor memory usage
4. **Physics Validation Failures**: Check tolerances and assumptions

### Debugging Commands

```bash
# Run single test with full output
pytest tests/unit/test_file.py::test_function -v -s

# Drop into debugger on failure
pytest --pdb

# Show local variables on failure
pytest --tb=long

# Run with coverage to find untested code
pytest --cov=src --cov-report=html
```

### Test Profiling

```bash
# Profile test execution time
pytest --durations=10

# Memory profiling
pytest --memray

# Line profiling for slow tests
pytest --profile
```

## Contributing Tests

### Test Review Checklist

- [ ] Tests cover all new functionality
- [ ] Tests follow naming conventions
- [ ] Tests are properly categorized with markers
- [ ] Mock external dependencies appropriately
- [ ] Include both positive and negative test cases
- [ ] Performance tests for performance-critical code
- [ ] Safety tests for safety-critical features
- [ ] Documentation includes testing information

### Adding New Test Categories

1. Create new directory under `tests/`
2. Add `__init__.py` file
3. Define category-specific fixtures in `conftest.py`
4. Add marker to `pytest_configure()`
5. Update Makefile with new test command
6. Document category in this guide

---

For questions about testing, check the [FAQ](faq/technical.md) or file an issue on GitHub.