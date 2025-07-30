# Development Setup Guide

## 🚀 Quick Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/tokamak-rl-control-suite.git
cd tokamak-rl-control-suite

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev,docs]"

# Setup pre-commit hooks
pre-commit install

# Run tests to verify setup
pytest
```

## 📁 Project Structure

```
tokamak-rl-control-suite/
├── src/tokamak_rl/          # Main package source
│   ├── __init__.py          # Package initialization
│   └── environment.py       # Environment factory and base classes
├── tests/                   # Test suite
│   ├── __init__.py
│   └── test_environment.py  # Environment tests
├── docs/                    # Documentation (future)
├── examples/                # Usage examples (future)
├── pyproject.toml          # Project configuration
├── README.md               # Project overview
├── CONTRIBUTING.md         # Contribution guidelines
├── CODE_OF_CONDUCT.md      # Community standards
├── SECURITY.md             # Security policy
├── DEVELOPMENT.md          # This file
├── .editorconfig           # Editor configuration
├── .pre-commit-config.yaml # Pre-commit hooks
└── .gitignore             # Git ignore patterns
```

## 🛠️ Development Tools

### Code Quality
- **Black**: Code formatting (88 char line limit)
- **Ruff**: Fast Python linter and code analysis
- **MyPy**: Static type checking
- **Pre-commit**: Automated quality checks

### Testing
- **Pytest**: Test framework with coverage reporting
- **Pytest-cov**: Coverage analysis

### Documentation
- **Sphinx**: Documentation generation (planned)
- **MyST**: Markdown support for docs

## 🔧 Development Workflow

### 1. Setting Up Development Environment

```bash
# Install development dependencies
pip install -e ".[dev]"

# Verify installation
python -c "import tokamak_rl; print(tokamak_rl.__version__)"

# Install pre-commit hooks
pre-commit install

# Test pre-commit setup
pre-commit run --all-files
```

### 2. Code Quality Checks

```bash
# Format code
black src/ tests/

# Check code style
ruff check src/ tests/

# Type checking
mypy src/tokamak_rl/

# Run all quality checks
pre-commit run --all-files
```

### 3. Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/tokamak_rl --cov-report=html

# Run specific test file
pytest tests/test_environment.py

# Run with verbose output
pytest -v
```

### 4. Documentation

```bash
# Install docs dependencies
pip install -e ".[docs]"

# Build documentation (when implemented)
# sphinx-build -b html docs/ docs/_build/
```

## 🧪 Testing Strategy

### Test Categories
1. **Unit Tests**: Individual component functionality
2. **Integration Tests**: Component interaction (planned)
3. **Physics Tests**: Plasma simulation accuracy (planned)
4. **Safety Tests**: Control safety validation (planned)

### Current Test Coverage
- ✅ Environment factory interface
- ✅ Base environment class structure
- ✅ Package imports and version
- 🔄 Physics simulation tests (planned)
- 🔄 RL agent integration tests (planned)

## 🔬 Physics Development

### Key Physics Components (Planned)
- **Grad-Shafranov Solver**: MHD equilibrium calculation
- **Plasma Shape Analysis**: Boundary reconstruction
- **Safety Factor Calculation**: q-profile computation
- **Disruption Prediction**: ML-based early warning

### Physics Validation
- Compare against established tokamak databases
- Validate equilibrium solutions with EFIT
- Cross-check safety factors with experimental data
- Verify disruption predictions against historical events

## 🤖 RL Development

### Supported Algorithms (Planned)
- **SAC**: Soft Actor-Critic for continuous control
- **Dreamer**: Model-based RL for sample efficiency
- **PPO**: Proximal Policy Optimization
- **TD3**: Twin Delayed Deep Deterministic Policy Gradient

### Environment Design
- **Observation Space**: 45D physics state vector
- **Action Space**: 8D continuous control actions
- **Reward Function**: Multi-objective plasma optimization
- **Safety Constraints**: Hard limits and disruption prevention

## 🐛 Debugging Tips

### Common Issues
1. **Import Errors**: Ensure installed with `pip install -e .`
2. **Pre-commit Failures**: Run `black` and `ruff` manually first
3. **Test Failures**: Check Python path and dependencies
4. **Physics Errors**: Validate input parameters against realistic ranges

### Debug Environment
```bash
# Enable debug logging
export TOKAMAK_RL_DEBUG=1

# Run with verbose pytest output
pytest -v -s

# Check package installation
pip show tokamak-rl-control-suite
```

## 📊 Performance Considerations

### Computational Requirements
- **CPU**: Multi-core for parallel training
- **Memory**: 8GB+ for large plasma simulations  
- **GPU**: Optional for neural network training
- **Storage**: Minimal for code, substantial for training data

### Optimization Tips
- Use vectorized NumPy operations for physics calculations
- Implement JIT compilation for hot physics loops
- Consider MPI parallelization for distributed training
- Profile physics solvers for computational bottlenecks

## 🚀 Contributing Workflow

1. **Create feature branch**: `git checkout -b feature/your-feature`
2. **Implement changes**: Follow coding standards
3. **Add tests**: Ensure new functionality is tested
4. **Update docs**: Document any API changes
5. **Run quality checks**: `pre-commit run --all-files`
6. **Submit PR**: Include detailed description and tests

## 📋 Release Process (Future)

1. **Version Bump**: Update `__init__.py` version
2. **Update Changelog**: Document all changes
3. **Tag Release**: `git tag -a v0.1.0 -m "Release v0.1.0"`
4. **Build Package**: `python -m build`
5. **Upload to PyPI**: `twine upload dist/*`

## 🔗 Useful Resources

- [Fusion Physics Primer](https://www.iter.org/sci/plasmabasics)
- [Gymnasium RL Framework](https://gymnasium.farama.org/)
- [Stable-Baselines3 Algorithms](https://stable-baselines3.readthedocs.io/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Black Code Style](https://black.readthedocs.io/)

## 📞 Getting Help

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Development questions
- **Email**: daniel@terragonlabs.io for direct contact

Happy coding! 🔥⚛️