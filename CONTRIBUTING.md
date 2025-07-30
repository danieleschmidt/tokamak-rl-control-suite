# Contributing to Tokamak RL Control Suite

We welcome contributions from the fusion energy and machine learning communities! This guide outlines how to contribute effectively to this project.

## 🚀 Quick Start

1. **Fork** the repository on GitHub
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/yourusername/tokamak-rl-control-suite.git
   cd tokamak-rl-control-suite
   ```
3. **Install** in development mode:
   ```bash
   pip install -e ".[dev]"
   ```
4. **Create** a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## 🔬 Development Setup

### Prerequisites
- Python 3.8+
- Git
- Understanding of reinforcement learning and/or plasma physics

### Installation
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install with development dependencies
pip install -e ".[dev,docs]"

# Install pre-commit hooks
pre-commit install
```

## 🧪 Testing

Run the test suite before submitting changes:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/tokamak_rl --cov-report=html

# Run specific test files
pytest tests/test_environment.py
```

## 🔍 Code Quality

We maintain high code quality standards:

```bash
# Format code
black src/ tests/

# Lint code  
ruff check src/ tests/

# Type checking
mypy src/tokamak_rl
```

Pre-commit hooks automatically run these checks.

## 📝 Pull Request Process

1. **Ensure tests pass** and coverage is maintained
2. **Update documentation** for any API changes
3. **Write clear commit messages** following conventional commits
4. **Create detailed PR description** explaining:
   - What changes were made
   - Why they were necessary
   - How to test the changes
   - Any breaking changes

### PR Template
```markdown
## Summary
Brief description of changes

## Changes Made
- [ ] Feature implementation
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Breaking changes noted

## Testing
How were these changes tested?

## Related Issues
Fixes #123
```

## 🏗️ Development Guidelines

### Code Style
- Follow PEP 8 with 88-character line limits
- Use type hints for all public APIs
- Write docstrings for all public functions/classes
- Prefer composition over inheritance

### Physics Accuracy
- Cite relevant papers for physics implementations
- Include units in variable names/docstrings
- Validate against established tokamak parameters
- Consider realistic engineering constraints

### Safety Considerations
- Never compromise plasma safety for performance
- Include proper error handling for disruption scenarios
- Document safety assumptions clearly
- Test edge cases thoroughly

## 📚 Documentation

### Code Documentation
- Use Google-style docstrings
- Include parameter types and units
- Provide usage examples
- Document any physics assumptions

### User Documentation
- Update README for new features
- Add examples to `examples/` directory
- Update API documentation
- Include performance benchmarks

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Environment details** (Python version, OS, dependencies)
2. **Minimal reproducible example**
3. **Expected vs actual behavior**
4. **Error messages/stack traces**
5. **Physics context** if relevant

Use the bug report template in GitHub Issues.

## 💡 Feature Requests

For new features, please:

1. **Check existing issues** for similar requests
2. **Describe the use case** and motivation
3. **Outline proposed implementation** if possible
4. **Consider physics/safety implications**
5. **Discuss with maintainers** before major changes

## 🔬 Research Contributions

We especially welcome:

- **New RL algorithms** adapted for plasma control
- **Physics model improvements** with proper validation
- **Safety system enhancements**
- **Benchmarking studies** against real tokamak data
- **Sim-to-real transfer** techniques

Please include relevant citations and validation data.

## 📞 Communication

- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: General questions, research ideas  
- **Email**: daniel@terragonlabs.io for sensitive matters

## 🏆 Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes for significant contributions
- Academic papers (with consent)
- Conference presentations

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🙏 Questions?

Don't hesitate to ask questions! We're here to help make your contribution successful.

Thank you for helping advance fusion energy research! 🔥