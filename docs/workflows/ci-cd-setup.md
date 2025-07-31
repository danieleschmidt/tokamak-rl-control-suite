# CI/CD Workflow Setup Guide

This document provides GitHub Actions workflow templates for the tokamak-rl-control-suite project.

## Required Workflows

### 1. Main CI/CD Pipeline (`.github/workflows/ci.yml`)

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.8', '3.9', '3.10', '3.11', '3.12']
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
    
    - name: Lint with ruff
      run: ruff check src/ tests/
    
    - name: Format check with black
      run: black --check src/ tests/
    
    - name: Type check with mypy
      run: mypy src/tokamak_rl/
    
    - name: Test with pytest
      run: pytest --cov=src/tokamak_rl --cov-report=xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        fail_ci_if_error: true

  security:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Run safety check
      run: |
        pip install safety
        safety check --json
    
    - name: Run bandit security scan
      run: |
        pip install bandit
        bandit -r src/ -f json -o bandit-report.json
    
    - name: Upload security scan results
      uses: actions/upload-artifact@v3
      with:
        name: security-reports
        path: |
          bandit-report.json
```

### 2. Release Workflow (`.github/workflows/release.yml`)

```yaml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  build-and-publish:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install build dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine
    
    - name: Build package
      run: python -m build
    
    - name: Publish to PyPI
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
      run: twine upload dist/*
    
    - name: Create GitHub Release
      uses: actions/create-release@v1
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      with:
        tag_name: ${{ github.ref }}
        release_name: Release ${{ github.ref }}
        draft: false
        prerelease: false
```

### 3. Dependency Updates (`.github/workflows/dependencies.yml`)

```yaml
name: Update Dependencies

on:
  schedule:
    - cron: '0 2 * * MON'  # Weekly on Monday at 2 AM
  workflow_dispatch:

jobs:
  update-dependencies:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
      with:
        token: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install pip-tools
      run: pip install pip-tools
    
    - name: Update dependencies
      run: |
        pip-compile --upgrade pyproject.toml
        pip-compile --upgrade --extra dev pyproject.toml
    
    - name: Create Pull Request
      uses: peter-evans/create-pull-request@v5
      with:
        token: ${{ secrets.GITHUB_TOKEN }}
        commit-message: 'chore: update dependencies'
        title: 'Automated dependency updates'
        body: |
          Automated dependency updates
          
          - Updated all dependencies to latest compatible versions
          - Ran automated tests to ensure compatibility
        branch: automated-dependency-updates
```

## Security Workflows

### 4. CodeQL Analysis (`.github/workflows/codeql.yml`)

```yaml
name: "CodeQL"

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '30 1 * * 2'

jobs:
  analyze:
    name: Analyze
    runs-on: ubuntu-latest
    permissions:
      actions: read
      contents: read
      security-events: write

    strategy:
      fail-fast: false
      matrix:
        language: [ 'python' ]

    steps:
    - name: Checkout repository
      uses: actions/checkout@v4

    - name: Initialize CodeQL
      uses: github/codeql-action/init@v2
      with:
        languages: ${{ matrix.language }}

    - name: Autobuild
      uses: github/codeql-action/autobuild@v2

    - name: Perform CodeQL Analysis
      uses: github/codeql-action/analyze@v2
```

## Integration Requirements

### Environment Variables
Set these in GitHub repository secrets:
- `PYPI_API_TOKEN`: For package publishing
- `CODECOV_TOKEN`: For coverage reporting (optional)

### Branch Protection Rules

Configure the following branch protection rules for `main`:

1. **Require status checks to pass before merging**
   - CI/CD Pipeline (test)
   - Security scan
   - CodeQL analysis

2. **Require branches to be up to date before merging**

3. **Require pull request reviews before merging**
   - Required reviewers: 1
   - Dismiss stale reviews when new commits are pushed

4. **Restrict pushes that create files**
   - Enable "Restrict pushes that create files"

### Required Repository Settings

1. **Actions permissions**: Allow all actions and reusable workflows
2. **Workflow permissions**: Read and write permissions
3. **Fork pull request workflows**: Require approval for first-time contributors

## Performance Testing Integration

### 5. Performance Benchmarks (`.github/workflows/benchmarks.yml`)

```yaml
name: Performance Benchmarks

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -e ".[dev]"
        pip install pytest-benchmark
    
    - name: Run benchmarks
      run: pytest tests/benchmarks/ --benchmark-json=benchmark.json
    
    - name: Store benchmark result
      uses: benchmark-action/github-action-benchmark@v1
      with:
        tool: 'pytest'
        output-file-path: benchmark.json
        github-token: ${{ secrets.GITHUB_TOKEN }}
        auto-push: true
```

## Setup Instructions

1. **Create `.github/workflows/` directory**:
   ```bash
   mkdir -p .github/workflows
   ```

2. **Add workflow files**: Copy the YAML content above into respective files

3. **Configure repository secrets**: Add required tokens in GitHub settings

4. **Test workflows**: Create a test PR to verify all workflows execute correctly

5. **Monitor and iterate**: Review workflow performance and adjust as needed

## Best Practices

- **Keep workflows fast**: Parallel jobs and efficient caching
- **Security first**: Never expose secrets in logs
- **Comprehensive testing**: Multiple Python versions and environments
- **Clear failure reporting**: Detailed error messages and artifacts
- **Regular maintenance**: Update workflow dependencies monthly

This setup provides a robust CI/CD foundation that automatically validates code quality, security, and functionality while enabling automated releases and dependency management.