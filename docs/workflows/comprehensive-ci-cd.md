# Comprehensive CI/CD Workflows for Tokamak RL Control Suite

This document outlines the complete CI/CD automation required for the project. Copy these workflows to `.github/workflows/` to enable automated testing, security scanning, and deployment.

## Required GitHub Actions Workflows

### 1. Main CI/CD Pipeline (`ci-cd.yml`)

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]
  schedule:
    - cron: '0 2 * * *'  # Daily security scans

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: ['3.8', '3.9', '3.10', '3.11', '3.12']
        exclude:
          - os: windows-latest
            python-version: '3.8'
          - os: macos-latest
            python-version: '3.8'

    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
        
    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/pyproject.toml') }}
        restore-keys: |
          ${{ runner.os }}-pip-
          
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev,docs]"
        
    - name: Run pre-commit hooks
      run: |
        pre-commit install
        pre-commit run --all-files
        
    - name: Run tests with coverage
      run: |
        pytest --cov=src/tokamak_rl --cov-report=xml --cov-report=html
        
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella
        fail_ci_if_error: true

  security:
    runs-on: ubuntu-latest
    needs: test
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
        
    - name: Install security tools
      run: |
        pip install bandit safety pip-audit semgrep
        
    - name: Run Bandit security scan
      run: |
        bandit -r src/ -f json -o bandit-report.json
        
    - name: Run Safety check
      run: |
        safety check --json --output safety-report.json
        
    - name: Run pip-audit
      run: |
        pip-audit --format=json --output=pip-audit-report.json
        
    - name: Run Semgrep
      run: |
        semgrep --config=auto src/ --json --output=semgrep-report.json
        
    - name: Upload security reports
      uses: actions/upload-artifact@v3
      with:
        name: security-reports
        path: |
          bandit-report.json
          safety-report.json
          pip-audit-report.json
          semgrep-report.json

  performance:
    runs-on: ubuntu-latest
    needs: test
    
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
        
    - name: Run performance benchmarks
      run: |
        pytest tests/performance/ --benchmark-json=benchmark.json
        
    - name: Store benchmark results
      uses: benchmark-action/github-action-benchmark@v1
      with:
        tool: 'pytest'
        output-file-path: benchmark.json
        github-token: ${{ secrets.GITHUB_TOKEN }}
        auto-push: true

  build:
    runs-on: ubuntu-latest
    needs: [test, security]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
        
    - name: Install build dependencies
      run: |
        pip install build twine
        
    - name: Build package
      run: |
        python -m build
        
    - name: Check package
      run: |
        twine check dist/*
        
    - name: Upload build artifacts
      uses: actions/upload-artifact@v3
      with:
        name: dist
        path: dist/

  docker:
    runs-on: ubuntu-latest
    needs: [test, security]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
      
    - name: Login to GitHub Container Registry
      uses: docker/login-action@v3
      with:
        registry: ghcr.io
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
        
    - name: Build and push Docker images
      uses: docker/build-push-action@v5
      with:
        context: .
        push: ${{ github.event_name != 'pull_request' }}
        tags: |
          ghcr.io/${{ github.repository }}:latest
          ghcr.io/${{ github.repository }}:${{ github.sha }}
        cache-from: type=gha
        cache-to: type=gha,mode=max

  release:
    runs-on: ubuntu-latest
    needs: [build, docker]
    if: github.event_name == 'push' && startsWith(github.ref, 'refs/tags/')
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Download build artifacts
      uses: actions/download-artifact@v3
      with:
        name: dist
        path: dist/
        
    - name: Publish to PyPI
      uses: pypa/gh-action-pypi-publish@release/v1
      with:
        password: ${{ secrets.PYPI_API_TOKEN }}
        
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

### 2. Dependency Update Automation (`dependency-update.yml`)

```yaml
name: Dependency Updates

on:
  schedule:
    - cron: '0 6 * * 1'  # Weekly on Monday
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
        
    - name: Create Pull Request
      uses: peter-evans/create-pull-request@v5
      with:
        token: ${{ secrets.GITHUB_TOKEN }}
        commit-message: 'chore: update dependencies'
        title: 'Automated dependency updates'
        body: |
          This PR updates project dependencies to their latest versions.
          
          Please review the changes and ensure all tests pass before merging.
        branch: dependency-updates
        delete-branch: true
```

### 3. Code Quality Monitoring (`code-quality.yml`)

```yaml
name: Code Quality

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  sonarcloud:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0
        
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
        
    - name: Install dependencies and run tests
      run: |
        pip install -e ".[dev]"
        pytest --cov=src/tokamak_rl --cov-report=xml
        
    - name: SonarCloud Scan
      uses: SonarSource/sonarcloud-github-action@master
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        SONAR_TOKEN: ${{ secrets.SONAR_TOKEN }}

  code-climate:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
        
    - name: Install dependencies and run tests
      run: |
        pip install -e ".[dev]"
        pytest --cov=src/tokamak_rl --cov-report=xml
        
    - name: Publish code coverage
      uses: paambaati/codeclimate-action@v5.0.0
      env:
        CC_TEST_REPORTER_ID: ${{ secrets.CC_TEST_REPORTER_ID }}
      with:
        coverageLocations: coverage.xml:coverage.py
```

### 4. Performance Monitoring (`performance-monitor.yml`)

```yaml
name: Performance Monitor

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 8 * * *'  # Daily performance checks

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
      run: |
        pytest tests/performance/ --benchmark-json=benchmark.json
        
    - name: Store benchmark result
      uses: benchmark-action/github-action-benchmark@v1
      with:
        tool: 'pytest'
        output-file-path: benchmark.json
        github-token: ${{ secrets.GITHUB_TOKEN }}
        auto-push: true
        comment-on-alert: true
        alert-threshold: '110%'
        fail-on-alert: true

  memory-profiling:
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
        pip install memory-profiler psutil
        
    - name: Run memory profiling
      run: |
        python -m memory_profiler tests/performance/test_memory_usage.py
```

## Required Secrets Configuration

Add these secrets to your GitHub repository settings:

1. **PYPI_API_TOKEN**: PyPI API token for package publishing
2. **SONAR_TOKEN**: SonarCloud authentication token
3. **CC_TEST_REPORTER_ID**: Code Climate test reporter ID
4. **GITHUB_TOKEN**: Automatically provided by GitHub Actions

## Branch Protection Rules

Configure these branch protection rules for `main`:

```yaml
# .github/branch-protection.yml
protection_rules:
  main:
    required_status_checks:
      strict: true
      contexts:
        - "test"
        - "security"
        - "performance"
        - "build"
    enforce_admins: true
    required_pull_request_reviews:
      required_approving_review_count: 1
      dismiss_stale_reviews: true
      require_code_owner_reviews: true
    restrictions: null
```

## Integration Setup Checklist

- [ ] Copy workflows to `.github/workflows/` directory
- [ ] Configure required secrets in repository settings
- [ ] Set up branch protection rules
- [ ] Configure SonarCloud project (optional)
- [ ] Configure Code Climate project (optional)
- [ ] Enable GitHub Pages for documentation (optional)
- [ ] Configure Dependabot for automated dependency updates
- [ ] Set up container registry permissions

## Monitoring and Alerts

The workflows include comprehensive monitoring:

1. **Test Coverage**: Tracks code coverage trends
2. **Security Scans**: Daily vulnerability assessments
3. **Performance**: Benchmark regression detection
4. **Dependencies**: Automated security updates
5. **Code Quality**: Continuous quality metrics

All failures trigger GitHub notifications and can be configured to send alerts to Slack, Discord, or email.