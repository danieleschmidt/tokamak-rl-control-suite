# GitHub Workflows Setup Guide

This guide provides step-by-step instructions for setting up the GitHub Actions workflows for the tokamak-rl-control-suite.

## Overview

Due to GitHub App permission limitations, workflow files cannot be created automatically. This guide helps you manually set up the complete CI/CD pipeline using the provided templates.

## Required Workflows

### 1. Continuous Integration (CI)
**File**: `.github/workflows/ci.yml`
**Source**: [`docs/workflows/examples/ci.yml`](examples/ci.yml)

**Features**:
- Multi-version Python testing (3.8-3.11)
- Code quality checks (linting, formatting, type checking)
- Security scanning (Bandit, Trivy)
- Performance benchmarking
- GPU testing (if runners available)
- Documentation building and link checking
- Container security scanning

### 2. Continuous Deployment (CD)
**File**: `.github/workflows/cd.yml`
**Source**: [`docs/workflows/examples/cd.yml`](examples/cd.yml)

**Features**:
- Environment-specific deployment (staging, production)
- Container image building and pushing
- Security scanning of built images
- Blue-green deployment strategy
- Smoke testing and health checks
- Rollback capabilities
- Manual approval for production

### 3. Security Scanning
**File**: `.github/workflows/security-scan.yml`
**Source**: [`docs/workflows/examples/security-scan.yml`](examples/security-scan.yml)

**Features**:
- SAST (Static Application Security Testing)
- Secret detection (TruffleHog, GitLeaks)
- Dependency vulnerability scanning
- Container security scanning
- Infrastructure as Code (IaC) scanning
- License compliance checking
- SBOM generation

### 4. Dependency Updates
**File**: `.github/workflows/dependency-update.yml`
**Source**: [`docs/workflows/examples/dependency-update.yml`](examples/dependency-update.yml)

**Features**:
- Automated dependency updates (security, patch, minor, major)
- Security vulnerability resolution
- Automated testing of updates
- Pull request creation
- Branch cleanup

## Setup Instructions

### Step 1: Create Workflow Directory

```bash
mkdir -p .github/workflows
```

### Step 2: Copy Workflow Files

Copy each workflow template from `docs/workflows/examples/` to `.github/workflows/`:

```bash
cp docs/workflows/examples/ci.yml .github/workflows/
cp docs/workflows/examples/cd.yml .github/workflows/
cp docs/workflows/examples/security-scan.yml .github/workflows/
cp docs/workflows/examples/dependency-update.yml .github/workflows/
```

### Step 3: Configure Repository Secrets

Add the following secrets in your repository settings (`Settings` → `Secrets and variables` → `Actions`):

#### Required Secrets
```bash
# Container Registry
GITHUB_TOKEN                # Automatically provided by GitHub

# Security Scanning
SNYK_TOKEN                  # Snyk authentication token
FOSSA_API_KEY              # FOSSA license scanning (optional)

# Deployment
PRODUCTION_APPROVERS        # Comma-separated list of GitHub usernames

# Notifications (optional)
SLACK_WEBHOOK_URL          # Slack webhook for notifications
SECURITY_METRICS_ENDPOINT  # Endpoint for security metrics
```

#### Secret Configuration Examples

**Snyk Token**:
1. Sign up at [snyk.io](https://snyk.io)
2. Go to Account Settings → General → API Token
3. Copy the token and add as `SNYK_TOKEN` secret

**Production Approvers**:
```
user1,user2,user3
```

### Step 4: Configure Environment Protection Rules

#### Staging Environment
1. Go to `Settings` → `Environments`
2. Create `staging` environment
3. Configure protection rules:
   - Required reviewers: 1 person
   - Wait timer: 0 minutes
   - Deployment branches: `main`, `develop`

#### Production Environment
1. Create `production` environment
2. Configure protection rules:
   - Required reviewers: 2 people
   - Wait timer: 5 minutes
   - Deployment branches: `main` only
   - Environment secrets if needed

### Step 5: Configure Branch Protection Rules

#### Main Branch Protection
1. Go to `Settings` → `Branches`
2. Add rule for `main` branch:
   - Require pull request reviews (2 reviewers)
   - Require status checks to pass before merging:
     - `CI Success`
     - `Security Scanning`
   - Require branches to be up to date before merging
   - Require linear history
   - Include administrators

#### Develop Branch Protection (if using)
1. Add rule for `develop` branch:
   - Require pull request reviews (1 reviewer)
   - Require status checks to pass before merging:
     - `CI Success`

### Step 6: Configure Self-Hosted Runners (Optional)

For GPU testing and better performance, set up self-hosted runners:

#### GPU Runner Setup
1. Go to `Settings` → `Actions` → `Runners`
2. Click "New self-hosted runner"
3. Follow setup instructions for your platform
4. Add labels: `self-hosted`, `gpu`
5. Install CUDA and required GPU libraries

#### Runner Configuration
```yaml
# In workflow files, use:
runs-on: [self-hosted, gpu]
```

### Step 7: Customize Workflows

#### Modify Python Versions
In `ci.yml`, update the Python version matrix:
```yaml
strategy:
  matrix:
    python-version: ['3.8', '3.9', '3.10', '3.11']
```

#### Update Container Registry
If using a different registry, update in `cd.yml`:
```yaml
env:
  REGISTRY: your-registry.com
  IMAGE_NAME: your-org/tokamak-rl-control
```

#### Configure Deployment Targets
Update deployment configurations in `cd.yml`:
```yaml
# Staging deployment
kubectl apply -f k8s/staging/

# Production deployment  
kubectl apply -f k8s/production/
```

### Step 8: Test Workflow Setup

#### Test CI Workflow
1. Create a feature branch
2. Make a small change
3. Push to trigger CI workflow
4. Verify all checks pass

#### Test Security Scanning
1. Push to `main` branch
2. Verify security scan runs automatically
3. Check security tab for results

#### Test Dependency Updates
1. Trigger manually: `Actions` → `Dependency Updates` → `Run workflow`
2. Check for created pull requests
3. Review and merge dependency updates

## Workflow Customization

### Adding Custom Tests

#### GPU-Specific Tests
```yaml
# Add to ci.yml
gpu-test:
  runs-on: [self-hosted, gpu]
  steps:
    - name: Run GPU tests
      run: pytest tests/gpu/ --gpu-required
```

#### Performance Benchmarks
```yaml
# Add custom benchmarks
performance:
  steps:
    - name: Run custom benchmarks
      run: |
        pytest tests/performance/ \
          --benchmark-json=results.json \
          --benchmark-compare-fail=min:5%
```

### Environment-Specific Configuration

#### Staging Configuration
```yaml
environment: staging
env:
  API_URL: https://staging-api.tokamak.internal
  DATABASE_URL: ${{ secrets.STAGING_DATABASE_URL }}
```

#### Production Configuration
```yaml
environment: production
env:
  API_URL: https://api.tokamak.internal
  DATABASE_URL: ${{ secrets.PRODUCTION_DATABASE_URL }}
```

## Monitoring and Maintenance

### Workflow Health Monitoring

#### Check Workflow Success Rates
```bash
# Use GitHub CLI to monitor workflow runs
gh run list --workflow=ci.yml --limit=50
gh run list --workflow=security-scan.yml --limit=20
```

#### Set Up Workflow Failure Alerts
Add to existing monitoring system or create dedicated alerts for:
- CI failure rate > 10%
- Security scan failures
- Deployment failures
- Dependency update failures

### Regular Maintenance Tasks

#### Weekly
- Review dependency update PRs
- Check security scan results
- Verify workflow performance

#### Monthly
- Update workflow configurations
- Review and update secrets
- Audit runner usage and costs

#### Quarterly
- Review and update protection rules
- Audit environment access
- Update workflow documentation

## Troubleshooting

### Common Issues

#### Permission Errors
```
Error: Permission denied
```
**Solution**: Check repository secrets and environment permissions

#### Runner Unavailable
```
Error: No runners available
```
**Solution**: Check self-hosted runner status or use GitHub-hosted runners

#### Security Scan Failures
```
Error: Critical vulnerabilities found
```
**Solution**: Review security scan results and update dependencies

#### Deployment Failures
```
Error: Deployment failed
```
**Solution**: Check environment health and rollback if necessary

### Debug Workflows

#### Enable Debug Logging
Add to workflow file:
```yaml
env:
  ACTIONS_STEP_DEBUG: true
  ACTIONS_RUNNER_DEBUG: true
```

#### Access Workflow Logs
1. Go to `Actions` tab
2. Click on failed workflow run
3. Expand job and step logs
4. Download logs if needed

## Security Considerations

### Secret Management
- Use environment-specific secrets
- Rotate secrets regularly
- Limit secret access to necessary workflows
- Never log secret values

### Runner Security
- Use dedicated runners for sensitive operations
- Regularly update runner software
- Monitor runner access and usage
- Use ephemeral runners when possible

### Workflow Security
- Pin action versions to specific commits
- Review third-party actions before use
- Limit workflow permissions using GITHUB_TOKEN
- Use environment protection for sensitive deployments

## Performance Optimization

### Caching Strategy
```yaml
# Cache Python dependencies
- uses: actions/cache@v3
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('**/*.txt') }}

# Cache Docker layers
- uses: docker/build-push-action@v5
  with:
    cache-from: type=gha
    cache-to: type=gha,mode=max
```

### Parallel Execution
```yaml
# Run jobs in parallel
jobs:
  test-unit:
    # Unit tests
  test-integration:
    # Integration tests
  security-scan:
    # Security scanning
```

### Resource Optimization
- Use appropriate runner sizes
- Optimize Docker image builds
- Use matrix strategies efficiently
- Clean up artifacts regularly

## Next Steps

After setting up the workflows:

1. **Monitor Performance**: Track workflow execution times and success rates
2. **Iterate and Improve**: Regularly update workflows based on team feedback
3. **Document Changes**: Keep workflow documentation updated
4. **Train Team**: Ensure team members understand the CI/CD process
5. **Security Review**: Regularly audit workflow security and permissions

For additional help, refer to:
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Workflow Troubleshooting Guide](../runbooks/workflow-issues.md)
- [Security Best Practices](../SECURITY.md)