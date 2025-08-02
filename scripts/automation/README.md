# Automation Scripts

This directory contains automation scripts for the tokamak-rl-control-suite project to streamline development, maintenance, and monitoring tasks.

## Scripts Overview

### [`collect_metrics.py`](../metrics/collect_metrics.py)
Automated metrics collection from various sources including GitHub, testing frameworks, security tools, and system metrics.

**Features:**
- Multi-source metrics aggregation
- Configurable collection schedules
- Health status evaluation
- Report generation

**Usage:**
```bash
# Collect all metrics
python scripts/metrics/collect_metrics.py

# Collect specific category
python scripts/metrics/collect_metrics.py --category performance

# Generate human-readable report
python scripts/metrics/collect_metrics.py --report

# Save to specific file
python scripts/metrics/collect_metrics.py --output metrics_report.json
```

### [`dependency_updater.py`](dependency_updater.py)
Automated dependency management with security vulnerability resolution, compatibility testing, and backup/restore capabilities.

**Features:**
- Security vulnerability updates
- Patch/minor/major version updates
- Automated testing after updates
- Backup and restore functionality
- Update reporting

**Usage:**
```bash
# Update security vulnerabilities only
python scripts/automation/dependency_updater.py --type security

# Update patch versions with testing
python scripts/automation/dependency_updater.py --type patch --test

# Dry run to see what would be updated
python scripts/automation/dependency_updater.py --type all --dry-run

# Create backup before updating
python scripts/automation/dependency_updater.py --type minor --backup

# Generate update report
python scripts/automation/dependency_updater.py --type all --report update_report.md
```

### [`repo_maintenance.py`](repo_maintenance.py)
Repository maintenance automation including cleanup, optimization, and health monitoring.

**Features:**
- Old branch cleanup
- Artifact and temporary file cleanup
- Git repository optimization
- Docker resource cleanup
- Documentation link checking
- Repository health assessment

**Usage:**
```bash
# Run health check only
python scripts/automation/repo_maintenance.py --health

# Clean up old merged branches (30+ days old)
python scripts/automation/repo_maintenance.py --branches

# Clean up all artifacts and temp files
python scripts/automation/repo_maintenance.py --artifacts

# Optimize Git repository
python scripts/automation/repo_maintenance.py --git

# Run all maintenance tasks
python scripts/automation/repo_maintenance.py --all

# Dry run to see what would be done
python scripts/automation/repo_maintenance.py --all --dry-run

# Generate maintenance report
python scripts/automation/repo_maintenance.py --all --report maintenance_report.md
```

## Configuration

### Environment Variables

```bash
# Required for GitHub metrics
export GITHUB_TOKEN="your_github_token"
export GITHUB_REPOSITORY="owner/repo-name"

# Optional for enhanced metrics
export CODECOV_TOKEN="your_codecov_token"
export PROMETHEUS_URL="http://your-prometheus-instance:9090"
export SECURITY_METRICS_ENDPOINT="https://your-metrics-endpoint"
```

### Metrics Configuration

Metrics collection is configured via [`.github/project-metrics.json`](../../.github/project-metrics.json):

```json
{
  "metrics": {
    "code_quality": {
      "test_coverage": {
        "target": 85,
        "source": "codecov",
        "frequency": "per_commit"
      }
    }
  }
}
```

## Scheduling Automation

### GitHub Actions Integration

Add to your workflow files:

```yaml
# .github/workflows/maintenance.yml
name: Repository Maintenance
on:
  schedule:
    - cron: '0 2 * * 1'  # Weekly on Monday at 2 AM
  workflow_dispatch:

jobs:
  maintenance:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run maintenance
        run: |
          python scripts/automation/repo_maintenance.py --all
          python scripts/automation/dependency_updater.py --type security
```

### Cron Jobs

For server environments:

```bash
# Add to crontab (crontab -e)

# Daily metrics collection
0 6 * * * cd /path/to/repo && python scripts/metrics/collect_metrics.py

# Weekly maintenance
0 2 * * 1 cd /path/to/repo && python scripts/automation/repo_maintenance.py --all

# Monthly dependency updates
0 3 1 * * cd /path/to/repo && python scripts/automation/dependency_updater.py --type minor
```

## Integration with Monitoring

### Prometheus Metrics Export

```python
# Integration example
from prometheus_client import CollectorRegistry, Gauge, write_to_textfile

def export_metrics_to_prometheus():
    registry = CollectorRegistry()
    
    # Create metrics
    test_coverage = Gauge('test_coverage_percent', 'Test coverage percentage', registry=registry)
    vulnerability_count = Gauge('vulnerability_count', 'Number of vulnerabilities', registry=registry)
    
    # Collect metrics
    collector = MetricsCollector()
    metrics = collector.collect_all_metrics()
    
    # Update Prometheus metrics
    test_coverage.set(metrics['metrics']['code_quality']['test_coverage']['value'])
    vulnerability_count.set(metrics['metrics']['security']['vulnerability_count']['value'])
    
    # Export to file for node_exporter
    write_to_textfile('/var/lib/node_exporter/textfile_collector/repo_metrics.prom', registry)
```

### Grafana Dashboard

Create dashboards using the exported metrics:

```json
{
  "dashboard": {
    "title": "Repository Health",
    "panels": [
      {
        "title": "Test Coverage",
        "type": "stat",
        "targets": [{"expr": "test_coverage_percent"}]
      },
      {
        "title": "Security Vulnerabilities",
        "type": "stat",
        "targets": [{"expr": "vulnerability_count"}]
      }
    ]
  }
}
```

### Alerting Integration

```yaml
# Prometheus alert rules
groups:
  - name: repository-health
    rules:
      - alert: LowTestCoverage
        expr: test_coverage_percent < 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Test coverage dropped below 80%"
      
      - alert: SecurityVulnerabilities
        expr: vulnerability_count > 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Security vulnerabilities detected"
```

## Custom Automation Scripts

### Adding New Automation

1. **Create script in appropriate directory**:
   ```bash
   touch scripts/automation/new_automation.py
   chmod +x scripts/automation/new_automation.py
   ```

2. **Follow template structure**:
   ```python
   #!/usr/bin/env python3
   """
   Description of automation script
   """
   
   import logging
   import argparse
   from pathlib import Path
   
   class NewAutomation:
       def __init__(self, dry_run: bool = False):
           self.dry_run = dry_run
           self.setup_logging()
       
       def setup_logging(self):
           logging.basicConfig(level=logging.INFO)
           self.logger = logging.getLogger(__name__)
       
       def run_automation(self):
           # Implementation here
           pass
   
   def main():
       parser = argparse.ArgumentParser()
       parser.add_argument('--dry-run', action='store_true')
       args = parser.parse_args()
       
       automation = NewAutomation(dry_run=args.dry_run)
       automation.run_automation()
   
   if __name__ == '__main__':
       main()
   ```

3. **Update documentation and configuration**

### Testing Automation Scripts

```bash
# Unit tests for automation scripts
python -m pytest tests/automation/ -v

# Integration tests
python -m pytest tests/integration/test_automation.py -v

# End-to-end tests
python scripts/automation/dependency_updater.py --dry-run
python scripts/automation/repo_maintenance.py --dry-run
```

## Best Practices

### Security
- Never commit sensitive tokens or keys
- Use environment variables for credentials
- Validate inputs and sanitize outputs
- Run with minimal required permissions

### Error Handling
- Implement comprehensive error handling
- Log errors with sufficient detail
- Provide meaningful error messages
- Implement retry logic where appropriate

### Performance
- Use caching for expensive operations
- Implement timeout handling
- Process data in chunks for large datasets
- Monitor resource usage

### Maintainability
- Follow consistent coding style
- Write comprehensive documentation
- Include unit tests
- Version control configuration changes

## Troubleshooting

### Common Issues

**Permission Errors**:
```bash
# Fix file permissions
chmod +x scripts/automation/*.py

# Fix Git permissions
git config --global --add safe.directory /path/to/repo
```

**Missing Dependencies**:
```bash
# Install required packages
pip install -r requirements.txt
pip install safety pip-audit bandit
```

**Token Issues**:
```bash
# Check token validity
curl -H "Authorization: token $GITHUB_TOKEN" https://api.github.com/user

# Set tokens in environment
export GITHUB_TOKEN="your_token_here"
```

**Metrics Collection Failures**:
```bash
# Test individual metrics
python scripts/metrics/collect_metrics.py --category code_quality

# Check configuration
python -c "import json; print(json.load(open('.github/project-metrics.json')))"
```

### Debug Mode

Enable debug logging:

```bash
# Set debug level
export PYTHONPATH=$PWD
python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
# Run automation script
"
```

## Support and Contributing

- Report issues in the project repository
- Submit feature requests via GitHub issues
- Contribute improvements via pull requests
- Follow the project's contribution guidelines

For more information, see:
- [Project Documentation](../../README.md)
- [Development Guide](../../DEVELOPMENT.md)
- [Contributing Guidelines](../../CONTRIBUTING.md)