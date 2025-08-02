#!/usr/bin/env python3
"""
Automated metrics collection script for tokamak-rl-control-suite

This script collects various metrics from different sources and provides
a unified interface for metrics monitoring and reporting.
"""

import json
import os
import sys
import time
import logging
import argparse
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import requests
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

class MetricsCollector:
    """Main metrics collection class"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or str(project_root / ".github" / "project-metrics.json")
        self.config = self.load_config()
        self.setup_logging()
        self.session = requests.Session()
        
    def load_config(self) -> Dict[str, Any]:
        """Load metrics configuration"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logging.error(f"Config file not found: {self.config_path}")
            return {}
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON in config file: {e}")
            return {}
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('metrics_collection.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def collect_all_metrics(self) -> Dict[str, Any]:
        """Collect all configured metrics"""
        results = {
            'timestamp': datetime.utcnow().isoformat(),
            'project': self.config.get('project', {}),
            'metrics': {}
        }
        
        # Collect each category of metrics
        for category in self.config.get('metrics', {}):
            self.logger.info(f"Collecting {category} metrics...")
            results['metrics'][category] = self.collect_category_metrics(category)
        
        return results
    
    def collect_category_metrics(self, category: str) -> Dict[str, Any]:
        """Collect metrics for a specific category"""
        category_config = self.config['metrics'].get(category, {})
        results = {}
        
        for metric_name, metric_config in category_config.items():
            try:
                value = self.collect_single_metric(metric_name, metric_config)
                results[metric_name] = {
                    'value': value,
                    'target': metric_config.get('target'),
                    'unit': metric_config.get('unit'),
                    'timestamp': datetime.utcnow().isoformat(),
                    'source': metric_config.get('source'),
                    'status': self.evaluate_metric(value, metric_config)
                }
            except Exception as e:
                self.logger.error(f"Failed to collect {metric_name}: {e}")
                results[metric_name] = {
                    'value': None,
                    'error': str(e),
                    'timestamp': datetime.utcnow().isoformat()
                }
        
        return results
    
    def collect_single_metric(self, metric_name: str, config: Dict[str, Any]) -> Optional[float]:
        """Collect a single metric based on its configuration"""
        source = config.get('source', '')
        
        if 'github' in source:
            return self.collect_github_metric(metric_name, config)
        elif 'codecov' in source:
            return self.collect_codecov_metric(metric_name, config)
        elif 'pytest' in source:
            return self.collect_pytest_metric(metric_name, config)
        elif 'docker' in source:
            return self.collect_docker_metric(metric_name, config)
        elif 'git' in source:
            return self.collect_git_metric(metric_name, config)
        elif 'prometheus' in source:
            return self.collect_prometheus_metric(metric_name, config)
        elif 'safety' in source or 'bandit' in source:
            return self.collect_security_metric(metric_name, config)
        else:
            self.logger.warning(f"Unknown source for metric {metric_name}: {source}")
            return None
    
    def collect_github_metric(self, metric_name: str, config: Dict[str, Any]) -> Optional[float]:
        """Collect metrics from GitHub API"""
        github_token = os.getenv('GITHUB_TOKEN')
        if not github_token:
            self.logger.warning("GITHUB_TOKEN not set, skipping GitHub metrics")
            return None
        
        repo = os.getenv('GITHUB_REPOSITORY', 'user/repo')
        headers = {
            'Authorization': f'token {github_token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        
        try:
            if metric_name == 'commit_frequency':
                # Get commits from last week
                since = (datetime.utcnow() - timedelta(days=7)).isoformat()
                url = f"https://api.github.com/repos/{repo}/commits"
                params = {'since': since, 'per_page': 100}
                response = self.session.get(url, headers=headers, params=params)
                response.raise_for_status()
                return len(response.json())
            
            elif metric_name == 'pr_review_time':
                # Get recent PRs and calculate average review time
                url = f"https://api.github.com/repos/{repo}/pulls"
                params = {'state': 'closed', 'per_page': 50}
                response = self.session.get(url, headers=headers, params=params)
                response.raise_for_status()
                
                review_times = []
                for pr in response.json():
                    if pr.get('merged_at'):
                        created = datetime.fromisoformat(pr['created_at'].replace('Z', '+00:00'))
                        merged = datetime.fromisoformat(pr['merged_at'].replace('Z', '+00:00'))
                        review_times.append((merged - created).total_seconds() / 3600)
                
                return sum(review_times) / len(review_times) if review_times else None
            
            elif metric_name == 'deployment_success_rate':
                # Get workflow runs for deployment
                url = f"https://api.github.com/repos/{repo}/actions/runs"
                params = {'per_page': 100}
                response = self.session.get(url, headers=headers, params=params)
                response.raise_for_status()
                
                deployment_runs = [r for r in response.json()['workflow_runs'] 
                                 if 'deploy' in r['name'].lower()]
                
                if not deployment_runs:
                    return None
                
                success_count = sum(1 for r in deployment_runs if r['conclusion'] == 'success')
                return (success_count / len(deployment_runs)) * 100
            
        except requests.RequestException as e:
            self.logger.error(f"GitHub API error for {metric_name}: {e}")
            return None
    
    def collect_codecov_metric(self, metric_name: str, config: Dict[str, Any]) -> Optional[float]:
        """Collect metrics from Codecov"""
        codecov_token = os.getenv('CODECOV_TOKEN')
        if not codecov_token:
            self.logger.warning("CODECOV_TOKEN not set, skipping Codecov metrics")
            return None
        
        repo = os.getenv('GITHUB_REPOSITORY', 'user/repo')
        
        try:
            url = f"https://codecov.io/api/v2/github/{repo}"
            headers = {'Authorization': f'Bearer {codecov_token}'}
            response = self.session.get(url, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            if metric_name == 'test_coverage':
                return data.get('totals', {}).get('coverage', 0)
            
        except requests.RequestException as e:
            self.logger.error(f"Codecov API error for {metric_name}: {e}")
            return None
    
    def collect_pytest_metric(self, metric_name: str, config: Dict[str, Any]) -> Optional[float]:
        """Collect metrics from pytest execution"""
        try:
            if metric_name == 'test_execution_time':
                # Run pytest with timing
                result = subprocess.run(
                    ['python', '-m', 'pytest', '--tb=no', '-q', '--durations=0'],
                    cwd=project_root,
                    capture_output=True,
                    text=True,
                    timeout=600
                )
                
                # Parse execution time from output
                for line in result.stdout.split('\n'):
                    if 'seconds' in line and 'passed' in line:
                        # Extract time value
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if 'seconds' in part and i > 0:
                                return float(parts[i-1])
                
                return None
            
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError) as e:
            self.logger.error(f"Pytest metric collection error for {metric_name}: {e}")
            return None
    
    def collect_docker_metric(self, metric_name: str, config: Dict[str, Any]) -> Optional[float]:
        """Collect Docker-related metrics"""
        try:
            if metric_name == 'docker_image_size':
                # Get image size
                result = subprocess.run(
                    ['docker', 'images', '--format', 'table {{.Size}}', 'tokamak-rl-control'],
                    capture_output=True,
                    text=True
                )
                
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:
                    size_str = lines[1].strip()
                    # Convert size to MB
                    if 'GB' in size_str:
                        return float(size_str.replace('GB', '')) * 1024
                    elif 'MB' in size_str:
                        return float(size_str.replace('MB', ''))
                
                return None
            
        except (subprocess.CalledProcessError, ValueError) as e:
            self.logger.error(f"Docker metric collection error for {metric_name}: {e}")
            return None
    
    def collect_git_metric(self, metric_name: str, config: Dict[str, Any]) -> Optional[float]:
        """Collect Git-related metrics"""
        try:
            if metric_name == 'code_churn':
                # Calculate code churn for last week
                since = (datetime.utcnow() - timedelta(days=7)).strftime('%Y-%m-%d')
                result = subprocess.run(
                    ['git', 'log', '--since', since, '--numstat', '--pretty=format:'],
                    cwd=project_root,
                    capture_output=True,
                    text=True
                )
                
                lines_added = 0
                lines_deleted = 0
                
                for line in result.stdout.split('\n'):
                    if line.strip() and '\t' in line:
                        parts = line.split('\t')
                        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                            lines_added += int(parts[0])
                            lines_deleted += int(parts[1])
                
                total_changes = lines_added + lines_deleted
                if total_changes > 0:
                    return (lines_deleted / total_changes) * 100
                
                return 0
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Git metric collection error for {metric_name}: {e}")
            return None
    
    def collect_prometheus_metric(self, metric_name: str, config: Dict[str, Any]) -> Optional[float]:
        """Collect metrics from Prometheus"""
        prometheus_url = os.getenv('PROMETHEUS_URL', 'http://localhost:9090')
        
        # Define queries for different metrics
        queries = {
            'uptime': 'avg_over_time(up[1h])',
            'error_rate': 'rate(application_errors_total[5m])',
            'inference_latency': 'histogram_quantile(0.95, rl_inference_duration_seconds_bucket)'
        }
        
        query = queries.get(metric_name)
        if not query:
            return None
        
        try:
            url = f"{prometheus_url}/api/v1/query"
            params = {'query': query}
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            if data['status'] == 'success' and data['data']['result']:
                return float(data['data']['result'][0]['value'][1])
            
            return None
            
        except (requests.RequestException, KeyError, ValueError) as e:
            self.logger.error(f"Prometheus metric collection error for {metric_name}: {e}")
            return None
    
    def collect_security_metric(self, metric_name: str, config: Dict[str, Any]) -> Optional[float]:
        """Collect security-related metrics"""
        try:
            if metric_name == 'vulnerability_count':
                # Run safety check
                result = subprocess.run(
                    ['python', '-m', 'safety', 'check', '--json'],
                    cwd=project_root,
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    return 0  # No vulnerabilities
                else:
                    # Parse JSON output to count vulnerabilities
                    try:
                        data = json.loads(result.stdout)
                        return len(data.get('vulnerabilities', []))
                    except json.JSONDecodeError:
                        # Count lines in output as rough estimate
                        return len([l for l in result.stdout.split('\n') if 'vulnerability' in l.lower()])
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Security metric collection error for {metric_name}: {e}")
            return None
    
    def evaluate_metric(self, value: Optional[float], config: Dict[str, Any]) -> str:
        """Evaluate metric status based on target and thresholds"""
        if value is None:
            return 'unknown'
        
        target = config.get('target')
        if target is None:
            return 'no_target'
        
        # Determine if higher or lower values are better based on metric type
        metric_name = config.get('name', '')
        
        # Metrics where lower is better
        lower_is_better = [
            'vulnerability_count', 'error_rate', 'build_time', 'test_execution_time',
            'docker_image_size', 'inference_latency', 'recovery_time', 'pr_review_time',
            'issue_resolution_time', 'code_churn', 'technical_debt', 'plasma_shape_error',
            'disruption_rate'
        ]
        
        if any(metric in metric_name for metric in lower_is_better):
            # Lower is better
            if value <= target:
                return 'good'
            elif value <= target * 1.2:
                return 'warning'
            else:
                return 'critical'
        else:
            # Higher is better
            if value >= target:
                return 'good'
            elif value >= target * 0.8:
                return 'warning'
            else:
                return 'critical'
    
    def save_metrics(self, metrics: Dict[str, Any], output_file: str = None):
        """Save collected metrics to file"""
        if output_file is None:
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            output_file = f"metrics_{timestamp}.json"
        
        try:
            with open(output_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            self.logger.info(f"Metrics saved to {output_file}")
        except IOError as e:
            self.logger.error(f"Failed to save metrics: {e}")
    
    def generate_report(self, metrics: Dict[str, Any]) -> str:
        """Generate a human-readable metrics report"""
        report = []
        report.append("# Metrics Collection Report")
        report.append(f"Generated: {metrics['timestamp']}")
        report.append(f"Project: {metrics['project'].get('name', 'Unknown')}")
        report.append("")
        
        for category, category_metrics in metrics['metrics'].items():
            report.append(f"## {category.replace('_', ' ').title()}")
            report.append("")
            
            for metric_name, metric_data in category_metrics.items():
                status = metric_data.get('status', 'unknown')
                value = metric_data.get('value', 'N/A')
                target = metric_data.get('target', 'N/A')
                unit = metric_data.get('unit', '')
                
                status_emoji = {
                    'good': '✅',
                    'warning': '⚠️',
                    'critical': '❌',
                    'unknown': '❓',
                    'no_target': '➖'
                }.get(status, '❓')
                
                report.append(f"- **{metric_name.replace('_', ' ').title()}**: {value} {unit} (target: {target}) {status_emoji}")
            
            report.append("")
        
        return '\n'.join(report)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Collect project metrics')
    parser.add_argument('--config', help='Path to metrics configuration file')
    parser.add_argument('--output', help='Output file for metrics data')
    parser.add_argument('--category', help='Collect only specific category of metrics')
    parser.add_argument('--report', action='store_true', help='Generate human-readable report')
    parser.add_argument('--quiet', action='store_true', help='Suppress output')
    
    args = parser.parse_args()
    
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    collector = MetricsCollector(args.config)
    
    if args.category:
        metrics = {
            'timestamp': datetime.utcnow().isoformat(),
            'project': collector.config.get('project', {}),
            'metrics': {args.category: collector.collect_category_metrics(args.category)}
        }
    else:
        metrics = collector.collect_all_metrics()
    
    if args.output:
        collector.save_metrics(metrics, args.output)
    
    if args.report:
        report = collector.generate_report(metrics)
        print(report)
    elif not args.quiet:
        print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()