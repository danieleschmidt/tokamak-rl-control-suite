#!/usr/bin/env python3
"""
Repository maintenance automation script for tokamak-rl-control-suite

This script performs various repository maintenance tasks including
cleanup, optimization, and health checks.
"""

import os
import sys
import json
import subprocess
import logging
import argparse
import shutil
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import tempfile

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

class RepositoryMaintenance:
    """Repository maintenance automation"""
    
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.project_root = project_root
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('repo_maintenance.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def cleanup_old_branches(self, days_old: int = 30) -> Dict[str, Any]:
        """Clean up old merged branches"""
        results = {
            'cleaned_branches': [],
            'skipped_branches': [],
            'errors': []
        }
        
        try:
            # Get list of merged branches
            result = subprocess.run(
                ['git', 'branch', '--merged', 'main'],
                capture_output=True,
                text=True,
                check=True,
                cwd=self.project_root
            )
            
            merged_branches = []
            for line in result.stdout.split('\n'):
                branch = line.strip().replace('*', '').strip()
                if branch and branch not in ['main', 'master', 'develop']:
                    merged_branches.append(branch)
            
            # Check age of each branch
            cutoff_date = datetime.now() - timedelta(days=days_old)
            
            for branch in merged_branches:
                try:
                    # Get last commit date for branch
                    result = subprocess.run(
                        ['git', 'log', '-1', '--format=%ci', branch],
                        capture_output=True,
                        text=True,
                        check=True,
                        cwd=self.project_root
                    )
                    
                    last_commit_str = result.stdout.strip()
                    last_commit_date = datetime.strptime(
                        last_commit_str.split(' ')[0], '%Y-%m-%d'
                    )
                    
                    if last_commit_date < cutoff_date:
                        if self.dry_run:
                            self.logger.info(f"DRY RUN: Would delete branch {branch}")
                        else:
                            subprocess.run(
                                ['git', 'branch', '-d', branch],
                                check=True,
                                cwd=self.project_root
                            )
                            self.logger.info(f"Deleted old branch: {branch}")
                        
                        results['cleaned_branches'].append(branch)
                    else:
                        results['skipped_branches'].append(branch)
                        
                except subprocess.CalledProcessError as e:
                    error_msg = f"Error processing branch {branch}: {e}"
                    self.logger.error(error_msg)
                    results['errors'].append(error_msg)
            
        except subprocess.CalledProcessError as e:
            error_msg = f"Error getting merged branches: {e}"
            self.logger.error(error_msg)
            results['errors'].append(error_msg)
        
        return results
    
    def cleanup_old_artifacts(self) -> Dict[str, Any]:
        """Clean up old build artifacts and temporary files"""
        results = {
            'cleaned_files': [],
            'cleaned_dirs': [],
            'space_freed': 0,
            'errors': []
        }
        
        # Patterns for files/directories to clean
        cleanup_patterns = [
            '**/__pycache__',
            '**/*.pyc',
            '**/*.pyo',
            '**/.pytest_cache',
            '**/node_modules',
            '**/.coverage',
            '**/htmlcov',
            '**/*.log',
            '**/build',
            '**/dist',
            '**/.tox',
            '**/.mypy_cache',
            '**/pytest-*.xml',
            '**/coverage.xml'
        ]
        
        for pattern in cleanup_patterns:
            try:
                for path in self.project_root.glob(pattern):
                    if path.exists():
                        # Calculate size before deletion
                        size = self.get_path_size(path)
                        
                        if self.dry_run:
                            self.logger.info(f"DRY RUN: Would delete {path} ({self.format_size(size)})")
                        else:
                            if path.is_file():
                                path.unlink()
                                results['cleaned_files'].append(str(path))
                            else:
                                shutil.rmtree(path)
                                results['cleaned_dirs'].append(str(path))
                            
                            self.logger.info(f"Deleted {path} ({self.format_size(size)})")
                        
                        results['space_freed'] += size
                        
            except Exception as e:
                error_msg = f"Error cleaning pattern {pattern}: {e}"
                self.logger.error(error_msg)
                results['errors'].append(error_msg)
        
        return results
    
    def optimize_git_repository(self) -> Dict[str, Any]:
        """Optimize Git repository (gc, prune, etc.)"""
        results = {
            'operations': [],
            'errors': []
        }
        
        git_operations = [
            (['git', 'gc', '--aggressive'], "Garbage collection"),
            (['git', 'prune'], "Prune unreachable objects"),
            (['git', 'remote', 'prune', 'origin'], "Prune remote tracking branches"),
            (['git', 'reflog', 'expire', '--expire=30.days.ago', '--all'], "Expire old reflog entries")
        ]
        
        for cmd, description in git_operations:
            try:
                if self.dry_run:
                    self.logger.info(f"DRY RUN: Would run {description}")
                    results['operations'].append(f"DRY RUN: {description}")
                else:
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        check=True,
                        cwd=self.project_root
                    )
                    self.logger.info(f"Completed: {description}")
                    results['operations'].append(description)
                    
            except subprocess.CalledProcessError as e:
                error_msg = f"Error in {description}: {e}"
                self.logger.error(error_msg)
                results['errors'].append(error_msg)
        
        return results
    
    def check_repository_health(self) -> Dict[str, Any]:
        """Perform repository health checks"""
        health_results = {
            'git_status': {},
            'file_checks': {},
            'dependency_checks': {},
            'security_checks': {},
            'overall_health': 'unknown'
        }
        
        # Git status check
        try:
            result = subprocess.run(
                ['git', 'status', '--porcelain'],
                capture_output=True,
                text=True,
                check=True,
                cwd=self.project_root
            )
            
            uncommitted_files = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
            health_results['git_status'] = {
                'uncommitted_files': uncommitted_files,
                'clean': uncommitted_files == 0
            }
            
        except subprocess.CalledProcessError as e:
            health_results['git_status'] = {'error': str(e)}
        
        # File structure checks
        required_files = [
            'README.md', 'LICENSE', 'pyproject.toml', 'requirements.txt',
            '.gitignore', '.pre-commit-config.yaml'
        ]
        
        missing_files = []
        for file_name in required_files:
            if not (self.project_root / file_name).exists():
                missing_files.append(file_name)
        
        health_results['file_checks'] = {
            'missing_files': missing_files,
            'all_present': len(missing_files) == 0
        }
        
        # Dependency health check
        try:
            result = subprocess.run(
                ['pip', 'check'],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            health_results['dependency_checks'] = {
                'conflicts': result.returncode != 0,
                'details': result.stdout if result.returncode != 0 else "No conflicts"
            }
            
        except subprocess.CalledProcessError as e:
            health_results['dependency_checks'] = {'error': str(e)}
        
        # Security check
        try:
            result = subprocess.run(
                ['python', '-m', 'safety', 'check', '--short-report'],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            health_results['security_checks'] = {
                'vulnerabilities_found': result.returncode != 0,
                'report': result.stdout
            }
            
        except subprocess.CalledProcessError as e:
            health_results['security_checks'] = {'error': str(e)}
        
        # Overall health assessment
        issues = []
        if not health_results['git_status'].get('clean', True):
            issues.append("Uncommitted changes")
        if health_results['file_checks']['missing_files']:
            issues.append("Missing required files")
        if health_results['dependency_checks'].get('conflicts', False):
            issues.append("Dependency conflicts")
        if health_results['security_checks'].get('vulnerabilities_found', False):
            issues.append("Security vulnerabilities")
        
        if not issues:
            health_results['overall_health'] = 'excellent'
        elif len(issues) <= 2:
            health_results['overall_health'] = 'good'
        else:
            health_results['overall_health'] = 'needs_attention'
        
        health_results['issues'] = issues
        
        return health_results
    
    def update_documentation_links(self) -> Dict[str, Any]:
        """Check and update documentation links"""
        results = {
            'checked_files': [],
            'broken_links': [],
            'updated_links': [],
            'errors': []
        }
        
        # Find all markdown files
        md_files = list(self.project_root.glob('**/*.md'))
        
        for md_file in md_files:
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                results['checked_files'].append(str(md_file))
                
                # Find markdown links
                link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
                links = re.findall(link_pattern, content)
                
                updated_content = content
                content_changed = False
                
                for link_text, link_url in links:
                    # Check if it's a relative file link
                    if not link_url.startswith(('http://', 'https://', 'mailto:')):
                        # Resolve relative path
                        if link_url.startswith('/'):
                            target_path = self.project_root / link_url[1:]
                        else:
                            target_path = md_file.parent / link_url
                        
                        if not target_path.exists():
                            self.logger.warning(f"Broken link in {md_file}: {link_url}")
                            results['broken_links'].append({
                                'file': str(md_file),
                                'link': link_url,
                                'text': link_text
                            })
                
                if content_changed and not self.dry_run:
                    with open(md_file, 'w', encoding='utf-8') as f:
                        f.write(updated_content)
                    results['updated_links'].append(str(md_file))
                
            except Exception as e:
                error_msg = f"Error processing {md_file}: {e}"
                self.logger.error(error_msg)
                results['errors'].append(error_msg)
        
        return results
    
    def cleanup_docker_resources(self) -> Dict[str, Any]:
        """Clean up Docker images and containers"""
        results = {
            'removed_containers': [],
            'removed_images': [],
            'space_freed': 0,
            'errors': []
        }
        
        docker_commands = [
            (['docker', 'container', 'prune', '-f'], "Remove stopped containers"),
            (['docker', 'image', 'prune', '-f'], "Remove dangling images"),
            (['docker', 'volume', 'prune', '-f'], "Remove unused volumes")
        ]
        
        for cmd, description in docker_commands:
            try:
                if self.dry_run:
                    self.logger.info(f"DRY RUN: Would run {description}")
                else:
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        check=True
                    )
                    self.logger.info(f"Completed: {description}")
                    
                    # Parse output for space freed
                    if "Total reclaimed space" in result.stdout:
                        space_match = re.search(r'Total reclaimed space: ([\d.]+)([KMGT]?B)', result.stdout)
                        if space_match:
                            amount, unit = space_match.groups()
                            space_bytes = self.parse_size(amount, unit)
                            results['space_freed'] += space_bytes
                    
            except subprocess.CalledProcessError as e:
                error_msg = f"Error in {description}: {e}"
                self.logger.error(error_msg)
                results['errors'].append(error_msg)
            except FileNotFoundError:
                self.logger.info("Docker not available, skipping Docker cleanup")
                break
        
        return results
    
    def generate_maintenance_report(self, all_results: Dict[str, Any]) -> str:
        """Generate a comprehensive maintenance report"""
        report = []
        report.append("# Repository Maintenance Report")
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append(f"Repository: {self.project_root.name}")
        report.append("")
        
        # Branch cleanup
        if 'branch_cleanup' in all_results:
            branch_results = all_results['branch_cleanup']
            report.append("## Branch Cleanup")
            report.append(f"- Cleaned branches: {len(branch_results['cleaned_branches'])}")
            report.append(f"- Skipped branches: {len(branch_results['skipped_branches'])}")
            if branch_results['cleaned_branches']:
                report.append("- Deleted branches:")
                for branch in branch_results['cleaned_branches']:
                    report.append(f"  - {branch}")
            report.append("")
        
        # Artifact cleanup
        if 'artifact_cleanup' in all_results:
            artifact_results = all_results['artifact_cleanup']
            report.append("## Artifact Cleanup")
            report.append(f"- Files cleaned: {len(artifact_results['cleaned_files'])}")
            report.append(f"- Directories cleaned: {len(artifact_results['cleaned_dirs'])}")
            report.append(f"- Space freed: {self.format_size(artifact_results['space_freed'])}")
            report.append("")
        
        # Git optimization
        if 'git_optimization' in all_results:
            git_results = all_results['git_optimization']
            report.append("## Git Optimization")
            for operation in git_results['operations']:
                report.append(f"- ✅ {operation}")
            for error in git_results['errors']:
                report.append(f"- ❌ {error}")
            report.append("")
        
        # Health check
        if 'health_check' in all_results:
            health_results = all_results['health_check']
            report.append("## Repository Health")
            
            health_emoji = {
                'excellent': '🟢',
                'good': '🟡',
                'needs_attention': '🔴',
                'unknown': '⚪'
            }
            
            overall_health = health_results.get('overall_health', 'unknown')
            report.append(f"**Overall Health: {health_emoji[overall_health]} {overall_health.title()}**")
            report.append("")
            
            if health_results.get('issues'):
                report.append("### Issues Found:")
                for issue in health_results['issues']:
                    report.append(f"- ⚠️ {issue}")
                report.append("")
            
            # Git status
            git_status = health_results.get('git_status', {})
            if git_status.get('clean', True):
                report.append("- ✅ Git status: Clean")
            else:
                report.append(f"- ⚠️ Git status: {git_status.get('uncommitted_files', 0)} uncommitted files")
            
            # File checks
            file_checks = health_results.get('file_checks', {})
            if file_checks.get('all_present', True):
                report.append("- ✅ Required files: All present")
            else:
                missing = file_checks.get('missing_files', [])
                report.append(f"- ⚠️ Missing files: {', '.join(missing)}")
            
            # Dependencies
            dep_checks = health_results.get('dependency_checks', {})
            if not dep_checks.get('conflicts', False):
                report.append("- ✅ Dependencies: No conflicts")
            else:
                report.append("- ⚠️ Dependencies: Conflicts detected")
            
            # Security
            sec_checks = health_results.get('security_checks', {})
            if not sec_checks.get('vulnerabilities_found', False):
                report.append("- ✅ Security: No vulnerabilities")
            else:
                report.append("- ⚠️ Security: Vulnerabilities found")
            
            report.append("")
        
        # Docker cleanup
        if 'docker_cleanup' in all_results:
            docker_results = all_results['docker_cleanup']
            report.append("## Docker Cleanup")
            report.append(f"- Space freed: {self.format_size(docker_results['space_freed'])}")
            if docker_results['errors']:
                report.append("- Errors:")
                for error in docker_results['errors']:
                    report.append(f"  - {error}")
            report.append("")
        
        # Documentation links
        if 'doc_links' in all_results:
            doc_results = all_results['doc_links']
            report.append("## Documentation Links")
            report.append(f"- Files checked: {len(doc_results['checked_files'])}")
            report.append(f"- Broken links: {len(doc_results['broken_links'])}")
            if doc_results['broken_links']:
                report.append("- Broken links found:")
                for link in doc_results['broken_links']:
                    report.append(f"  - {link['file']}: {link['link']}")
            report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        report.append("- Run this maintenance script weekly")
        report.append("- Monitor repository health metrics")
        report.append("- Address any security vulnerabilities promptly")
        report.append("- Keep dependencies up to date")
        report.append("- Maintain clean commit history")
        
        return '\n'.join(report)
    
    def get_path_size(self, path: Path) -> int:
        """Get size of file or directory"""
        if path.is_file():
            return path.stat().st_size
        elif path.is_dir():
            total = 0
            for item in path.rglob('*'):
                if item.is_file():
                    try:
                        total += item.stat().st_size
                    except (OSError, FileNotFoundError):
                        pass
            return total
        return 0
    
    def format_size(self, size_bytes: int) -> str:
        """Format size in human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"
    
    def parse_size(self, amount: str, unit: str) -> int:
        """Parse size string to bytes"""
        multipliers = {
            'B': 1,
            'KB': 1024,
            'MB': 1024**2,
            'GB': 1024**3,
            'TB': 1024**4
        }
        
        return int(float(amount) * multipliers.get(unit, 1))


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Repository maintenance automation')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without making changes')
    parser.add_argument('--branches', action='store_true',
                       help='Clean up old merged branches')
    parser.add_argument('--artifacts', action='store_true',
                       help='Clean up build artifacts and temp files')
    parser.add_argument('--git', action='store_true',
                       help='Optimize Git repository')
    parser.add_argument('--docker', action='store_true',
                       help='Clean up Docker resources')
    parser.add_argument('--docs', action='store_true',
                       help='Check documentation links')
    parser.add_argument('--health', action='store_true',
                       help='Perform repository health check')
    parser.add_argument('--all', action='store_true',
                       help='Run all maintenance tasks')
    parser.add_argument('--report', help='Generate maintenance report to file')
    parser.add_argument('--branch-age', type=int, default=30,
                       help='Age in days for branch cleanup (default: 30)')
    
    args = parser.parse_args()
    
    # If no specific tasks selected, run health check by default
    if not any([args.branches, args.artifacts, args.git, args.docker, 
               args.docs, args.health, args.all]):
        args.health = True
    
    maintenance = RepositoryMaintenance(dry_run=args.dry_run)
    results = {}
    
    try:
        if args.all or args.health:
            maintenance.logger.info("Running repository health check...")
            results['health_check'] = maintenance.check_repository_health()
        
        if args.all or args.branches:
            maintenance.logger.info("Cleaning up old branches...")
            results['branch_cleanup'] = maintenance.cleanup_old_branches(args.branch_age)
        
        if args.all or args.artifacts:
            maintenance.logger.info("Cleaning up artifacts...")
            results['artifact_cleanup'] = maintenance.cleanup_old_artifacts()
        
        if args.all or args.git:
            maintenance.logger.info("Optimizing Git repository...")
            results['git_optimization'] = maintenance.optimize_git_repository()
        
        if args.all or args.docker:
            maintenance.logger.info("Cleaning up Docker resources...")
            results['docker_cleanup'] = maintenance.cleanup_docker_resources()
        
        if args.all or args.docs:
            maintenance.logger.info("Checking documentation links...")
            results['doc_links'] = maintenance.update_documentation_links()
        
        # Generate report
        if args.report:
            report_content = maintenance.generate_maintenance_report(results)
            with open(args.report, 'w') as f:
                f.write(report_content)
            maintenance.logger.info(f"Maintenance report saved to {args.report}")
        
        # Print summary
        maintenance.logger.info("Repository maintenance completed successfully")
        
        # Check if any critical issues were found
        health_check = results.get('health_check', {})
        if health_check.get('overall_health') == 'needs_attention':
            maintenance.logger.warning("Repository health needs attention!")
            sys.exit(1)
        
    except KeyboardInterrupt:
        maintenance.logger.info("Maintenance interrupted by user")
        sys.exit(1)
    except Exception as e:
        maintenance.logger.error(f"Unexpected error during maintenance: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()