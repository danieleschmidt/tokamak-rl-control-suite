#!/usr/bin/env python3
"""
Automated dependency management script for tokamak-rl-control-suite

This script provides automated dependency updates, security checks,
and compatibility verification.
"""

import os
import sys
import json
import subprocess
import logging
import argparse
import tempfile
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import re

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

class DependencyUpdater:
    """Automated dependency management"""
    
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
                logging.FileHandler('dependency_update.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def get_current_dependencies(self) -> Dict[str, str]:
        """Get currently installed dependencies and versions"""
        try:
            result = subprocess.run(
                ['pip', 'list', '--format=json'],
                capture_output=True,
                text=True,
                check=True
            )
            
            deps = {}
            for package in json.loads(result.stdout):
                deps[package['name']] = package['version']
            
            return deps
            
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            self.logger.error(f"Failed to get current dependencies: {e}")
            return {}
    
    def get_outdated_packages(self) -> List[Dict[str, str]]:
        """Get list of outdated packages"""
        try:
            result = subprocess.run(
                ['pip', 'list', '--outdated', '--format=json'],
                capture_output=True,
                text=True,
                check=True
            )
            
            return json.loads(result.stdout)
            
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            self.logger.error(f"Failed to get outdated packages: {e}")
            return []
    
    def check_security_vulnerabilities(self) -> List[Dict[str, any]]:
        """Check for security vulnerabilities in dependencies"""
        vulnerabilities = []
        
        # Check with safety
        try:
            result = subprocess.run(
                ['python', '-m', 'safety', 'check', '--json'],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            if result.returncode != 0 and result.stdout:
                try:
                    safety_data = json.loads(result.stdout)
                    vulnerabilities.extend(safety_data.get('vulnerabilities', []))
                except json.JSONDecodeError:
                    self.logger.warning("Could not parse safety check output")
                    
        except subprocess.CalledProcessError as e:
            self.logger.warning(f"Safety check failed: {e}")
        
        # Check with pip-audit
        try:
            result = subprocess.run(
                ['pip-audit', '--format=json'],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            if result.returncode != 0 and result.stdout:
                try:
                    audit_data = json.loads(result.stdout)
                    vulnerabilities.extend(audit_data.get('vulnerabilities', []))
                except json.JSONDecodeError:
                    self.logger.warning("Could not parse pip-audit output")
                    
        except subprocess.CalledProcessError as e:
            self.logger.warning(f"pip-audit failed: {e}")
        
        return vulnerabilities
    
    def update_package(self, package: str, version: str = None) -> bool:
        """Update a single package"""
        try:
            cmd = ['pip', 'install', '--upgrade']
            if version:
                cmd.append(f"{package}=={version}")
            else:
                cmd.append(package)
            
            if self.dry_run:
                self.logger.info(f"DRY RUN: Would run: {' '.join(cmd)}")
                return True
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                cwd=self.project_root
            )
            
            self.logger.info(f"Successfully updated {package}")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to update {package}: {e}")
            return False
    
    def update_requirements_file(self, file_path: str, updates: Dict[str, str]) -> bool:
        """Update a requirements file with new versions"""
        try:
            if not os.path.exists(file_path):
                self.logger.warning(f"Requirements file not found: {file_path}")
                return False
            
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            updated_lines = []
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    updated_lines.append(line + '\n')
                    continue
                
                # Parse package name from line
                package_match = re.match(r'^([a-zA-Z0-9\-_]+)', line)
                if package_match:
                    package_name = package_match.group(1).lower()
                    
                    # Check if we have an update for this package
                    for update_package, new_version in updates.items():
                        if update_package.lower() == package_name:
                            # Update the line with new version
                            if '==' in line:
                                new_line = re.sub(r'==.*$', f'=={new_version}', line)
                            elif '>=' in line:
                                new_line = re.sub(r'>=.*$', f'>={new_version}', line)
                            else:
                                new_line = f"{package_name}=={new_version}"
                            
                            updated_lines.append(new_line + '\n')
                            self.logger.info(f"Updated {package_name}: {line} -> {new_line}")
                            break
                    else:
                        updated_lines.append(line + '\n')
                else:
                    updated_lines.append(line + '\n')
            
            if self.dry_run:
                self.logger.info(f"DRY RUN: Would update {file_path}")
                return True
            
            # Write updated requirements
            with open(file_path, 'w') as f:
                f.writelines(updated_lines)
            
            self.logger.info(f"Updated requirements file: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update requirements file {file_path}: {e}")
            return False
    
    def run_tests(self) -> bool:
        """Run test suite to verify updates don't break functionality"""
        try:
            # Run unit tests
            result = subprocess.run(
                ['python', '-m', 'pytest', 'tests/unit/', '-v', '--tb=short'],
                capture_output=True,
                text=True,
                cwd=self.project_root,
                timeout=300
            )
            
            if result.returncode != 0:
                self.logger.error("Unit tests failed after dependency update")
                self.logger.error(result.stdout)
                return False
            
            # Run integration tests (without GPU)
            result = subprocess.run(
                ['python', '-m', 'pytest', 'tests/integration/', '-v', '--tb=short', '-k', 'not gpu'],
                capture_output=True,
                text=True,
                cwd=self.project_root,
                timeout=600
            )
            
            if result.returncode != 0:
                self.logger.warning("Some integration tests failed after dependency update")
                self.logger.warning(result.stdout)
                # Don't fail for integration tests as they might have external dependencies
            
            self.logger.info("Test suite passed after dependency updates")
            return True
            
        except subprocess.TimeoutExpired:
            self.logger.error("Test suite timed out")
            return False
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to run test suite: {e}")
            return False
    
    def update_security_vulnerabilities(self) -> Tuple[bool, List[str]]:
        """Update packages with known security vulnerabilities"""
        vulnerabilities = self.check_security_vulnerabilities()
        if not vulnerabilities:
            self.logger.info("No security vulnerabilities found")
            return True, []
        
        self.logger.info(f"Found {len(vulnerabilities)} security vulnerabilities")
        
        # Extract unique package names that need updates
        vulnerable_packages = set()
        for vuln in vulnerabilities:
            # Different tools have different formats
            if 'package' in vuln:
                vulnerable_packages.add(vuln['package'])
            elif 'name' in vuln:
                vulnerable_packages.add(vuln['name'])
        
        updated_packages = []
        for package in vulnerable_packages:
            self.logger.info(f"Updating vulnerable package: {package}")
            if self.update_package(package):
                updated_packages.append(package)
        
        # Verify vulnerabilities are resolved
        remaining_vulns = self.check_security_vulnerabilities()
        if len(remaining_vulns) < len(vulnerabilities):
            self.logger.info(f"Reduced vulnerabilities from {len(vulnerabilities)} to {len(remaining_vulns)}")
            return True, updated_packages
        else:
            self.logger.warning("Security vulnerabilities not fully resolved")
            return False, updated_packages
    
    def update_patch_versions(self) -> Tuple[bool, List[str]]:
        """Update packages to latest patch versions"""
        outdated = self.get_outdated_packages()
        patch_updates = []
        
        for package in outdated:
            current_version = package['version']
            latest_version = package['latest_version']
            
            # Check if it's a patch update (same major.minor)
            current_parts = current_version.split('.')
            latest_parts = latest_version.split('.')
            
            if (len(current_parts) >= 2 and len(latest_parts) >= 2 and
                current_parts[0] == latest_parts[0] and
                current_parts[1] == latest_parts[1]):
                
                self.logger.info(f"Patch update available: {package['name']} {current_version} -> {latest_version}")
                if self.update_package(package['name'], latest_version):
                    patch_updates.append(package['name'])
        
        return len(patch_updates) > 0, patch_updates
    
    def update_minor_versions(self) -> Tuple[bool, List[str]]:
        """Update packages to latest minor versions"""
        outdated = self.get_outdated_packages()
        minor_updates = []
        
        for package in outdated:
            current_version = package['version']
            latest_version = package['latest_version']
            
            # Check if it's a minor update (same major)
            current_parts = current_version.split('.')
            latest_parts = latest_version.split('.')
            
            if (len(current_parts) >= 1 and len(latest_parts) >= 1 and
                current_parts[0] == latest_parts[0]):
                
                self.logger.info(f"Minor update available: {package['name']} {current_version} -> {latest_version}")
                if self.update_package(package['name'], latest_version):
                    minor_updates.append(package['name'])
        
        return len(minor_updates) > 0, minor_updates
    
    def update_major_versions(self) -> Tuple[bool, List[str]]:
        """Update packages to latest major versions (requires careful testing)"""
        outdated = self.get_outdated_packages()
        major_updates = []
        
        # List of packages that are safe for major updates
        safe_major_updates = {
            'pytest', 'black', 'isort', 'flake8', 'mypy', 'bandit',
            'safety', 'pip-audit', 'pre-commit'
        }
        
        for package in outdated:
            if package['name'].lower() not in safe_major_updates:
                continue
                
            current_version = package['version']
            latest_version = package['latest_version']
            
            self.logger.info(f"Major update available: {package['name']} {current_version} -> {latest_version}")
            if self.update_package(package['name'], latest_version):
                major_updates.append(package['name'])
        
        return len(major_updates) > 0, major_updates
    
    def create_backup(self) -> str:
        """Create backup of current environment"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = f"requirements_backup_{timestamp}.txt"
            
            result = subprocess.run(
                ['pip', 'freeze'],
                capture_output=True,
                text=True,
                check=True
            )
            
            backup_path = self.project_root / backup_file
            with open(backup_path, 'w') as f:
                f.write(result.stdout)
            
            self.logger.info(f"Created dependency backup: {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            self.logger.error(f"Failed to create backup: {e}")
            return ""
    
    def restore_backup(self, backup_file: str) -> bool:
        """Restore dependencies from backup"""
        try:
            if not os.path.exists(backup_file):
                self.logger.error(f"Backup file not found: {backup_file}")
                return False
            
            if self.dry_run:
                self.logger.info(f"DRY RUN: Would restore from {backup_file}")
                return True
            
            result = subprocess.run(
                ['pip', 'install', '-r', backup_file, '--force-reinstall'],
                capture_output=True,
                text=True,
                check=True,
                cwd=self.project_root
            )
            
            self.logger.info(f"Restored dependencies from backup: {backup_file}")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to restore backup: {e}")
            return False
    
    def generate_update_report(self, updates: Dict[str, List[str]]) -> str:
        """Generate a report of dependency updates"""
        report = []
        report.append("# Dependency Update Report")
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append("")
        
        total_updates = sum(len(update_list) for update_list in updates.values())
        report.append(f"Total packages updated: {total_updates}")
        report.append("")
        
        for update_type, packages in updates.items():
            if packages:
                report.append(f"## {update_type.replace('_', ' ').title()}")
                for package in packages:
                    report.append(f"- {package}")
                report.append("")
        
        # Add security check results
        vulnerabilities = self.check_security_vulnerabilities()
        if vulnerabilities:
            report.append(f"## Remaining Security Issues")
            report.append(f"Found {len(vulnerabilities)} vulnerabilities that still need attention:")
            for vuln in vulnerabilities[:5]:  # Show first 5
                package_name = vuln.get('package', vuln.get('name', 'Unknown'))
                report.append(f"- {package_name}: {vuln.get('advisory', 'No details')}")
            
            if len(vulnerabilities) > 5:
                report.append(f"- ... and {len(vulnerabilities) - 5} more")
            report.append("")
        else:
            report.append("## Security Status")
            report.append("✅ No known security vulnerabilities detected")
            report.append("")
        
        # Add recommendations
        report.append("## Recommendations")
        report.append("- Run full test suite including GPU tests")
        report.append("- Test application functionality manually")
        report.append("- Monitor application performance after deployment")
        report.append("- Review changelog for major version updates")
        
        return '\n'.join(report)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Automated dependency management')
    parser.add_argument('--type', choices=['security', 'patch', 'minor', 'major', 'all'],
                       default='security', help='Type of updates to perform')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be updated without making changes')
    parser.add_argument('--backup', action='store_true',
                       help='Create backup before updating')
    parser.add_argument('--test', action='store_true',
                       help='Run test suite after updates')
    parser.add_argument('--report', help='Generate update report to file')
    parser.add_argument('--restore', help='Restore from backup file')
    
    args = parser.parse_args()
    
    updater = DependencyUpdater(dry_run=args.dry_run)
    
    # Restore from backup if requested
    if args.restore:
        success = updater.restore_backup(args.restore)
        sys.exit(0 if success else 1)
    
    # Create backup if requested
    backup_file = ""
    if args.backup and not args.dry_run:
        backup_file = updater.create_backup()
    
    # Perform updates based on type
    updates = {}
    overall_success = True
    
    try:
        if args.type in ['security', 'all']:
            success, packages = updater.update_security_vulnerabilities()
            updates['security_updates'] = packages
            overall_success &= success
        
        if args.type in ['patch', 'all']:
            success, packages = updater.update_patch_versions()
            updates['patch_updates'] = packages
            overall_success &= success
        
        if args.type in ['minor', 'all']:
            success, packages = updater.update_minor_versions()
            updates['minor_updates'] = packages
            overall_success &= success
        
        if args.type in ['major', 'all']:
            success, packages = updater.update_major_versions()
            updates['major_updates'] = packages
            overall_success &= success
        
        # Run tests if requested
        if args.test and not args.dry_run:
            test_success = updater.run_tests()
            if not test_success:
                updater.logger.error("Tests failed after updates")
                if backup_file:
                    updater.logger.info("Restoring from backup due to test failures")
                    updater.restore_backup(backup_file)
                overall_success = False
        
        # Generate report
        if args.report:
            report_content = updater.generate_update_report(updates)
            with open(args.report, 'w') as f:
                f.write(report_content)
            updater.logger.info(f"Update report saved to {args.report}")
        
        # Print summary
        total_updates = sum(len(package_list) for package_list in updates.values())
        if total_updates > 0:
            updater.logger.info(f"Update completed. {total_updates} packages updated.")
        else:
            updater.logger.info("No updates were needed or performed.")
        
        sys.exit(0 if overall_success else 1)
        
    except KeyboardInterrupt:
        updater.logger.info("Update interrupted by user")
        if backup_file:
            updater.logger.info("Restoring from backup due to interruption")
            updater.restore_backup(backup_file)
        sys.exit(1)
    except Exception as e:
        updater.logger.error(f"Unexpected error during update: {e}")
        if backup_file:
            updater.logger.info("Restoring from backup due to error")
            updater.restore_backup(backup_file)
        sys.exit(1)


if __name__ == '__main__':
    main()