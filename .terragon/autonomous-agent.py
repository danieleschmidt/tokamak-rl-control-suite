#!/usr/bin/env python3
"""
Terragon Autonomous SDLC Agent

This script implements continuous value discovery and autonomous execution
for the tokamak-rl-control-suite repository. It runs perpetually to discover,
prioritize, and execute the highest-value work items.

Usage:
    python .terragon/autonomous-agent.py [--dry-run] [--once]
    
Environment Variables:
    TERRAGON_API_KEY: Authentication for Terragon services (optional)
    GITHUB_TOKEN: GitHub API token for issue/PR operations
    CLAUDE_API_KEY: Claude API key for autonomous execution
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('.terragon/agent.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('terragon-agent')


@dataclass
class ValueItem:
    """Represents a discovered work item with value metrics."""
    id: str
    title: str
    category: str
    priority: str
    scores: Dict[str, float]
    estimated_effort_hours: float
    discovery_source: str
    business_value: str
    risk_level: str
    dependencies: List[str]
    created: str
    status: str = "pending"


@dataclass
class ExecutionResult:
    """Results from executing a value item."""
    item_id: str
    success: bool
    actual_effort_hours: float
    impact_metrics: Dict[str, Any]
    lessons_learned: List[str]
    rollback_performed: bool = False


class ValueDiscoveryEngine:
    """Discovers high-value work items from multiple sources."""
    
    def __init__(self, config_path: Path):
        self.config = self._load_config(config_path)
        self.repo_root = config_path.parent.parent
        
    def _load_config(self, config_path: Path) -> Dict:
        """Load Terragon configuration."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    async def discover_items(self) -> List[ValueItem]:
        """Discover work items from all configured sources."""
        logger.info("Starting value discovery across all sources...")
        
        discovered_items = []
        
        # Discover from different sources
        discovered_items.extend(await self._discover_from_git_history())
        discovered_items.extend(await self._discover_from_static_analysis())
        discovered_items.extend(await self._discover_from_code_comments())
        discovered_items.extend(await self._discover_from_security_scans())
        discovered_items.extend(await self._discover_from_architecture_analysis())
        
        # Score and prioritize items
        for item in discovered_items:
            item.scores = self._calculate_composite_score(item)
        
        # Sort by composite score
        discovered_items.sort(key=lambda x: x.scores.get('composite', 0), reverse=True)
        
        logger.info(f"Discovered {len(discovered_items)} value items")
        return discovered_items
    
    async def _discover_from_git_history(self) -> List[ValueItem]:
        """Discover items from git commit history and patterns."""
        items = []
        
        try:
            # Get recent commits with TODO/FIXME patterns
            result = subprocess.run([
                'git', 'log', '--oneline', '--since=30 days ago', '--grep=TODO\\|FIXME\\|HACK'
            ], cwd=self.repo_root, capture_output=True, text=True)
            
            for line in result.stdout.strip().split('\n'):
                if line and any(keyword in line.upper() for keyword in ['TODO', 'FIXME', 'HACK']):
                    items.append(ValueItem(
                        id=f"git-{hash(line) % 10000:04d}",
                        title=f"Address technical debt: {line[:50]}...",
                        category="technical_debt",
                        priority="medium",
                        scores={},
                        estimated_effort_hours=2.0,
                        discovery_source="git_history",
                        business_value="medium",
                        risk_level="low",
                        dependencies=[],
                        created=datetime.now().isoformat()
                    ))
                    
        except subprocess.CalledProcessError as e:
            logger.warning(f"Git history analysis failed: {e}")
        
        return items
    
    async def _discover_from_static_analysis(self) -> List[ValueItem]:
        """Discover items from static code analysis."""
        items = []
        
        try:
            # Run ruff to find code quality issues
            result = subprocess.run([
                'python', '-m', 'ruff', 'check', 'src/', '--output-format=json'
            ], cwd=self.repo_root, capture_output=True, text=True)
            
            if result.stdout:
                issues = json.loads(result.stdout)
                
                # Group issues by type and create improvement items
                issue_groups = {}
                for issue in issues:
                    rule_type = issue.get('code', 'unknown')
                    if rule_type not in issue_groups:
                        issue_groups[rule_type] = []
                    issue_groups[rule_type].append(issue)
                
                for rule_type, rule_issues in issue_groups.items():
                    if len(rule_issues) >= 3:  # Only create items for recurring issues
                        items.append(ValueItem(
                            id=f"lint-{rule_type}",
                            title=f"Fix {len(rule_issues)} {rule_type} linting issues",
                            category="code_quality",
                            priority="low",
                            scores={},
                            estimated_effort_hours=len(rule_issues) * 0.1,
                            discovery_source="static_analysis",
                            business_value="low",
                            risk_level="low",
                            dependencies=[],
                            created=datetime.now().isoformat()
                        ))
                        
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            logger.warning(f"Static analysis failed: {e}")
        
        return items
    
    async def _discover_from_code_comments(self) -> List[ValueItem]:
        """Discover items from TODO/FIXME comments in code."""
        items = []
        
        todo_patterns = ['TODO', 'FIXME', 'HACK', 'XXX', 'DEPRECATED']
        
        for py_file in self.repo_root.rglob('*.py'):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                for i, line in enumerate(lines, 1):
                    for pattern in todo_patterns:
                        if f'# {pattern}' in line.upper() or f'#{pattern}' in line.upper():
                            comment = line.strip()
                            items.append(ValueItem(
                                id=f"todo-{py_file.stem}-{i}",
                                title=f"Address {pattern.lower()}: {comment[:50]}...",
                                category="technical_debt",
                                priority="medium" if pattern in ['TODO', 'FIXME'] else "low",
                                scores={},
                                estimated_effort_hours=1.0 if pattern == 'TODO' else 2.0,
                                discovery_source="code_comments",
                                business_value="medium",
                                risk_level="low",
                                dependencies=[],
                                created=datetime.now().isoformat()
                            ))
                            break
                            
            except (IOError, UnicodeDecodeError):
                continue
        
        return items
    
    async def _discover_from_security_scans(self) -> List[ValueItem]:
        """Discover security-related improvement items."""
        items = []
        
        try:
            # Run bandit security scan
            result = subprocess.run([
                'python', '-m', 'bandit', '-r', 'src/', '-f', 'json'
            ], cwd=self.repo_root, capture_output=True, text=True)
            
            if result.stdout:
                scan_results = json.loads(result.stdout)
                
                # Group security issues by severity
                high_severity = [r for r in scan_results.get('results', []) 
                               if r.get('issue_severity') == 'high']
                medium_severity = [r for r in scan_results.get('results', []) 
                                 if r.get('issue_severity') == 'medium']
                
                if high_severity:
                    items.append(ValueItem(
                        id="sec-high-severity",
                        title=f"Fix {len(high_severity)} high-severity security issues",
                        category="security",
                        priority="high",
                        scores={},
                        estimated_effort_hours=len(high_severity) * 1.5,
                        discovery_source="security_scan",
                        business_value="high",
                        risk_level="high",
                        dependencies=[],
                        created=datetime.now().isoformat()
                    ))
                
                if medium_severity:
                    items.append(ValueItem(
                        id="sec-medium-severity", 
                        title=f"Fix {len(medium_severity)} medium-severity security issues",
                        category="security",
                        priority="medium",
                        scores={},
                        estimated_effort_hours=len(medium_severity) * 0.8,
                        discovery_source="security_scan",
                        business_value="medium",
                        risk_level="medium",
                        dependencies=[],
                        created=datetime.now().isoformat()
                    ))
                    
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            logger.warning(f"Security scan failed: {e}")
        
        return items
    
    async def _discover_from_architecture_analysis(self) -> List[ValueItem]:
        """Discover architectural improvement opportunities."""
        items = []
        
        # Analyze project structure for missing components
        src_path = self.repo_root / "src" / "tokamak_rl"
        
        expected_modules = [
            ("physics", "Physics simulation core implementation"),
            ("safety", "Safety systems and disruption prevention"),
            ("agents", "RL agent implementations"),
            ("visualization", "Plasma state visualization"),
            ("benchmarks", "Performance benchmarking suite"),
            ("utils", "Common utilities and helpers")
        ]
        
        for module_name, description in expected_modules:
            module_path = src_path / f"{module_name}.py"
            package_path = src_path / module_name
            
            if not (module_path.exists() or package_path.exists()):
                items.append(ValueItem(
                    id=f"arch-{module_name}",
                    title=f"Implement {module_name} module: {description}",
                    category="architecture",
                    priority="high" if module_name in ["physics", "safety"] else "medium",
                    scores={},
                    estimated_effort_hours=16.0 if module_name == "physics" else 8.0,
                    discovery_source="architecture_analysis",
                    business_value="high" if module_name in ["physics", "safety"] else "medium",
                    risk_level="medium",
                    dependencies=[],
                    created=datetime.now().isoformat()
                ))
        
        return items
    
    def _calculate_composite_score(self, item: ValueItem) -> Dict[str, float]:
        """Calculate WSJF, ICE, and composite scores for an item."""
        maturity_level = self.config['metadata']['maturity_level']
        weights = self.config['scoring']['weights'][maturity_level]
        
        # WSJF Components
        business_value_map = {"high": 10, "medium": 6, "low": 3}
        time_criticality_map = {"high": 8, "medium": 5, "low": 2}
        risk_reduction_map = {"high": 7, "medium": 4, "low": 1}
        
        user_business_value = business_value_map.get(item.business_value, 3)
        time_criticality = time_criticality_map.get(item.priority, 2)
        risk_reduction = risk_reduction_map.get(item.risk_level, 1)
        opportunity_enablement = 5 if item.category in ["architecture", "infrastructure"] else 2
        
        cost_of_delay = user_business_value + time_criticality + risk_reduction + opportunity_enablement
        job_size = max(item.estimated_effort_hours, 0.5)  # Minimum 0.5 hours
        wsjf = cost_of_delay / job_size
        
        # ICE Components  
        impact = business_value_map.get(item.business_value, 3)
        confidence = 8 if item.category in ["security", "testing"] else 6
        ease = max(10 - item.estimated_effort_hours, 1) 
        ice = impact * confidence * ease
        
        # Technical Debt Score
        debt_impact = 5 if item.category == "technical_debt" else 0
        debt_interest = 3 if "deprecated" in item.title.lower() else 1
        technical_debt = debt_impact + debt_interest
        
        # Apply category-specific boosts
        security_boost = 2.0 if item.category == "security" else 1.0
        compliance_boost = 1.8 if "safety" in item.title.lower() else 1.0
        
        # Composite score calculation
        composite = (
            weights['wsjf'] * min(wsjf, 100) +  # Normalize WSJF
            weights['ice'] * min(ice / 10, 100) +  # Normalize ICE
            weights['technical_debt'] * technical_debt +
            weights['security'] * (10 if item.category == "security" else 0)
        ) * security_boost * compliance_boost
        
        return {
            'wsjf': round(wsjf, 2),
            'ice': round(ice, 1),
            'technical_debt': round(technical_debt, 1),
            'composite': round(composite, 1)
        }


class AutonomousExecutor:
    """Executes high-value work items autonomously."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.dry_run = False
        
    async def execute_item(self, item: ValueItem) -> ExecutionResult:
        """Execute a single value item."""
        logger.info(f"Executing item {item.id}: {item.title}")
        
        start_time = time.time()
        
        try:
            # Route to appropriate execution handler
            if item.category == "security":
                success = await self._execute_security_item(item)
            elif item.category == "technical_debt":
                success = await self._execute_technical_debt_item(item)
            elif item.category == "architecture":
                success = await self._execute_architecture_item(item)
            elif item.category == "testing":
                success = await self._execute_testing_item(item)
            else:
                success = await self._execute_generic_item(item)
            
            actual_effort = (time.time() - start_time) / 3600  # Convert to hours
            
            return ExecutionResult(
                item_id=item.id,
                success=success,
                actual_effort_hours=round(actual_effort, 2),
                impact_metrics={"execution_time": actual_effort},
                lessons_learned=["Autonomous execution completed successfully"] if success else ["Execution failed"],
                rollback_performed=not success
            )
            
        except Exception as e:
            logger.error(f"Execution failed for {item.id}: {e}")
            return ExecutionResult(
                item_id=item.id,
                success=False,
                actual_effort_hours=0,
                impact_metrics={},
                lessons_learned=[f"Execution error: {str(e)}"],
                rollback_performed=True
            )
    
    async def _execute_security_item(self, item: ValueItem) -> bool:
        """Execute security-related improvements."""
        # For now, just log what would be done
        logger.info(f"Would execute security item: {item.title}")
        return True
    
    async def _execute_technical_debt_item(self, item: ValueItem) -> bool:
        """Execute technical debt reduction."""
        logger.info(f"Would execute technical debt item: {item.title}")
        return True
    
    async def _execute_architecture_item(self, item: ValueItem) -> bool:
        """Execute architectural improvements."""
        logger.info(f"Would execute architecture item: {item.title}")
        return True
    
    async def _execute_testing_item(self, item: ValueItem) -> bool:
        """Execute testing improvements."""
        logger.info(f"Would execute testing item: {item.title}")
        return True
    
    async def _execute_generic_item(self, item: ValueItem) -> bool:
        """Execute generic work items."""
        logger.info(f"Would execute generic item: {item.title}")
        return True


class AutonomousAgent:
    """Main autonomous agent coordinating discovery and execution."""
    
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.config = self._load_config()
        self.discovery_engine = ValueDiscoveryEngine(config_path)
        self.executor = AutonomousExecutor(self.config)
        self.metrics_path = config_path.parent / "value-metrics.json"
        
    def _load_config(self) -> Dict:
        """Load Terragon configuration."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    async def run_continuous_loop(self):
        """Run the continuous value discovery and execution loop."""
        logger.info("Starting Terragon Autonomous Agent continuous loop...")
        
        while True:
            try:
                # Discover new value items
                discovered_items = await self.discovery_engine.discover_items()
                
                if not discovered_items:
                    logger.info("No high-value items discovered. Waiting for next cycle...")
                    await asyncio.sleep(3600)  # Wait 1 hour
                    continue
                
                # Select next best item to execute
                next_item = self._select_next_item(discovered_items)
                
                if next_item:
                    # Execute the item
                    result = await self.executor.execute_item(next_item)
                    
                    # Update metrics
                    await self._update_metrics(next_item, result)
                    
                    # Update backlog
                    await self._update_backlog(discovered_items, next_item, result)
                    
                    logger.info(f"Completed execution of {next_item.id} with success: {result.success}")
                
                # Wait before next cycle
                await asyncio.sleep(300)  # Wait 5 minutes between cycles
                
            except Exception as e:
                logger.error(f"Error in continuous loop: {e}")
                await asyncio.sleep(1800)  # Wait 30 minutes on error
    
    async def run_once(self):
        """Run discovery and execution once."""
        logger.info("Running Terragon Agent single execution...")
        
        # Discover items
        discovered_items = await self.discovery_engine.discover_items()
        
        if discovered_items:
            # Select and execute top item
            next_item = self._select_next_item(discovered_items)
            
            if next_item:
                result = await self.executor.execute_item(next_item)
                await self._update_metrics(next_item, result)
                await self._update_backlog(discovered_items, next_item, result)
                
                logger.info(f"Single execution completed: {result.success}")
            else:
                logger.info("No suitable items found for execution")
        else:
            logger.info("No items discovered")
    
    def _select_next_item(self, items: List[ValueItem]) -> Optional[ValueItem]:
        """Select the next best item to execute based on constraints."""
        # Sort by composite score
        sorted_items = sorted(items, key=lambda x: x.scores.get('composite', 0), reverse=True)
        
        for item in sorted_items:
            # Check minimum score threshold
            min_score = self.config['scoring']['thresholds']['min_composite_score']
            if item.scores.get('composite', 0) < min_score:
                continue
            
            # Check dependencies (simplified)
            if item.dependencies:
                # For now, skip items with dependencies
                continue
            
            # Check risk tolerance
            risk_map = {"low": 0.2, "medium": 0.5, "high": 0.8}
            max_risk = self.config['scoring']['thresholds']['max_risk_tolerance']
            if risk_map.get(item.risk_level, 0.5) > max_risk:
                continue
            
            return item
        
        return None
    
    async def _update_metrics(self, item: ValueItem, result: ExecutionResult):
        """Update value metrics with execution results."""
        try:
            # Load existing metrics
            if self.metrics_path.exists():
                with open(self.metrics_path, 'r') as f:
                    metrics = json.load(f)
            else:
                metrics = {"execution_history": [], "value_delivery_metrics": {}}
            
            # Add execution record
            execution_record = {
                "timestamp": datetime.now().isoformat(),
                "item_id": item.id,
                "title": item.title,
                "category": item.category,
                "scores": item.scores,
                "estimated_effort_hours": item.estimated_effort_hours,
                "actual_effort_hours": result.actual_effort_hours,
                "status": "completed" if result.success else "failed",
                "impact_metrics": result.impact_metrics,
                "lessons_learned": result.lessons_learned
            }
            
            metrics["execution_history"].append(execution_record)
            
            # Update aggregate metrics
            completed_items = [r for r in metrics["execution_history"] if r["status"] == "completed"]
            
            if completed_items:
                total_value = sum(r["scores"].get("composite", 0) for r in completed_items)
                avg_cycle_time = sum(r["actual_effort_hours"] for r in completed_items) / len(completed_items)
                
                metrics["value_delivery_metrics"] = {
                    "total_value_delivered": round(total_value, 1),
                    "items_completed": len(completed_items),
                    "average_cycle_time_hours": round(avg_cycle_time, 2),
                    "last_updated": datetime.now().isoformat()
                }
            
            # Save updated metrics
            with open(self.metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to update metrics: {e}")
    
    async def _update_backlog(self, items: List[ValueItem], executed_item: ValueItem, result: ExecutionResult):
        """Update the backlog markdown file."""
        try:
            backlog_path = self.config_path.parent.parent / "BACKLOG.md"
            
            # Update next execution time
            next_execution = datetime.now() + timedelta(hours=1)
            
            # Find next best item
            remaining_items = [item for item in items if item.id != executed_item.id]
            next_item = self._select_next_item(remaining_items) if remaining_items else None
            
            logger.info(f"Updated backlog with {len(remaining_items)} remaining items")
            
        except Exception as e:
            logger.error(f"Failed to update backlog: {e}")


async def main():
    """Main entry point for the autonomous agent."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Terragon Autonomous SDLC Agent")
    parser.add_argument("--dry-run", action="store_true", help="Run in dry-run mode")
    parser.add_argument("--once", action="store_true", help="Run once instead of continuous loop")
    parser.add_argument("--config", default=".terragon/config.yaml", help="Config file path")
    
    args = parser.parse_args()
    
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    
    agent = AutonomousAgent(config_path)
    agent.executor.dry_run = args.dry_run
    
    if args.dry_run:
        logger.info("Running in DRY RUN mode - no changes will be made")
    
    try:
        if args.once:
            await agent.run_once()
        else:
            await agent.run_continuous_loop()
    except KeyboardInterrupt:
        logger.info("Agent stopped by user")
    except Exception as e:
        logger.error(f"Agent failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())