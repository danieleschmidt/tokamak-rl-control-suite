"""
Tokamak RL Control Suite

Open-source reinforcement learning plasma-shape controllers for compact tokamaks.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragonlabs.io"

from tokamak_rl.environment import make_tokamak_env, TokamakEnv
from tokamak_rl.physics import TokamakConfig, GradShafranovSolver
from tokamak_rl.agents import SACAgent, DreamerAgent
from tokamak_rl.safety import SafetyShield, DisruptionPredictor
from tokamak_rl.monitoring import create_monitoring_system, PlasmaMonitor
from tokamak_rl.business import create_business_system, PlasmaOptimizer, ScenarioPlanner, PerformanceAnalyzer
from tokamak_rl.analytics import create_analytics_system, AnomalyDetector, PerformancePredictor, TrendAnalyzer
from tokamak_rl.dashboard import create_dashboard_system, WebDashboard, RealTimePlotter
from tokamak_rl.database import create_data_repository, ExperimentDatabase

__all__ = [
    "make_tokamak_env",
    "TokamakEnv", 
    "TokamakConfig",
    "GradShafranovSolver",
    "SACAgent",
    "DreamerAgent",
    "SafetyShield",
    "DisruptionPredictor",
    "create_monitoring_system",
    "PlasmaMonitor",
    "create_business_system",
    "PlasmaOptimizer",
    "ScenarioPlanner", 
    "PerformanceAnalyzer",
    "create_analytics_system",
    "AnomalyDetector",
    "PerformancePredictor",
    "TrendAnalyzer",
    "create_dashboard_system",
    "WebDashboard",
    "RealTimePlotter",
    "create_data_repository",
    "ExperimentDatabase",
    "__version__"
]