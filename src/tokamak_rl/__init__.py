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

__all__ = [
    "make_tokamak_env",
    "TokamakEnv", 
    "TokamakConfig",
    "GradShafranovSolver",
    "SACAgent",
    "DreamerAgent",
    "SafetyShield",
    "DisruptionPredictor",
    "__version__"
]