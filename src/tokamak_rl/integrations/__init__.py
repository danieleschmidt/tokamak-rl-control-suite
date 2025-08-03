"""
External integrations for tokamak RL control suite.

This module provides integrations with external services and systems
for data sharing, notifications, and workflow automation.
"""

from .github import GitHubIntegration
from .notifications import NotificationService
from .auth import AuthenticationService

__all__ = [
    "GitHubIntegration",
    "NotificationService", 
    "AuthenticationService"
]