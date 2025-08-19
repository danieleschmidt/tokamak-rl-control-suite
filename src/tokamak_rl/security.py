"""
Security framework for tokamak RL control systems.

This module implements security measures, access controls, and input sanitization
to protect against malicious inputs and ensure safe operation.
"""

import hashlib
import hmac
import secrets
import time
from typing import Any, Dict, List, Optional, Callable, Set, Tuple
import logging
from dataclasses import dataclass, field
from enum import Enum
import json
import re

# Setup module logger
logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security clearance levels for system access."""
    READONLY = "readonly"
    OPERATOR = "operator" 
    ENGINEER = "engineer"
    ADMIN = "admin"
    SAFETY_OVERRIDE = "safety_override"


class ThreatLevel(Enum):
    """Threat assessment levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityEvent:
    """Security event for logging and analysis."""
    timestamp: float
    event_type: str
    threat_level: ThreatLevel
    source_ip: Optional[str] = None
    user_id: Optional[str] = None
    description: str = ""
    action_taken: str = ""
    additional_data: Dict[str, Any] = field(default_factory=dict)


class SecurityAuditLog:
    """Security audit logging system."""
    
    def __init__(self, max_entries: int = 10000):
        self.events: List[SecurityEvent] = []
        self.max_entries = max_entries
        self._lock = False  # Simple lock mechanism
        
    def log_event(self, event: SecurityEvent) -> None:
        """Log a security event."""
        if self._lock:
            return
            
        try:
            self._lock = True
            self.events.append(event)
            
            # Maintain size limit
            if len(self.events) > self.max_entries:
                self.events.pop(0)
                
            # Log to system logger
            logger.warning(f"Security Event [{event.threat_level.value.upper()}]: {event.description}")
            
        finally:
            self._lock = False
    
    def get_events_by_threat_level(self, threat_level: ThreatLevel, 
                                  last_n_hours: float = 24.0) -> List[SecurityEvent]:
        """Get events by threat level within time window."""
        cutoff_time = time.time() - (last_n_hours * 3600)
        
        return [
            event for event in self.events 
            if event.threat_level == threat_level and event.timestamp >= cutoff_time
        ]
    
    def get_threat_summary(self, last_n_hours: float = 24.0) -> Dict[str, int]:
        """Get summary of threats in time window."""
        cutoff_time = time.time() - (last_n_hours * 3600)
        recent_events = [e for e in self.events if e.timestamp >= cutoff_time]
        
        summary = {level.value: 0 for level in ThreatLevel}
        for event in recent_events:
            summary[event.threat_level.value] += 1
            
        return summary


class InputSanitizer:
    """Comprehensive input sanitization for security."""
    
    def __init__(self):
        self.audit_log = SecurityAuditLog()
        
        # Dangerous patterns to detect
        self.malicious_patterns = [
            r'(__import__|exec|eval|compile)',  # Python code execution
            r'(system|popen|subprocess)',       # System commands
            r'(rm\s+-rf|del\s+/)',             # Destructive commands
            r'(\.\./|\.\.\\)',                 # Directory traversal
            r'(<script|javascript:)',          # Script injection
            r'(DROP|DELETE|INSERT|UPDATE).*TABLE',  # SQL injection
        ]
        
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.malicious_patterns]
    
    def sanitize_string(self, input_str: str, field_name: str = "input",
                       max_length: int = 1000, allow_special_chars: bool = False) -> str:
        """Sanitize string input against various attacks."""
        
        if not isinstance(input_str, str):
            self._log_security_event(
                ThreatLevel.MEDIUM,
                f"Non-string input received for {field_name}: {type(input_str)}"
            )
            return str(input_str)[:max_length]
        
        # Check length
        if len(input_str) > max_length:
            self._log_security_event(
                ThreatLevel.LOW,
                f"Oversized input truncated for {field_name}: {len(input_str)} chars"
            )
            input_str = input_str[:max_length]
        
        # Check for malicious patterns
        for pattern in self.compiled_patterns:
            if pattern.search(input_str):
                self._log_security_event(
                    ThreatLevel.HIGH,
                    f"Malicious pattern detected in {field_name}: {pattern.pattern}"
                )
                # Remove the malicious content
                input_str = pattern.sub("[SANITIZED]", input_str)
        
        # Remove control characters except basic whitespace
        sanitized = ''.join(char for char in input_str 
                          if ord(char) >= 32 or char in '\t\n\r')
        
        # Optionally remove special characters
        if not allow_special_chars:
            # Keep alphanumeric, spaces, and basic punctuation
            sanitized = re.sub(r'[^a-zA-Z0-9\s\.\-_,()[\]{}]', '', sanitized)
        
        return sanitized
    
    def sanitize_numeric_input(self, value: Any, field_name: str = "numeric_input",
                             min_val: float = None, max_val: float = None) -> float:
        """Sanitize numeric input with bounds checking."""
        
        try:
            # Convert to float
            if isinstance(value, str):
                # Check for obviously malicious numeric strings
                if any(pattern.search(value) for pattern in self.compiled_patterns):
                    self._log_security_event(
                        ThreatLevel.HIGH,
                        f"Malicious numeric string for {field_name}: {value}"
                    )
                    return 0.0
                
                numeric_value = float(value)
            else:
                numeric_value = float(value)
            
            # Check for invalid values
            if not (numeric_value == numeric_value):  # NaN check
                self._log_security_event(
                    ThreatLevel.MEDIUM,
                    f"NaN value provided for {field_name}"
                )
                return 0.0
            
            if abs(numeric_value) == float('inf'):
                self._log_security_event(
                    ThreatLevel.MEDIUM,
                    f"Infinite value provided for {field_name}"
                )
                return 0.0
            
            # Apply bounds
            if min_val is not None and numeric_value < min_val:
                self._log_security_event(
                    ThreatLevel.LOW,
                    f"Value below minimum for {field_name}: {numeric_value} < {min_val}"
                )
                numeric_value = min_val
            
            if max_val is not None and numeric_value > max_val:
                self._log_security_event(
                    ThreatLevel.LOW,
                    f"Value above maximum for {field_name}: {numeric_value} > {max_val}"
                )
                numeric_value = max_val
            
            return numeric_value
            
        except (ValueError, TypeError, OverflowError) as e:
            self._log_security_event(
                ThreatLevel.MEDIUM,
                f"Invalid numeric input for {field_name}: {value} ({e})"
            )
            return 0.0
    
    def sanitize_json_input(self, json_str: str, max_depth: int = 10,
                          max_keys: int = 100) -> Dict[str, Any]:
        """Sanitize JSON input against various attacks."""
        
        if not isinstance(json_str, str):
            self._log_security_event(
                ThreatLevel.MEDIUM,
                f"Non-string JSON input: {type(json_str)}"
            )
            return {}
        
        # Check for malicious patterns in JSON string
        sanitized_json = self.sanitize_string(json_str, "json_input", max_length=10000)
        
        try:
            data = json.loads(sanitized_json)
            
            # Validate structure
            if not isinstance(data, dict):
                self._log_security_event(
                    ThreatLevel.MEDIUM,
                    "JSON input is not a dictionary"
                )
                return {}
            
            # Check depth and key count
            if self._get_dict_depth(data) > max_depth:
                self._log_security_event(
                    ThreatLevel.MEDIUM,
                    f"JSON depth exceeds limit: {max_depth}"
                )
                return {}
            
            if self._count_dict_keys(data) > max_keys:
                self._log_security_event(
                    ThreatLevel.MEDIUM,
                    f"JSON key count exceeds limit: {max_keys}"
                )
                return {}
            
            # Recursively sanitize string values
            return self._sanitize_dict_values(data)
            
        except json.JSONDecodeError as e:
            self._log_security_event(
                ThreatLevel.MEDIUM,
                f"Invalid JSON format: {e}"
            )
            return {}
    
    def _sanitize_dict_values(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively sanitize dictionary values."""
        sanitized = {}
        
        for key, value in data.items():
            # Sanitize key
            clean_key = self.sanitize_string(str(key), "dict_key", max_length=100)
            
            # Sanitize value based on type
            if isinstance(value, str):
                clean_value = self.sanitize_string(value, f"dict_value[{clean_key}]")
            elif isinstance(value, (int, float)):
                clean_value = self.sanitize_numeric_input(value, f"dict_value[{clean_key}]")
            elif isinstance(value, dict):
                clean_value = self._sanitize_dict_values(value)
            elif isinstance(value, list):
                clean_value = [self.sanitize_string(str(item)) if isinstance(item, str) 
                             else self.sanitize_numeric_input(item) if isinstance(item, (int, float))
                             else str(item) for item in value[:100]]  # Limit list size
            else:
                clean_value = self.sanitize_string(str(value), f"dict_value[{clean_key}]")
            
            sanitized[clean_key] = clean_value
        
        return sanitized
    
    def _get_dict_depth(self, d: Dict[str, Any], current_depth: int = 0) -> int:
        """Calculate maximum depth of nested dictionary."""
        if not isinstance(d, dict):
            return current_depth
        
        if not d:
            return current_depth + 1
        
        return max(self._get_dict_depth(v, current_depth + 1) 
                  for v in d.values() if isinstance(v, dict))
    
    def _count_dict_keys(self, d: Dict[str, Any]) -> int:
        """Count total number of keys in nested dictionary."""
        count = len(d)
        for value in d.values():
            if isinstance(value, dict):
                count += self._count_dict_keys(value)
        return count
    
    def _log_security_event(self, threat_level: ThreatLevel, description: str) -> None:
        """Log a security event."""
        event = SecurityEvent(
            timestamp=time.time(),
            event_type="input_sanitization",
            threat_level=threat_level,
            description=description,
            action_taken="sanitized"
        )
        self.audit_log.log_event(event)


class AccessController:
    """Role-based access control for tokamak systems."""
    
    def __init__(self):
        self.audit_log = SecurityAuditLog()
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.failed_attempts: Dict[str, List[float]] = {}
        self.max_failed_attempts = 5
        self.lockout_duration = 3600  # 1 hour
        
        # Define permission matrix
        self.permissions = {
            SecurityLevel.READONLY: {
                'view_data', 'view_status', 'view_diagnostics'
            },
            SecurityLevel.OPERATOR: {
                'view_data', 'view_status', 'view_diagnostics',
                'modify_setpoints', 'start_experiments', 'stop_experiments'
            },
            SecurityLevel.ENGINEER: {
                'view_data', 'view_status', 'view_diagnostics',
                'modify_setpoints', 'start_experiments', 'stop_experiments',
                'modify_config', 'run_diagnostics', 'export_data'
            },
            SecurityLevel.ADMIN: {
                'view_data', 'view_status', 'view_diagnostics',
                'modify_setpoints', 'start_experiments', 'stop_experiments',
                'modify_config', 'run_diagnostics', 'export_data',
                'manage_users', 'system_shutdown', 'emergency_stop'
            },
            SecurityLevel.SAFETY_OVERRIDE: {
                'emergency_stop', 'safety_override', 'system_shutdown'
            }
        }
    
    def authenticate_user(self, user_id: str, token: str, 
                         source_ip: str = None) -> Optional[str]:
        """Authenticate user and return session token."""
        
        # Check for lockout
        if self._is_locked_out(user_id):
            self._log_security_event(
                ThreatLevel.HIGH,
                f"Authentication attempt for locked out user: {user_id}",
                source_ip, user_id
            )
            return None
        
        # Validate token (simplified - in production use proper auth)
        if self._validate_token(user_id, token):
            # Create session
            session_token = secrets.token_urlsafe(32)
            self.active_sessions[session_token] = {
                'user_id': user_id,
                'security_level': self._get_user_security_level(user_id),
                'created_at': time.time(),
                'last_activity': time.time(),
                'source_ip': source_ip
            }
            
            # Clear failed attempts
            if user_id in self.failed_attempts:
                del self.failed_attempts[user_id]
            
            self._log_security_event(
                ThreatLevel.LOW,
                f"Successful authentication: {user_id}",
                source_ip, user_id
            )
            
            return session_token
        else:
            # Record failed attempt
            if user_id not in self.failed_attempts:
                self.failed_attempts[user_id] = []
            self.failed_attempts[user_id].append(time.time())
            
            self._log_security_event(
                ThreatLevel.MEDIUM,
                f"Failed authentication attempt: {user_id}",
                source_ip, user_id
            )
            
            return None
    
    def check_permission(self, session_token: str, action: str) -> bool:
        """Check if session has permission for action."""
        
        if session_token not in self.active_sessions:
            self._log_security_event(
                ThreatLevel.MEDIUM,
                f"Permission check with invalid session token for action: {action}"
            )
            return False
        
        session = self.active_sessions[session_token]
        
        # Check session expiry (24 hours)
        if time.time() - session['created_at'] > 86400:
            self._log_security_event(
                ThreatLevel.LOW,
                f"Expired session attempted action: {action}",
                session.get('source_ip'), session.get('user_id')
            )
            del self.active_sessions[session_token]
            return False
        
        # Update last activity
        session['last_activity'] = time.time()
        
        # Check permission
        security_level = session['security_level']
        allowed_actions = self.permissions.get(security_level, set())
        
        has_permission = action in allowed_actions
        
        if not has_permission:
            self._log_security_event(
                ThreatLevel.MEDIUM,
                f"Unauthorized action attempted: {action} by {session.get('user_id')}",
                session.get('source_ip'), session.get('user_id')
            )
        
        return has_permission
    
    def revoke_session(self, session_token: str) -> None:
        """Revoke a session token."""
        if session_token in self.active_sessions:
            session = self.active_sessions[session_token]
            self._log_security_event(
                ThreatLevel.LOW,
                f"Session revoked for user: {session.get('user_id')}",
                session.get('source_ip'), session.get('user_id')
            )
            del self.active_sessions[session_token]
    
    def _is_locked_out(self, user_id: str) -> bool:
        """Check if user is locked out due to failed attempts."""
        if user_id not in self.failed_attempts:
            return False
        
        recent_failures = [
            timestamp for timestamp in self.failed_attempts[user_id]
            if time.time() - timestamp < self.lockout_duration
        ]
        
        return len(recent_failures) >= self.max_failed_attempts
    
    def _validate_token(self, user_id: str, token: str) -> bool:
        """Validate authentication token (simplified)."""
        # In production, this would check against a secure database
        # For demo purposes, use simple validation
        expected = hashlib.sha256(f"{user_id}_secret_key".encode()).hexdigest()
        return hmac.compare_digest(token, expected)
    
    def _get_user_security_level(self, user_id: str) -> SecurityLevel:
        """Get user security level (simplified)."""
        # In production, this would query a user database
        # For demo purposes, use simple mapping
        level_map = {
            'admin': SecurityLevel.ADMIN,
            'engineer': SecurityLevel.ENGINEER,
            'operator': SecurityLevel.OPERATOR,
            'safety': SecurityLevel.SAFETY_OVERRIDE
        }
        return level_map.get(user_id.lower(), SecurityLevel.READONLY)
    
    def _log_security_event(self, threat_level: ThreatLevel, description: str,
                          source_ip: str = None, user_id: str = None) -> None:
        """Log a security event."""
        event = SecurityEvent(
            timestamp=time.time(),
            event_type="access_control",
            threat_level=threat_level,
            source_ip=source_ip,
            user_id=user_id,
            description=description,
            action_taken="logged"
        )
        self.audit_log.log_event(event)


class SecureConfigManager:
    """Secure configuration management with encryption."""
    
    def __init__(self, encryption_key: Optional[bytes] = None):
        self.encryption_key = encryption_key or self._generate_key()
        self.sanitizer = InputSanitizer()
        self.audit_log = SecurityAuditLog()
    
    def _generate_key(self) -> bytes:
        """Generate encryption key."""
        return secrets.token_bytes(32)
    
    def load_secure_config(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Load and validate configuration securely."""
        
        # Sanitize the configuration
        sanitized_config = self.sanitizer.sanitize_json_input(
            json.dumps(config_data) if isinstance(config_data, dict) else config_data
        )
        
        # Validate critical security settings
        security_validation = self._validate_security_config(sanitized_config)
        
        if not security_validation['is_valid']:
            self._log_security_event(
                ThreatLevel.HIGH,
                f"Invalid security configuration: {security_validation['errors']}"
            )
            # Return minimal safe configuration
            return self._get_safe_default_config()
        
        return sanitized_config
    
    def _validate_security_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate security-related configuration."""
        errors = []
        warnings = []
        
        # Check for required security settings
        required_settings = ['enable_safety', 'safety_factor', 'control_frequency']
        for setting in required_settings:
            if setting not in config:
                errors.append(f"Missing required setting: {setting}")
        
        # Validate safety settings
        if 'safety_factor' in config:
            safety_factor = config['safety_factor']
            if not isinstance(safety_factor, (int, float)) or safety_factor < 1.0:
                errors.append("safety_factor must be >= 1.0")
        
        if 'control_frequency' in config:
            freq = config['control_frequency']
            if not isinstance(freq, (int, float)) or freq < 1 or freq > 1000:
                warnings.append("control_frequency outside recommended range (1-1000 Hz)")
        
        # Check for potentially dangerous settings
        dangerous_keys = ['debug_mode', 'disable_safety', 'raw_access']
        for key in dangerous_keys:
            if config.get(key, False):
                warnings.append(f"Potentially dangerous setting enabled: {key}")
        
        return {
            'is_valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    def _get_safe_default_config(self) -> Dict[str, Any]:
        """Get minimal safe default configuration."""
        return {
            'enable_safety': True,
            'safety_factor': 1.5,
            'control_frequency': 100,
            'emergency_shutdown': True,
            'log_level': 'WARNING'
        }
    
    def _log_security_event(self, threat_level: ThreatLevel, description: str) -> None:
        """Log a security event."""
        event = SecurityEvent(
            timestamp=time.time(),
            event_type="config_management",
            threat_level=threat_level,
            description=description,
            action_taken="config_rejected"
        )
        self.audit_log.log_event(event)


# Global security instances
input_sanitizer = InputSanitizer()
access_controller = AccessController()
config_manager = SecureConfigManager()


def require_permission(action: str):
    """Decorator to require specific permission for function access."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # In a real implementation, this would extract session token
            # from request context or function parameters
            session_token = kwargs.pop('session_token', None)
            
            if not session_token or not access_controller.check_permission(session_token, action):
                raise PermissionError(f"Insufficient permissions for action: {action}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def sanitize_input(sanitizer_func: Callable = None):
    """Decorator to automatically sanitize function inputs."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Apply default sanitization if no custom function provided
            if sanitizer_func:
                args, kwargs = sanitizer_func(args, kwargs)
            else:
                # Basic sanitization
                sanitized_kwargs = {}
                for key, value in kwargs.items():
                    if isinstance(value, str):
                        sanitized_kwargs[key] = input_sanitizer.sanitize_string(value, key)
                    elif isinstance(value, (int, float)):
                        sanitized_kwargs[key] = input_sanitizer.sanitize_numeric_input(value, key)
                    else:
                        sanitized_kwargs[key] = value
                kwargs = sanitized_kwargs
            
            return func(*args, **kwargs)
        return wrapper
    return decorator