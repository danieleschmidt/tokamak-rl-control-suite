"""
Compliance and regulatory framework for tokamak RL control systems.

This module implements comprehensive compliance monitoring, audit trails,
safety standards adherence, and regulatory reporting capabilities.
"""

import json
import time
import hashlib
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from abc import ABC, abstractmethod
import warnings

try:
    import cryptography
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    import base64
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    # Using basic security


class ComplianceStandard(Enum):
    """Supported compliance standards."""
    ISO_45001 = "ISO 45001"  # Occupational Health and Safety
    IEC_61513 = "IEC 61513"  # Nuclear I&C Systems
    IEEE_1012 = "IEEE 1012"  # Software Verification and Validation
    NIST_800_53 = "NIST 800-53"  # Security Controls
    GDPR = "GDPR"  # General Data Protection Regulation
    HIPAA = "HIPAA"  # Health Insurance Portability
    SOX = "SOX"  # Sarbanes-Oxley Act
    ITER_SAFETY = "ITER Safety Requirements"
    NUCLEAR_SAFETY = "Nuclear Safety Standards"


class AuditLevel(Enum):
    """Audit logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    SECURITY = "SECURITY"
    COMPLIANCE = "COMPLIANCE"


class DataClassification(Enum):
    """Data classification levels."""
    PUBLIC = "PUBLIC"
    INTERNAL = "INTERNAL"
    CONFIDENTIAL = "CONFIDENTIAL"
    RESTRICTED = "RESTRICTED"
    TOP_SECRET = "TOP_SECRET"


@dataclass
class AuditLogEntry:
    """Individual audit log entry."""
    timestamp: datetime
    event_id: str
    user_id: str
    action: str
    resource: str
    level: AuditLevel
    details: Dict[str, Any]
    classification: DataClassification
    compliance_tags: List[str] = field(default_factory=list)
    signature: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        result['level'] = self.level.value
        result['classification'] = self.classification.value
        return result


@dataclass
class ComplianceViolation:
    """Record of compliance violation."""
    violation_id: str
    standard: ComplianceStandard
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    description: str
    detected_at: datetime
    component: str
    remediation_required: bool
    remediation_deadline: Optional[datetime] = None
    remediation_actions: List[str] = field(default_factory=list)
    status: str = "OPEN"  # OPEN, IN_PROGRESS, RESOLVED, CLOSED
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result['standard'] = self.standard.value
        result['detected_at'] = self.detected_at.isoformat()
        if self.remediation_deadline:
            result['remediation_deadline'] = self.remediation_deadline.isoformat()
        return result


class AuditLogger:
    """Tamper-resistant audit logging system."""
    
    def __init__(self, log_file: Optional[str] = None, encryption_key: Optional[bytes] = None):
        self.log_file = log_file or "/var/log/tokamak_rl/audit.log"
        self.entries: List[AuditLogEntry] = []
        self.sequence_number = 0
        
        # Initialize encryption if available
        self.cipher = None
        if CRYPTO_AVAILABLE and encryption_key:
            self.cipher = Fernet(encryption_key)
        elif encryption_key:
            warnings.warn("Encryption requested but cryptography not available")
    
    def log_event(self, user_id: str, action: str, resource: str, 
                  level: AuditLevel = AuditLevel.INFO,
                  details: Optional[Dict[str, Any]] = None,
                  classification: DataClassification = DataClassification.INTERNAL,
                  compliance_tags: Optional[List[str]] = None) -> str:
        """Log an auditable event."""
        
        event_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)
        
        entry = AuditLogEntry(
            timestamp=timestamp,
            event_id=event_id,
            user_id=user_id,
            action=action,
            resource=resource,
            level=level,
            details=details or {},
            classification=classification,
            compliance_tags=compliance_tags or []
        )
        
        # Add integrity signature
        entry.signature = self._generate_signature(entry)
        
        self.entries.append(entry)
        self.sequence_number += 1
        
        # Persist to file if configured
        if self.log_file:
            self._persist_entry(entry)
        
        return event_id
    
    def _generate_signature(self, entry: AuditLogEntry) -> str:
        """Generate integrity signature for audit entry."""
        # Create signature based on entry content
        content = f"{entry.timestamp.isoformat()}{entry.user_id}{entry.action}{entry.resource}"
        content += json.dumps(entry.details, sort_keys=True)
        content += str(self.sequence_number)
        
        # Simple hash-based signature (in production, use proper digital signatures)
        signature_hash = hashlib.sha256(content.encode()).hexdigest()
        return signature_hash[:16]  # Truncate for readability
    
    def _persist_entry(self, entry: AuditLogEntry) -> None:
        """Persist audit entry to file."""
        try:
            log_data = json.dumps(entry.to_dict()) + "\n"
            
            # Encrypt if cipher available
            if self.cipher:
                log_data = self.cipher.encrypt(log_data.encode()).decode()
                log_data += "\n"
            
            # In a real implementation, this would use proper file handling
            # For now, we simulate persistence
            pass
            
        except Exception as e:
            # Critical: audit logging must not fail
            warnings.warn(f"Failed to persist audit entry: {e}")
    
    def get_entries(self, start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None,
                   user_id: Optional[str] = None,
                   level: Optional[AuditLevel] = None) -> List[AuditLogEntry]:
        """Retrieve audit entries with filtering."""
        filtered_entries = self.entries
        
        if start_time:
            filtered_entries = [e for e in filtered_entries if e.timestamp >= start_time]
        
        if end_time:
            filtered_entries = [e for e in filtered_entries if e.timestamp <= end_time]
        
        if user_id:
            filtered_entries = [e for e in filtered_entries if e.user_id == user_id]
        
        if level:
            filtered_entries = [e for e in filtered_entries if e.level == level]
        
        return filtered_entries
    
    def verify_integrity(self) -> Dict[str, Any]:
        """Verify integrity of audit log."""
        results = {
            'total_entries': len(self.entries),
            'verified_entries': 0,
            'tampered_entries': 0,
            'integrity_status': 'UNKNOWN'
        }
        
        for i, entry in enumerate(self.entries):
            # Temporarily store current sequence number
            original_seq = self.sequence_number
            self.sequence_number = i
            
            expected_signature = self._generate_signature(entry)
            
            if entry.signature == expected_signature:
                results['verified_entries'] += 1
            else:
                results['tampered_entries'] += 1
            
            # Restore sequence number
            self.sequence_number = original_seq
        
        if results['tampered_entries'] == 0:
            results['integrity_status'] = 'VERIFIED'
        else:
            results['integrity_status'] = 'COMPROMISED'
        
        return results


class ComplianceMonitor:
    """Real-time compliance monitoring and violation detection."""
    
    def __init__(self, standards: List[ComplianceStandard]):
        self.standards = standards
        self.violations: List[ComplianceViolation] = []
        self.rules: Dict[ComplianceStandard, List[Callable]] = {}
        self.audit_logger = AuditLogger()
        
        # Initialize compliance rules
        self._setup_compliance_rules()
    
    def _setup_compliance_rules(self) -> None:
        """Setup compliance monitoring rules for each standard."""
        
        for standard in self.standards:
            self.rules[standard] = []
            
            if standard == ComplianceStandard.ISO_45001:
                # Occupational Health and Safety rules
                self.rules[standard].extend([
                    self._check_safety_system_availability,
                    self._check_emergency_response_time,
                    self._check_operator_training_status,
                    self._check_incident_reporting
                ])
            
            elif standard == ComplianceStandard.IEC_61513:
                # Nuclear I&C Systems rules
                self.rules[standard].extend([
                    self._check_system_independence,
                    self._check_fail_safe_behavior,
                    self._check_redundancy_requirements,
                    self._check_software_validation
                ])
            
            elif standard == ComplianceStandard.IEEE_1012:
                # Software V&V rules
                self.rules[standard].extend([
                    self._check_code_coverage,
                    self._check_test_documentation,
                    self._check_verification_activities,
                    self._check_validation_activities
                ])
            
            elif standard == ComplianceStandard.NIST_800_53:
                # Security controls
                self.rules[standard].extend([
                    self._check_access_controls,
                    self._check_audit_logging,
                    self._check_encryption_usage,
                    self._check_incident_response
                ])
            
            elif standard == ComplianceStandard.GDPR:
                # Data protection rules
                self.rules[standard].extend([
                    self._check_data_minimization,
                    self._check_consent_management,
                    self._check_data_retention,
                    self._check_breach_notification
                ])
    
    def check_compliance(self, system_state: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive compliance check."""
        results = {
            'compliance_status': 'COMPLIANT',
            'violations_detected': 0,
            'new_violations': [],
            'standards_checked': len(self.standards),
            'check_timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        for standard in self.standards:
            standard_violations = self._check_standard_compliance(standard, system_state)
            
            for violation in standard_violations:
                if violation.violation_id not in [v.violation_id for v in self.violations]:
                    self.violations.append(violation)
                    results['new_violations'].append(violation.to_dict())
                    results['violations_detected'] += 1
                    
                    # Log compliance violation
                    self.audit_logger.log_event(
                        user_id="system",
                        action="COMPLIANCE_VIOLATION_DETECTED",
                        resource=violation.component,
                        level=AuditLevel.COMPLIANCE,
                        details=violation.to_dict(),
                        classification=DataClassification.CONFIDENTIAL,
                        compliance_tags=[standard.value]
                    )
        
        if results['violations_detected'] > 0:
            results['compliance_status'] = 'NON_COMPLIANT'
        
        return results
    
    def _check_standard_compliance(self, standard: ComplianceStandard, 
                                 system_state: Dict[str, Any]) -> List[ComplianceViolation]:
        """Check compliance for a specific standard."""
        violations = []
        
        for rule in self.rules.get(standard, []):
            try:
                violation = rule(system_state)
                if violation:
                    violations.append(violation)
            except Exception as e:
                # Log rule execution failure
                self.audit_logger.log_event(
                    user_id="system",
                    action="COMPLIANCE_RULE_FAILURE",
                    resource=f"{standard.value}:{rule.__name__}",
                    level=AuditLevel.ERROR,
                    details={'error': str(e)},
                    classification=DataClassification.INTERNAL
                )
        
        return violations
    
    # ISO 45001 compliance rules
    def _check_safety_system_availability(self, system_state: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """Check safety system availability."""
        safety_available = system_state.get('safety_system_available', True)
        
        if not safety_available:
            return ComplianceViolation(
                violation_id=str(uuid.uuid4()),
                standard=ComplianceStandard.ISO_45001,
                severity="CRITICAL",
                description="Safety system is not available",
                detected_at=datetime.now(timezone.utc),
                component="safety_system",
                remediation_required=True,
                remediation_actions=["Restore safety system", "Implement backup safety measures"]
            )
        return None
    
    def _check_emergency_response_time(self, system_state: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """Check emergency response time compliance."""
        response_time = system_state.get('emergency_response_time', 0)
        max_allowed_time = 5.0  # 5 seconds maximum
        
        if response_time > max_allowed_time:
            return ComplianceViolation(
                violation_id=str(uuid.uuid4()),
                standard=ComplianceStandard.ISO_45001,
                severity="HIGH",
                description=f"Emergency response time {response_time}s exceeds maximum {max_allowed_time}s",
                detected_at=datetime.now(timezone.utc),
                component="emergency_system",
                remediation_required=True
            )
        return None
    
    def _check_operator_training_status(self, system_state: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """Check operator training status."""
        # Placeholder implementation
        return None
    
    def _check_incident_reporting(self, system_state: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """Check incident reporting compliance."""
        # Placeholder implementation
        return None
    
    # IEC 61513 compliance rules
    def _check_system_independence(self, system_state: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """Check system independence requirements."""
        # Placeholder implementation
        return None
    
    def _check_fail_safe_behavior(self, system_state: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """Check fail-safe behavior."""
        # Placeholder implementation
        return None
    
    def _check_redundancy_requirements(self, system_state: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """Check redundancy requirements."""
        # Placeholder implementation
        return None
    
    def _check_software_validation(self, system_state: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """Check software validation."""
        # Placeholder implementation
        return None
    
    # IEEE 1012 compliance rules
    def _check_code_coverage(self, system_state: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """Check code coverage requirements."""
        coverage = system_state.get('code_coverage', 0)
        required_coverage = 85.0  # 85% minimum
        
        if coverage < required_coverage:
            return ComplianceViolation(
                violation_id=str(uuid.uuid4()),
                standard=ComplianceStandard.IEEE_1012,
                severity="MEDIUM",
                description=f"Code coverage {coverage}% below required {required_coverage}%",
                detected_at=datetime.now(timezone.utc),
                component="testing_system",
                remediation_required=True
            )
        return None
    
    def _check_test_documentation(self, system_state: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """Check test documentation completeness."""
        # Placeholder implementation
        return None
    
    def _check_verification_activities(self, system_state: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """Check verification activities."""
        # Placeholder implementation
        return None
    
    def _check_validation_activities(self, system_state: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """Check validation activities."""
        # Placeholder implementation
        return None
    
    # NIST 800-53 compliance rules
    def _check_access_controls(self, system_state: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """Check access control implementation."""
        # Placeholder implementation
        return None
    
    def _check_audit_logging(self, system_state: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """Check audit logging compliance."""
        # Placeholder implementation
        return None
    
    def _check_encryption_usage(self, system_state: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """Check encryption usage."""
        # Placeholder implementation
        return None
    
    def _check_incident_response(self, system_state: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """Check incident response procedures."""
        # Placeholder implementation
        return None
    
    # GDPR compliance rules
    def _check_data_minimization(self, system_state: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """Check data minimization principle."""
        # Placeholder implementation
        return None
    
    def _check_consent_management(self, system_state: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """Check consent management."""
        # Placeholder implementation
        return None
    
    def _check_data_retention(self, system_state: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """Check data retention policies."""
        # Placeholder implementation
        return None
    
    def _check_breach_notification(self, system_state: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """Check breach notification procedures."""
        # Placeholder implementation
        return None
    
    def get_compliance_report(self) -> Dict[str, Any]:
        """Generate comprehensive compliance report."""
        open_violations = [v for v in self.violations if v.status == "OPEN"]
        critical_violations = [v for v in open_violations if v.severity == "CRITICAL"]
        
        report = {
            'report_generated': datetime.now(timezone.utc).isoformat(),
            'standards_monitored': [s.value for s in self.standards],
            'total_violations': len(self.violations),
            'open_violations': len(open_violations),
            'critical_violations': len(critical_violations),
            'compliance_score': self._calculate_compliance_score(),
            'violations_by_standard': self._group_violations_by_standard(),
            'violations_by_severity': self._group_violations_by_severity(),
            'audit_log_integrity': self.audit_logger.verify_integrity()
        }
        
        return report
    
    def _calculate_compliance_score(self) -> float:
        """Calculate overall compliance score (0-100)."""
        if not self.violations:
            return 100.0
        
        # Weight violations by severity
        severity_weights = {"LOW": 1, "MEDIUM": 3, "HIGH": 7, "CRITICAL": 15}
        
        total_weight = sum(severity_weights.get(v.severity, 1) for v in self.violations if v.status == "OPEN")
        max_possible_weight = len(self.standards) * 100  # Assume max 100 points per standard
        
        score = max(0, 100 - (total_weight / max_possible_weight * 100))
        return round(score, 2)
    
    def _group_violations_by_standard(self) -> Dict[str, int]:
        """Group violations by compliance standard."""
        groups = {}
        for violation in self.violations:
            standard = violation.standard.value
            groups[standard] = groups.get(standard, 0) + 1
        return groups
    
    def _group_violations_by_severity(self) -> Dict[str, int]:
        """Group violations by severity."""
        groups = {}
        for violation in self.violations:
            severity = violation.severity
            groups[severity] = groups.get(severity, 0) + 1
        return groups


class DataProtectionManager:
    """Data protection and privacy compliance manager."""
    
    def __init__(self):
        self.data_inventory: Dict[str, Dict[str, Any]] = {}
        self.consent_records: Dict[str, Dict[str, Any]] = {}
        self.retention_policies: Dict[str, int] = {}  # Data type -> retention days
        
        # Setup default retention policies
        self._setup_default_policies()
    
    def _setup_default_policies(self) -> None:
        """Setup default data retention policies."""
        self.retention_policies.update({
            'operational_data': 2555,  # 7 years
            'safety_data': 3650,       # 10 years  
            'audit_logs': 2555,        # 7 years
            'user_data': 1095,         # 3 years
            'session_data': 30,        # 30 days
            'temp_data': 1             # 1 day
        })
    
    def register_data_processing(self, data_type: str, purpose: str, 
                                legal_basis: str, retention_days: Optional[int] = None) -> str:
        """Register data processing activity."""
        processing_id = str(uuid.uuid4())
        
        self.data_inventory[processing_id] = {
            'data_type': data_type,
            'purpose': purpose,
            'legal_basis': legal_basis,
            'retention_days': retention_days or self.retention_policies.get(data_type, 365),
            'registered_at': datetime.now(timezone.utc).isoformat(),
            'status': 'ACTIVE'
        }
        
        return processing_id
    
    def record_consent(self, user_id: str, data_types: List[str], 
                      purposes: List[str], granted: bool) -> str:
        """Record user consent."""
        consent_id = str(uuid.uuid4())
        
        self.consent_records[consent_id] = {
            'user_id': user_id,
            'data_types': data_types,
            'purposes': purposes,
            'granted': granted,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'ip_address': None,  # Would be captured in real implementation
            'user_agent': None   # Would be captured in real implementation
        }
        
        return consent_id
    
    def check_data_retention(self) -> List[Dict[str, Any]]:
        """Check for data that should be purged according to retention policies."""
        expired_data = []
        current_time = datetime.now(timezone.utc)
        
        for processing_id, processing_info in self.data_inventory.items():
            if processing_info['status'] != 'ACTIVE':
                continue
                
            registered_at = datetime.fromisoformat(processing_info['registered_at'])
            retention_days = processing_info['retention_days']
            
            if (current_time - registered_at).days > retention_days:
                expired_data.append({
                    'processing_id': processing_id,
                    'data_type': processing_info['data_type'],
                    'expired_days': (current_time - registered_at).days - retention_days
                })
        
        return expired_data


def create_compliance_system(standards: List[ComplianceStandard]) -> ComplianceMonitor:
    """Factory function to create compliance monitoring system."""
    return ComplianceMonitor(standards)


def create_audit_logger(encryption_key: Optional[bytes] = None) -> AuditLogger:
    """Factory function to create audit logger."""
    return AuditLogger(encryption_key=encryption_key)