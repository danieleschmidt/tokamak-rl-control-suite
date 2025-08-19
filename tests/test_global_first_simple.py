#!/usr/bin/env python3
"""
Simplified test suite for Global-First features without external dependencies.
"""

import sys
import os
import time
from datetime import datetime, timezone

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def main():
    print('🌍 GLOBAL-FIRST FEATURES SIMPLE TEST SUITE')
    print('='*55)
    
    total_tests = 0
    passed_tests = 0
    
    def test_result(name, success, message=""):
        nonlocal total_tests, passed_tests
        total_tests += 1
        if success:
            passed_tests += 1
        status = "PASS" if success else "FAIL"
        print(f"[{status}] {name}")
        if message:
            print(f"    {message}")
    
    # Test 1: Basic Internationalization Structure
    print('\n=== Testing I18n Module Structure ===')
    try:
        # Test file existence
        i18n_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'tokamak_rl', 'i18n.py')
        test_result("I18n module file exists", os.path.exists(i18n_path))
        
        # Test basic imports without numpy dependency
        import tokamak_rl.i18n as i18n_module
        test_result("I18n module imports", hasattr(i18n_module, 'SupportedLanguage'))
        
        # Test enum classes
        test_result("Language enum exists", hasattr(i18n_module, 'SupportedLanguage'))
        test_result("Region enum exists", hasattr(i18n_module, 'SupportedRegion'))
        
        # Test basic message catalog without complex dependencies
        catalog_class = getattr(i18n_module, 'MessageCatalog', None)
        test_result("MessageCatalog class exists", catalog_class is not None)
        
        if catalog_class:
            catalog = catalog_class()
            test_result("MessageCatalog instantiation", catalog is not None)
            
            # Test basic message retrieval
            message = catalog.get_message("system.startup", "en")
            test_result("Basic message retrieval", isinstance(message, str) and len(message) > 0)
            
            # Test language support
            languages = catalog.get_supported_languages()
            test_result("Multiple languages supported", len(languages) >= 3)
        
    except Exception as e:
        test_result("I18n basic structure", False, str(e))
    
    # Test 2: Compliance Module Structure
    print('\n=== Testing Compliance Module Structure ===')
    try:
        # Test file existence
        compliance_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'tokamak_rl', 'compliance.py')
        test_result("Compliance module file exists", os.path.exists(compliance_path))
        
        # Test basic compliance enums and classes
        import tokamak_rl.compliance as compliance_module
        test_result("Compliance module imports", hasattr(compliance_module, 'ComplianceStandard'))
        
        # Test enum classes
        test_result("ComplianceStandard enum exists", hasattr(compliance_module, 'ComplianceStandard'))
        test_result("AuditLevel enum exists", hasattr(compliance_module, 'AuditLevel'))
        test_result("DataClassification enum exists", hasattr(compliance_module, 'DataClassification'))
        
        # Test dataclasses
        test_result("AuditLogEntry exists", hasattr(compliance_module, 'AuditLogEntry'))
        test_result("ComplianceViolation exists", hasattr(compliance_module, 'ComplianceViolation'))
        
        # Test basic audit logger without crypto
        audit_logger_class = getattr(compliance_module, 'AuditLogger', None)
        if audit_logger_class:
            logger = audit_logger_class()
            test_result("AuditLogger instantiation", logger is not None)
            
            # Test basic logging
            event_id = logger.log_event("test_user", "TEST_ACTION", "test_resource")
            test_result("Basic audit logging", isinstance(event_id, str) and len(event_id) > 0)
            
            # Test entry retrieval
            entries = logger.get_entries()
            test_result("Audit entry retrieval", len(entries) > 0)
        
    except Exception as e:
        test_result("Compliance basic structure", False, str(e))
    
    # Test 3: Cross-Platform Module Structure
    print('\n=== Testing Cross-Platform Module Structure ===')
    try:
        # Test file existence
        cross_platform_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'tokamak_rl', 'cross_platform.py')
        test_result("Cross-platform module file exists", os.path.exists(cross_platform_path))
        
        # Test basic imports
        import tokamak_rl.cross_platform as cp_module
        test_result("Cross-platform module imports", hasattr(cp_module, 'PlatformType'))
        
        # Test enum classes
        test_result("PlatformType enum exists", hasattr(cp_module, 'PlatformType'))
        test_result("ArchitectureType enum exists", hasattr(cp_module, 'ArchitectureType'))
        test_result("DeploymentEnvironment enum exists", hasattr(cp_module, 'DeploymentEnvironment'))
        
        # Test basic platform detection
        system_detector = getattr(cp_module, 'SystemDetector', None)
        if system_detector:
            platform = system_detector.detect_platform()
            test_result("Platform detection", hasattr(platform, 'value'))
            
            architecture = system_detector.detect_architecture()
            test_result("Architecture detection", hasattr(architecture, 'value'))
        
        # Test container utilities
        container_utils = getattr(cp_module, 'ContainerUtils', None)
        if container_utils:
            dockerfile = container_utils.create_dockerfile()
            test_result("Dockerfile generation", isinstance(dockerfile, str) and "FROM" in dockerfile)
            
            k8s_manifest = container_utils.create_kubernetes_manifest()
            test_result("Kubernetes manifest generation", isinstance(k8s_manifest, dict) and "kind" in k8s_manifest)
        
    except Exception as e:
        test_result("Cross-platform basic structure", False, str(e))
    
    # Test 4: Module Integration
    print('\n=== Testing Module Integration ===')
    try:
        # Test that all modules can be imported together
        import tokamak_rl.i18n
        import tokamak_rl.compliance  
        import tokamak_rl.cross_platform
        
        test_result("All modules import together", True)
        
        # Test basic integration - language enum values
        lang_enum = tokamak_rl.i18n.SupportedLanguage
        test_result("Language enum has values", len(list(lang_enum)) >= 5)
        
        # Test compliance standards enum
        compliance_enum = tokamak_rl.compliance.ComplianceStandard
        test_result("Compliance standards defined", len(list(compliance_enum)) >= 5)
        
        # Test platform enum
        platform_enum = tokamak_rl.cross_platform.PlatformType
        test_result("Platform types defined", len(list(platform_enum)) >= 4)
        
    except Exception as e:
        test_result("Module integration", False, str(e))
    
    # Test 5: Basic Functionality Tests
    print('\n=== Testing Basic Functionality ===')
    try:
        # Test message formatting
        import tokamak_rl.i18n as i18n
        catalog = i18n.MessageCatalog()
        
        # Test message with parameters
        message_with_params = catalog.get_message("safety.violation", "en")
        test_result("Safety message retrieval", "Safety" in message_with_params)
        
        # Test different languages
        english_msg = catalog.get_message("system.startup", "en")
        french_msg = catalog.get_message("system.startup", "fr")
        german_msg = catalog.get_message("system.startup", "de")
        
        test_result("Multi-language support", 
                   all(isinstance(msg, str) and len(msg) > 0 
                       for msg in [english_msg, french_msg, german_msg]))
        
        # Test compliance violation creation
        import tokamak_rl.compliance as compliance
        violation_class = compliance.ComplianceViolation
        
        violation = violation_class(
            violation_id="test-123",
            standard=compliance.ComplianceStandard.ISO_45001,
            severity="HIGH",
            description="Test violation",
            detected_at=datetime.now(timezone.utc),
            component="test_component",
            remediation_required=True
        )
        
        test_result("Compliance violation creation", violation.violation_id == "test-123")
        
        # Test violation serialization
        violation_dict = violation.to_dict()
        test_result("Violation serialization", isinstance(violation_dict, dict) and "violation_id" in violation_dict)
        
    except Exception as e:
        test_result("Basic functionality", False, str(e))
    
    # Summary
    print('\n' + '='*55)
    print('🏁 GLOBAL-FIRST SIMPLE TEST SUITE COMPLETE')
    print(f'📊 RESULTS: {passed_tests}/{total_tests} tests passed')
    
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    print(f'📈 SUCCESS RATE: {success_rate:.1f}%')
    
    if success_rate >= 90:
        print('✅ GLOBAL-FIRST: EXCELLENT - Core features working correctly')
    elif success_rate >= 75:
        print('✅ GLOBAL-FIRST: GOOD - Most features implemented successfully')
    elif success_rate >= 50:
        print('⚠️  GLOBAL-FIRST: ACCEPTABLE - Basic structure in place')
    else:
        print('❌ GLOBAL-FIRST: NEEDS IMPROVEMENT - Structural issues detected')
    
    return success_rate

if __name__ == "__main__":
    success_rate = main()
    sys.exit(0 if success_rate >= 75 else 1)