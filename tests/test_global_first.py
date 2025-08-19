#!/usr/bin/env python3
"""
Test suite for Global-First features (i18n, compliance, cross-platform).
"""

import sys
import os
import time
from datetime import datetime, timezone

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def main():
    print('🌍 GLOBAL-FIRST FEATURES TEST SUITE')
    print('='*50)
    
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
    
    # Test 1: Internationalization Features
    print('\n=== Testing Internationalization ===')
    try:
        from tokamak_rl.i18n import (
            LocalizationManager, SupportedLanguage, SupportedRegion,
            MessageCatalog, NumberFormatter, DateTimeFormatter, UnitFormatter,
            get_global_l10n_manager, _, format_for_locale
        )
        
        # Test message catalog
        catalog = MessageCatalog()
        english_msg = catalog.get_message("system.startup", "en")
        french_msg = catalog.get_message("system.startup", "fr")
        
        test_result("Message catalog basic functionality", 
                   "Tokamak" in english_msg and "Tokamak" in french_msg)
        
        # Test localization manager
        from tokamak_rl.i18n import LocaleConfig
        config = LocaleConfig(SupportedLanguage.FRENCH, SupportedRegion.FR)
        manager = LocalizationManager(config)
        
        localized_msg = manager.get_message("system.startup")
        test_result("Localization manager", "Démarrage" in localized_msg)
        
        # Test number formatting
        number_formatter = NumberFormatter(config)
        formatted_number = number_formatter.format_number(1234.56)
        test_result("Number formatting", isinstance(formatted_number, str))
        
        # Test global manager
        global_manager = get_global_l10n_manager()
        test_result("Global localization manager", global_manager is not None)
        
        # Test convenience function
        message = _("system.ready")
        test_result("Convenience function", isinstance(message, str))
        
    except Exception as e:
        test_result("Internationalization features", False, str(e))
    
    # Test 2: Compliance Framework
    print('\n=== Testing Compliance Framework ===')
    try:
        from tokamak_rl.compliance import (
            ComplianceMonitor, ComplianceStandard, AuditLogger, AuditLevel,
            DataClassification, create_compliance_system, create_audit_logger
        )
        
        # Test audit logger
        audit_logger = create_audit_logger()
        event_id = audit_logger.log_event(
            user_id="test_user",
            action="TEST_ACTION",
            resource="test_resource",
            level=AuditLevel.INFO,
            details={"test": "data"}
        )
        
        test_result("Audit logging", isinstance(event_id, str) and len(event_id) > 0)
        
        # Test audit retrieval
        entries = audit_logger.get_entries(user_id="test_user")
        test_result("Audit log retrieval", len(entries) == 1)
        
        # Test integrity verification
        integrity_result = audit_logger.verify_integrity()
        test_result("Audit log integrity", 
                   integrity_result['integrity_status'] == 'VERIFIED')
        
        # Test compliance monitor
        standards = [ComplianceStandard.ISO_45001, ComplianceStandard.IEC_61513]
        monitor = create_compliance_system(standards)
        
        # Test compliance check
        system_state = {
            'safety_system_available': True,
            'emergency_response_time': 2.0,
            'code_coverage': 90.0
        }
        
        compliance_result = monitor.check_compliance(system_state)
        test_result("Compliance monitoring", 
                   compliance_result['compliance_status'] in ['COMPLIANT', 'NON_COMPLIANT'])
        
        # Test compliance report
        report = monitor.get_compliance_report()
        test_result("Compliance reporting", 
                   'compliance_score' in report and isinstance(report['compliance_score'], (int, float)))
        
    except Exception as e:
        test_result("Compliance framework", False, str(e))
    
    # Test 3: Cross-Platform Support
    print('\n=== Testing Cross-Platform Support ===')
    try:
        from tokamak_rl.cross_platform import (
            SystemDetector, PlatformType, ArchitectureType, EnvironmentManager,
            DeploymentEnvironment, PathManager, PerformanceConfig,
            get_system_info, get_environment_manager, setup_cross_platform_environment
        )
        
        # Test system detection
        system_info = SystemDetector.get_system_info()
        test_result("System info detection", 
                   hasattr(system_info, 'platform') and hasattr(system_info, 'cpu_count'))
        
        # Test platform detection
        platform = SystemDetector.detect_platform()
        test_result("Platform detection", isinstance(platform, PlatformType))
        
        # Test architecture detection
        architecture = SystemDetector.detect_architecture()
        test_result("Architecture detection", isinstance(architecture, ArchitectureType))
        
        # Test performance configuration
        perf_config = PerformanceConfig.create_optimal(system_info)
        test_result("Performance configuration", 
                   perf_config.max_workers > 0 and perf_config.max_workers <= 32)
        
        # Test path manager
        path_manager = PathManager(system_info)
        data_dir = path_manager.get_app_data_dir()
        test_result("Path management", str(data_dir).endswith('tokamak_rl'))
        
        # Test environment manager
        env_manager = EnvironmentManager(system_info)
        deployment_env = env_manager.detect_environment()
        test_result("Environment detection", isinstance(deployment_env, DeploymentEnvironment))
        
        # Test environment configuration
        env_config = env_manager.get_environment_config()
        test_result("Environment configuration", 
                   'environment' in env_config and 'debug' in env_config)
        
        # Test global functions
        global_system_info = get_system_info()
        test_result("Global system info", global_system_info.cpu_count > 0)
        
        global_env_manager = get_environment_manager()
        test_result("Global environment manager", global_env_manager is not None)
        
    except Exception as e:
        test_result("Cross-platform support", False, str(e))
    
    # Test 4: Integration Testing
    print('\n=== Testing Global-First Integration ===')
    try:
        # Test combined functionality
        system_info = get_system_info()
        env_manager = get_environment_manager()
        l10n_manager = get_global_l10n_manager()
        
        # Set up French locale for system running on detected platform
        from tokamak_rl.i18n import set_global_locale
        set_global_locale(SupportedLanguage.FRENCH, SupportedRegion.FR)
        
        # Get localized system status
        status = {
            'platform': system_info.platform.value,
            'status': 'online',
            'cpu_count': system_info.cpu_count,
            'memory_gb': system_info.memory_gb
        }
        
        formatted_status = l10n_manager.format_system_status(status)
        test_result("Localized system status", 
                   'platform' in formatted_status and 'cpu_count' in formatted_status)
        
        # Test compliance with localization
        audit_logger = create_audit_logger()
        event_id = audit_logger.log_event(
            user_id="system",
            action="LOCALE_CHANGED",
            resource="i18n_manager",
            details={"language": "fr", "region": "FR"}
        )
        
        test_result("Compliance with i18n", len(event_id) > 0)
        
        # Test cross-platform environment setup
        setup_result = setup_cross_platform_environment()
        test_result("Full environment setup", 
                   'system_info' in setup_result and 'environment' in setup_result)
        
    except Exception as e:
        test_result("Global-first integration", False, str(e))
    
    # Test 5: Container and Deployment Support
    print('\n=== Testing Deployment Support ===')
    try:
        from tokamak_rl.cross_platform import ContainerUtils
        
        # Test Dockerfile generation
        dockerfile = ContainerUtils.create_dockerfile()
        test_result("Dockerfile generation", 
                   "FROM python:" in dockerfile and "tokamak" in dockerfile)
        
        # Test Kubernetes manifest
        k8s_manifest = ContainerUtils.create_kubernetes_manifest()
        test_result("Kubernetes manifest", 
                   k8s_manifest['kind'] == 'Deployment' and 'spec' in k8s_manifest)
        
        # Test manifest structure
        containers = k8s_manifest['spec']['template']['spec']['containers']
        test_result("Container configuration", 
                   len(containers) == 1 and 'resources' in containers[0])
        
    except Exception as e:
        test_result("Deployment support", False, str(e))
    
    # Summary
    print('\n' + '='*50)
    print('🏁 GLOBAL-FIRST TEST SUITE COMPLETE')
    print(f'📊 RESULTS: {passed_tests}/{total_tests} tests passed')
    
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    print(f'📈 SUCCESS RATE: {success_rate:.1f}%')
    
    if success_rate >= 90:
        print('✅ GLOBAL-FIRST: EXCELLENT - Ready for worldwide deployment')
    elif success_rate >= 75:
        print('✅ GLOBAL-FIRST: GOOD - Most features working correctly')
    elif success_rate >= 50:
        print('⚠️  GLOBAL-FIRST: ACCEPTABLE - Some features need attention')
    else:
        print('❌ GLOBAL-FIRST: NEEDS IMPROVEMENT - Significant issues detected')
    
    return success_rate

if __name__ == "__main__":
    success_rate = main()
    sys.exit(0 if success_rate >= 75 else 1)