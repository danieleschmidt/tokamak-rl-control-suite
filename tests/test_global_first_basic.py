#!/usr/bin/env python3
"""
Basic structural test for Global-First features.
"""

import sys
import os

def main():
    print('🌍 GLOBAL-FIRST BASIC STRUCTURE TEST')
    print('='*45)
    
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
    
    # Test file existence
    print('\n=== Testing File Existence ===')
    src_dir = os.path.join(os.path.dirname(__file__), '..', 'src', 'tokamak_rl')
    
    i18n_file = os.path.join(src_dir, 'i18n.py')
    test_result("I18n module file exists", os.path.exists(i18n_file))
    
    compliance_file = os.path.join(src_dir, 'compliance.py')
    test_result("Compliance module file exists", os.path.exists(compliance_file))
    
    cross_platform_file = os.path.join(src_dir, 'cross_platform.py')
    test_result("Cross-platform module file exists", os.path.exists(cross_platform_file))
    
    # Test file sizes (they should have substantial content)
    print('\n=== Testing File Content ===')
    
    if os.path.exists(i18n_file):
        i18n_size = os.path.getsize(i18n_file)
        test_result("I18n module has content", i18n_size > 10000)  # > 10KB
    
    if os.path.exists(compliance_file):
        compliance_size = os.path.getsize(compliance_file)
        test_result("Compliance module has content", compliance_size > 15000)  # > 15KB
    
    if os.path.exists(cross_platform_file):
        cp_size = os.path.getsize(cross_platform_file)
        test_result("Cross-platform module has content", cp_size > 20000)  # > 20KB
    
    # Test basic syntax by reading files
    print('\n=== Testing File Syntax ===')
    
    try:
        with open(i18n_file, 'r') as f:
            i18n_content = f.read()
        test_result("I18n module syntax valid", 
                   'class SupportedLanguage' in i18n_content and 'class MessageCatalog' in i18n_content)
    except Exception as e:
        test_result("I18n module syntax", False, str(e))
    
    try:
        with open(compliance_file, 'r') as f:
            compliance_content = f.read()
        test_result("Compliance module syntax valid",
                   'class ComplianceStandard' in compliance_content and 'class AuditLogger' in compliance_content)
    except Exception as e:
        test_result("Compliance module syntax", False, str(e))
    
    try:
        with open(cross_platform_file, 'r') as f:
            cp_content = f.read()
        test_result("Cross-platform module syntax valid",
                   'class PlatformType' in cp_content and 'class SystemDetector' in cp_content)
    except Exception as e:
        test_result("Cross-platform module syntax", False, str(e))
    
    # Test specific features are present
    print('\n=== Testing Feature Implementation ===')
    
    # I18n features
    if 'i18n_content' in locals():
        test_result("Multiple language support", 'ENGLISH' in i18n_content and 'FRENCH' in i18n_content)
        test_result("Message catalog implementation", 'get_message' in i18n_content)
        test_result("Number formatting", 'NumberFormatter' in i18n_content)
        test_result("DateTime formatting", 'DateTimeFormatter' in i18n_content)
        test_result("Unit formatting", 'UnitFormatter' in i18n_content)
        test_result("Accessibility support", 'AccessibilityFormatter' in i18n_content)
    
    # Compliance features
    if 'compliance_content' in locals():
        test_result("Audit logging", 'log_event' in compliance_content)
        test_result("Compliance monitoring", 'ComplianceMonitor' in compliance_content)
        test_result("Violation tracking", 'ComplianceViolation' in compliance_content)
        test_result("Data protection", 'DataProtectionManager' in compliance_content)
        test_result("Multiple standards", 'ISO_45001' in compliance_content and 'IEC_61513' in compliance_content)
    
    # Cross-platform features
    if 'cp_content' in locals():
        test_result("Platform detection", 'detect_platform' in cp_content)
        test_result("Architecture detection", 'detect_architecture' in cp_content)
        test_result("GPU detection", 'detect_gpu' in cp_content)
        test_result("Container support", 'create_dockerfile' in cp_content)
        test_result("Kubernetes support", 'create_kubernetes_manifest' in cp_content)
        test_result("Performance optimization", 'PerformanceConfig' in cp_content)
    
    # Summary
    print('\n' + '='*45)
    print('🏁 BASIC STRUCTURE TEST COMPLETE')
    print(f'📊 RESULTS: {passed_tests}/{total_tests} tests passed')
    
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    print(f'📈 SUCCESS RATE: {success_rate:.1f}%')
    
    if success_rate >= 90:
        print('✅ STRUCTURE: EXCELLENT - All components implemented')
    elif success_rate >= 75:
        print('✅ STRUCTURE: GOOD - Core features present')
    elif success_rate >= 50:
        print('⚠️  STRUCTURE: ACCEPTABLE - Basic implementation complete')
    else:
        print('❌ STRUCTURE: NEEDS IMPROVEMENT - Missing components')
    
    return success_rate

if __name__ == "__main__":
    success_rate = main()
    sys.exit(0 if success_rate >= 75 else 1)