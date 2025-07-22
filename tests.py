#!/usr/bin/env python3
"""
Simple test runner for the MLOps pipeline.
This script runs all tests and extracts accuracy reports.
"""

import unittest
import sys
import os
import json
import subprocess
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def run_unit_tests():
    """Run all unit tests and return results."""
    print("=" * 60)
    print("RUNNING UNIT TESTS")
    print("=" * 60)
    
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = os.path.join(os.path.dirname(__file__), 'tests')
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result

def run_integration_tests():
    """Run integration tests and return results."""
    print("\n" + "=" * 60)
    print("RUNNING INTEGRATION TESTS")
    print("=" * 60)
    
    # Run integration tests specifically
    loader = unittest.TestLoader()
    start_dir = os.path.join(os.path.dirname(__file__), 'tests')
    suite = loader.discover(start_dir, pattern='test_integration.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result

def run_pipeline_and_extract_accuracy():
    """Run the complete pipeline and extract accuracy metrics."""
    print("\n" + "=" * 60)
    print("RUNNING COMPLETE PIPELINE")
    print("=" * 60)
    
    try:
        # Run the pipeline
        result = subprocess.run(
            [sys.executable, 'run_pipeline.py'],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(__file__)
        )
        
        if result.returncode == 0:
            print("✅ Pipeline completed successfully!")
            
            # Extract accuracy from metrics file
            metrics_file = os.path.join(os.path.dirname(__file__), 'artifacts', 'metrics.json')
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    data = json.load(f)
                
                metrics = data.get('metrics', {})
                summary = data.get('evaluation_summary', {})
                
                print("\n" + "=" * 60)
                print("ACCURACY REPORT")
                print("=" * 60)
                print(f"Overall Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
                print(f"Precision (Macro): {metrics.get('precision_macro', 'N/A'):.4f}")
                print(f"Recall (Macro): {metrics.get('recall_macro', 'N/A'):.4f}")
                print(f"F1-Score (Macro): {metrics.get('f1_macro', 'N/A'):.4f}")
                print(f"Meets Threshold: {'✅' if summary.get('meets_threshold', False) else '❌'}")
                print(f"Sample Count: {summary.get('sample_count', 'N/A')}")
                
                # Per-class metrics
                print("\nPer-Class Performance:")
                for class_name in ['setosa', 'versicolor', 'virginica']:
                    precision = metrics.get(f'precision_{class_name}', 'N/A')
                    recall = metrics.get(f'recall_{class_name}', 'N/A')
                    f1 = metrics.get(f'f1_{class_name}', 'N/A')
                    print(f"  {class_name.title()}:")
                    if isinstance(precision, (int, float)):
                        print(f"    Precision: {precision:.4f}")
                        print(f"    Recall: {recall:.4f}")
                        print(f"    F1-Score: {f1:.4f}")
                    else:
                        print(f"    Precision: {precision}")
                        print(f"    Recall: {recall}")
                        print(f"    F1-Score: {f1}")
                
                return {
                    'success': True,
                    'accuracy': metrics.get('accuracy', 0),
                    'meets_threshold': summary.get('meets_threshold', False),
                    'metrics': metrics,
                    'summary': summary
                }
            else:
                print("❌ Metrics file not found!")
                return {'success': False, 'error': 'Metrics file not found'}
        else:
            print("❌ Pipeline failed!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return {'success': False, 'error': result.stderr}
            
    except Exception as e:
        print(f"❌ Error running pipeline: {e}")
        return {'success': False, 'error': str(e)}

def run_code_quality_checks():
    """Run code quality checks."""
    print("\n" + "=" * 60)
    print("RUNNING CODE QUALITY CHECKS")
    print("=" * 60)
    
    checks = []
    
    # Check if black is available
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'black', '--check', 'src/', 'tests/', 'run_pipeline.py'],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(__file__)
        )
        if result.returncode == 0:
            print("✅ Code formatting (black): PASSED")
            checks.append(('Code Formatting', True))
        else:
            print("❌ Code formatting (black): FAILED")
            checks.append(('Code Formatting', False))
    except Exception as e:
        print(f"⚠️  Code formatting check skipped: {e}")
        checks.append(('Code Formatting', 'SKIPPED'))
    
    # Check if flake8 is available
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'flake8', 'src/', 'tests/', 'run_pipeline.py', '--max-line-length=88', '--ignore=E203,W503'],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(__file__)
        )
        if result.returncode == 0:
            print("✅ Code linting (flake8): PASSED")
            checks.append(('Code Linting', True))
        else:
            print("❌ Code linting (flake8): FAILED")
            print(result.stdout)
            checks.append(('Code Linting', False))
    except Exception as e:
        print(f"⚠️  Code linting check skipped: {e}")
        checks.append(('Code Linting', 'SKIPPED'))
    
    return checks

def generate_test_report(unit_result, integration_result, pipeline_result, quality_checks):
    """Generate a comprehensive test report."""
    print("\n" + "=" * 60)
    print("TEST SUMMARY REPORT")
    print("=" * 60)
    
    # Test results
    print(f"Unit Tests: {'✅ PASSED' if unit_result.wasSuccessful() else '❌ FAILED'}")
    print(f"  - Tests run: {unit_result.testsRun}")
    print(f"  - Failures: {len(unit_result.failures)}")
    print(f"  - Errors: {len(unit_result.errors)}")
    
    print(f"Integration Tests: {'✅ PASSED' if integration_result.wasSuccessful() else '❌ FAILED'}")
    print(f"  - Tests run: {integration_result.testsRun}")
    print(f"  - Failures: {len(integration_result.failures)}")
    print(f"  - Errors: {len(integration_result.errors)}")
    
    # Pipeline results
    if pipeline_result['success']:
        print(f"Pipeline Execution: ✅ SUCCESS")
        print(f"  - Accuracy: {pipeline_result['accuracy']:.4f}")
        print(f"  - Meets Threshold: {'✅' if pipeline_result['meets_threshold'] else '❌'}")
    else:
        print(f"Pipeline Execution: ❌ FAILED")
        print(f"  - Error: {pipeline_result['error']}")
    
    # Quality checks
    print("\nCode Quality Checks:")
    for check_name, status in quality_checks:
        if status is True:
            print(f"  - {check_name}: ✅ PASSED")
        elif status is False:
            print(f"  - {check_name}: ❌ FAILED")
        else:
            print(f"  - {check_name}: ⚠️  SKIPPED")
    
    # Overall status
    all_tests_passed = (unit_result.wasSuccessful() and 
                       integration_result.wasSuccessful() and 
                       pipeline_result['success'])
    
    print(f"\nOverall Status: {'✅ ALL TESTS PASSED' if all_tests_passed else '❌ SOME TESTS FAILED'}")
    
    return all_tests_passed

def main():
    """Main test runner function."""
    print("🧪 MLOps Pipeline Test Runner")
    print("=" * 60)
    
    # Change to the pipeline directory
    pipeline_dir = os.path.dirname(__file__)
    os.chdir(pipeline_dir)
    
    # Run all tests
    unit_result = run_unit_tests()
    integration_result = run_integration_tests()
    pipeline_result = run_pipeline_and_extract_accuracy()
    quality_checks = run_code_quality_checks()
    
    # Generate report
    all_passed = generate_test_report(unit_result, integration_result, pipeline_result, quality_checks)
    
    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main() 