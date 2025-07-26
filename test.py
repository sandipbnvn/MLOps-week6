#!/usr/bin/env python3
"""
Test script for Iris API Docker container
Tests all endpoints to ensure the API is working correctly
"""

import requests
import json
import time
import sys
import os

# Configuration
BASE_URL = "http://localhost:8200"
TIMEOUT = 30  # seconds to wait for container to start

def wait_for_service():
    """Wait for the service to be ready"""
    print("⏳ Waiting for service to start...")
    for i in range(TIMEOUT):
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                print("✅ Service is ready!")
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
        if (i + 1) % 5 == 0:
            print(f"   Still waiting... ({i + 1}/{TIMEOUT}s)")
    
    print("❌ Service failed to start within timeout")
    return False

def test_root_endpoint():
    """Test the root endpoint"""
    print("\n🧪 Testing root endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/", timeout=10)
        response.raise_for_status()
        
        expected = {"message": "Welcome to the Iris Classifier API!v4"}
        actual = response.json()
        
        if actual == expected:
            print("✅ Root endpoint test passed")
            return True
        else:
            print(f"❌ Root endpoint test failed")
            print(f"   Expected: {expected}")
            print(f"   Got: {actual}")
            return False
            
    except Exception as e:
        print(f"❌ Root endpoint test failed: {e}")
        return False

def test_health_endpoint():
    """Test the health endpoint"""
    print("\n🧪 Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        response.raise_for_status()
        
        expected = {"status": "healthy"}
        actual = response.json()
        
        if actual == expected:
            print("✅ Health endpoint test passed")
            return True
        else:
            print(f"❌ Health endpoint test failed")
            print(f"   Expected: {expected}")
            print(f"   Got: {actual}")
            return False
            
    except Exception as e:
        print(f"❌ Health endpoint test failed: {e}")
        return False

def test_prediction_endpoint():
    """Test the prediction endpoint"""
    print("\n🧪 Testing prediction endpoint...")
    try:
        test_data = {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
        
        response = requests.post(
            f"{BASE_URL}/predict/",
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        response.raise_for_status()
        
        result = response.json()
        
        if "predicted_class" in result:
            print(f"✅ Prediction endpoint test passed")
            print(f"   Predicted class: {result['predicted_class']}")
            return True
        else:
            print(f"❌ Prediction endpoint test failed")
            print(f"   Expected 'predicted_class' in response")
            print(f"   Got: {result}")
            return False
            
    except Exception as e:
        print(f"❌ Prediction endpoint test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Iris API tests...")
    
    # Wait for service to be ready
    if not wait_for_service():
        sys.exit(1)
    
    # Run all tests
    tests = [
        test_root_endpoint,
        test_health_endpoint,
        test_prediction_endpoint
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    # Summary
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 