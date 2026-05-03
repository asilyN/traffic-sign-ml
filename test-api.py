#!/usr/bin/env python3
"""
Quick test script for the API with CNN model.
Run the API server first, then run this script.
"""

import requests
import sys
from pathlib import Path

API_URL = "http://localhost:5000"
TEST_IMAGE = "server/app/ml/dataset/sample-image/00001.png"


def test_health():
    """Test health endpoint."""
    print("Testing /health endpoint...")
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}\n")
    return response.status_code == 200


def test_ready():
    """Test readiness endpoint."""
    print("Testing /v1/ready endpoint...")
    response = requests.get(f"{API_URL}/v1/ready")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}\n")
    return response.status_code == 200


def test_predict():
    """Test prediction endpoint with an image."""
    print("Testing /v1/predict endpoint...")
    
    image_path = Path(TEST_IMAGE)
    if not image_path.exists():
        print(f"Error: Test image not found at {image_path}")
        return False
    
    with open(image_path, "rb") as f:
        files = {"file": f}
        response = requests.post(f"{API_URL}/v1/predict", files=files)
    
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}\n")
    return response.status_code == 200


def main():
    print("=" * 60)
    print("API Testing with CNN Model")
    print("=" * 60 + "\n")
    
    try:
        # Test endpoints
        health_ok = test_health()
        ready_ok = test_ready()
        predict_ok = test_predict()
        
        # Summary
        print("=" * 60)
        print("Test Results:")
        print(f"  Health:   {'✓ PASS' if health_ok else '✗ FAIL'}")
        print(f"  Ready:    {'✓ PASS' if ready_ok else '✗ FAIL'}")
        print(f"  Predict:  {'✓ PASS' if predict_ok else '✗ FAIL'}")
        print("=" * 60)
        
        if all([health_ok, ready_ok, predict_ok]):
            print("\n✓ All tests passed! CNN model is deployed successfully.")
            return 0
        else:
            print("\n✗ Some tests failed. Check the output above.")
            return 1
    
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to API at", API_URL)
        print("Make sure the server is running: python server/run.py")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
