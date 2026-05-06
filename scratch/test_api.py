"""Test script to verify OpenVINO NPU API health and authentication."""

import os
import requests

# Configuration
API_URL = "http://127.0.0.1:8000"
API_KEY = os.getenv("OPENVINO_API_KEY", "test-key")


def test_health():
    """Verify that the /health endpoint is accessible without an API key."""
    print("Testing /health...")
    try:
        response = requests.get(f"{API_URL}/health", timeout=10)
        print(f"Status: {response.status_code}")
        print(f"Content: {response.json()}")
    except requests.RequestException as e:
        print(f"Error connecting to health endpoint: {e}")


def test_models_unauthorized():
    """Verify that /v1/models is protected and returns 401 when unauthorized."""
    print("\nTesting /v1/models (unauthorized)...")
    try:
        response = requests.get(f"{API_URL}/v1/models", timeout=10)
        print(f"Status: {response.status_code}")
        print(f"Content: {response.json()}")
    except requests.RequestException as e:
        print(f"Error during unauthorized check: {e}")


def test_models_authorized():
    """Verify that /v1/models is accessible with a valid API key."""
    print("\nTesting /v1/models (authorized)...")
    try:
        headers = {"Authorization": f"Bearer {API_KEY}"}
        response = requests.get(f"{API_URL}/v1/models", headers=headers, timeout=10)
        print(f"Status: {response.status_code}")
        print(f"Content: {response.json()}")
    except requests.RequestException as e:
        print(f"Error during authorized check: {e}")


if __name__ == "__main__":
    test_health()
    test_models_unauthorized()
    test_models_authorized()
