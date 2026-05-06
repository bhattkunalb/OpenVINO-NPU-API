import requests
import os

# Configuration
API_URL = "http://127.0.0.1:8000"
API_KEY = os.getenv("OPENVINO_API_KEY", "test-key")

def test_health():
    print("Testing /health...")
    try:
        response = requests.get(f"{API_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Content: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")

def test_models_unauthorized():
    print("\nTesting /v1/models (unauthorized)...")
    try:
        response = requests.get(f"{API_URL}/v1/models")
        print(f"Status: {response.status_code}")
        print(f"Content: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")

def test_models_authorized():
    print("\nTesting /v1/models (authorized)...")
    try:
        headers = {"Authorization": f"Bearer {API_KEY}"}
        response = requests.get(f"{API_URL}/v1/models", headers=headers)
        print(f"Status: {response.status_code}")
        print(f"Content: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_health()
    test_models_unauthorized()
    test_models_authorized()
