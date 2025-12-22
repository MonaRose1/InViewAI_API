import requests
import json
import base64
import asyncio
import websockets
import time
from PIL import Image
import io

BASE_URL = "http://127.0.0.1:8002"
WS_URL = "ws://127.0.0.1:8002/ws/analyze"
SESSION_ID = "test_session_123"

def create_mock_image_b64():
    # Create a small 224x224 RGB image
    img = Image.new('RGB', (224, 224), color=(73, 109, 137))
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

def test_health():
    print("\n--- Testing /health ---")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_analyze_frame():
    print("\n--- Testing /api/analyze-frame ---")
    img_b64 = create_mock_image_b64()
    payload = {
        "image_data": f"data:image/jpeg;base64,{img_b64}",
        "session_id": SESSION_ID
    }
    try:
        response = requests.post(f"{BASE_URL}/api/analyze-frame", json=payload)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_evaluate_answer():
    print("\n--- Testing /api/evaluate-answer ---")
    payload = {
        "question": "Explain polymorphism in Object Oriented Programming.",
        "answer": "Polymorphism allows objects of different types to be treated as objects of a common base type.",
        "job_role": "Python Developer"
    }
    try:
        response = requests.post(f"{BASE_URL}/api/evaluate-answer", json=payload)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_session_summary():
    print("\n--- Testing /api/session-summary ---")
    try:
        response = requests.get(f"{BASE_URL}/api/session-summary/{SESSION_ID}")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

async def test_websocket():
    print("\n--- Testing WebSocket /ws/analyze ---")
    ws_uri = f"{WS_URL}/{SESSION_ID}"
    img_b64 = create_mock_image_b64()
    
    try:
        async with websockets.connect(ws_uri) as websocket:
            payload = {
                "type": "analyze_frame",
                "image_data": f"data:image/jpeg;base64,{img_b64}"
            }
            await websocket.send(json.dumps(payload))
            print("Message sent to WebSocket")
            
            response = await websocket.recv()
            print(f"WebSocket Received: {json.dumps(json.loads(response), indent=2)}")
            return True
    except Exception as e:
        print(f"WebSocket Error: {e}")
        return False

def run_all_tests():
    print(f"Starting API Tests against {BASE_URL}")
    results = []
    
    results.append(("Health Check", test_health()))
    results.append(("Analyze Frame", test_analyze_frame()))
    results.append(("Evaluate Answer", test_evaluate_answer()))
    results.append(("Session Summary", test_session_summary()))
    
    # Run async WebSocket test
    ws_success = asyncio.run(test_websocket())
    results.append(("WebSocket Analysis", ws_success))
    
    print("\n" + "="*30)
    print("FINAL TEST RESULTS")
    print("="*30)
    all_passed = True
    for test_name, success in results:
        status = "PASSED" if success else "FAILED"
        print(f"{test_name:20}: {status}")
        if not success:
            all_passed = False
    
    if all_passed:
        print("\nAll endpoints are working correctly!")
    else:
        print("\nSome tests failed. Check the errors above.")

if __name__ == "__main__":
    run_all_tests()
