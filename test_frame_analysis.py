import requests
import json
import base64
import asyncio
import websockets
import time
from PIL import Image
import io
import sys

BASE_URL = "http://127.0.0.1:8003"
WS_URL = "ws://127.0.0.1:8003/ws/analyze"
SESSION_ID = "test_session_frame_analysis"

def create_mock_image_b64():
    """Creates a simple RGB image and converts it to base64."""
    try:
        # Create a small 224x224 RGB image (blue-ish)
        img = Image.new('RGB', (224, 224), color=(73, 109, 137))
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode()
    except Exception as e:
        print(f"Error creating mock image: {e}")
        sys.exit(1)

def test_analyze_frame_http():
    print(f"\n--- Testing HTTP POST {BASE_URL}/api/analyze-frame ---")
    img_b64 = create_mock_image_b64()
    payload = {
        "image_data": f"data:image/jpeg;base64,{img_b64}",
        "session_id": SESSION_ID
    }
    
    start_time = time.time()
    try:
        response = requests.post(f"{BASE_URL}/api/analyze-frame", json=payload)
        duration = time.time() - start_time
        
        print(f"Status Code: {response.status_code}")
        print(f"Duration: {duration:.4f}s")
        
        if response.status_code == 200:
            data = response.json()
            print("Response Data Keys:", list(data.keys()))
            # Validation
            if "gaze_analysis" in data and "behavior_analysis" in data:
                print("✅ Structure Validated")
                print(f"   Gaze: {data['gaze_analysis']}")
                print(f"   Emotion: {data['behavior_analysis'].get('dominant_emotion')}")
                return True
            else:
                print("❌ Missing expected keys in response")
                return False
        else:
            print(f"❌ Request Failed: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"❌ Connection Error. Is the server running on port 8002?")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

async def test_frame_analysis_ws():
    print(f"\n--- Testing WebSocket {WS_URL}/{SESSION_ID} ---")
    ws_uri = f"{WS_URL}/{SESSION_ID}"
    img_b64 = create_mock_image_b64()
    
    try:
        async with websockets.connect(ws_uri) as websocket:
            print("✅ Connected to WebSocket")
            
            payload = {
                "type": "analyze_frame",
                "image_data": f"data:image/jpeg;base64,{img_b64}"
            }
            
            start_time = time.time()
            await websocket.send(json.dumps(payload))
            print("📤 Sent frame for analysis")
            
            # Wait for response with timeout
            response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            duration = time.time() - start_time
            
            data = json.loads(response)
            print(f"📥 Received response in {duration:.4f}s")
            
            # Validation
            if data.get("type") == "analysis_result":
                print("✅ Response Type Valid")
                print(f"   Gaze: {data.get('gaze_analysis')}")
                return True
            else:
                print(f"❌ Unexpected response type: {data.get('type')}")
                print(f"Full response: {data}")
                return False
                
    except asyncio.TimeoutError:
        print("❌ WebSocket Timeout (No response in 5s)")
        return False
    except ConnectionRefusedError:
        print(f"❌ Connection Refused. Is the server running on port 8002?")
        return False
    except Exception as e:
        print(f"❌ WebSocket Error: {e}")
        return False

def run_tests():
    print("🚀 Starting Frame Analysis Tests...")
    
    # 1. HTTP Test
    http_success = test_analyze_frame_http()
    
    # 2. WebSocket Test
    ws_success = asyncio.run(test_frame_analysis_ws())
    
    print("\n" + "="*30)
    print("TEST SUMMARY")
    print("="*30)
    print(f"HTTP Endpoint: {'✅ PASS' if http_success else '❌ FAIL'}")
    print(f"WebSocket:     {'✅ PASS' if ws_success else '❌ FAIL'}")
    
    if http_success and ws_success:
        print("\n🎉 All frame analysis tests passed!")
        sys.exit(0)
    else:
        print("\n⚠️ Some tests failed.")
        sys.exit(1)

if __name__ == "__main__":
    run_tests()
