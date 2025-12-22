import os
import time
import math
import json
import base64
from pathlib import Path
from typing import Dict, Tuple, Optional
from collections import deque
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import cv2
import onnxruntime as ort
import mediapipe as mp

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel
from dotenv import load_dotenv

# -----------------------------
# Loade environment variables
# -----------------------------
load_dotenv()
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173").split(",")
PORT = int(os.getenv("PORT", "8002"))
ONNX_MODEL_PATH = os.getenv("ONNX_MODEL_PATH", "resnet18_gazevec_best.onnx")
MONGO_URI = os.getenv("MONGO_URI")

# MongoDB globals
db_client = None
db = None

# Config
MAX_BUFFER = 120
GAZE_SMOOTHING_WINDOW = 5
EAR_THRESHOLD = 0.25
GAZE_THRESHOLD_DEG = 15.0
FRAME_SAMPLING_INTERVAL_S = 0.1 # Sample often for real-time feel

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MODEL_INPUT_SIZE = (224, 224)
PITCH_CLAMP = (-60.0, 60.0)
YAW_CLAMP = (-60.0, 60.0)

# (FastAPI app and middleware moved below lifespan)

# -----------------------------
# Pydantic models
# -----------------------------
class ImageAnalysisRequest(BaseModel):
    image_data: str
    session_id: str

class AnswerEvaluationRequest(BaseModel):
    question: str
    answer: str
    job_role: str = "Software Engineer"

# -----------------------------
# Session & State
# -----------------------------
@dataclass
class BehaviorState:
    ear_series: deque
    gaze_history: deque
    blink_count: int = 0
    last_blink_time: float = 0.0
    total_frames: int = 0
    bg_frames_detected_count: int = 0
    stress_frames: int = 0
    smile_frames: int = 0
    gaze_metrics: dict = None
    gaze_metrics: dict = None
    last_sample_time: float = 0.0
    # Smoothing State
    last_pitch: float = 0.0
    last_yaw: float = 0.0

    def __post_init__(self):
        if self.gaze_metrics is None:
            self.gaze_metrics = {
                "avg_gaze_deviation": 0.0,
                "eye_contact_ratio": 0.0,
                "blink_rate": 0.0,
                "engagement_score": 0.0,
                "stress_level": "Low",
                "dominant_emotion": "Neutral"
            }

sessions: Dict[str, BehaviorState] = {}

# -----------------------------
# Globals
# -----------------------------
ort_sess: Optional[ort.InferenceSession] = None
mp_face_mesh = None
mp_face_detection = None

# -----------------------------
# Vision Utilities (Restored for ONNX)
# -----------------------------
def safe_clip(val: float, minv: float, maxv: float) -> float:
    return max(minv, min(maxv, val))

def clamp_angles(pitch: float, yaw: float) -> Tuple[float, float]:
    return (safe_clip(pitch, PITCH_CLAMP[0], PITCH_CLAMP[1]),
            safe_clip(yaw, YAW_CLAMP[0], YAW_CLAMP[1]))

def decode_base64_image(data_uri_or_b64: str):
    try:
        if "base64," in data_uri_or_b64:
            data = data_uri_or_b64.split("base64,")[1]
        else:
            data = data_uri_or_b64
        b = base64.b64decode(data)
        arr = np.frombuffer(b, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Image Decode Error: {e}")
        return None

def preprocess_face_for_model(face_bgr: np.ndarray) -> Optional[np.ndarray]:
    try:
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (MODEL_INPUT_SIZE[1], MODEL_INPUT_SIZE[0]))
        face_float = face_resized.astype(np.float32) / 255.0
        face_norm = (face_float - IMAGENET_MEAN) / IMAGENET_STD
        return np.expand_dims(np.transpose(face_norm, (2,0,1)).astype(np.float32), axis=0)
    except Exception:
        return None

def gaze_vector_to_pitch_yaw_deg(vec: np.ndarray) -> Tuple[float, float]:
    try:
        x, y, z = float(vec[0]), float(vec[1]), float(vec[2])
        # Simple projection
        pitch_rad = math.asin(y) 
        yaw_rad = math.atan2(x, -z)
        return clamp_angles(math.degrees(pitch_rad), math.degrees(yaw_rad))
    except Exception:
        return 0.0, 0.0

# -----------------------------
# Initialization & App
# -----------------------------
def init_models():
    global ort_sess, mp_face_mesh, mp_face_detection
    
    # 1. ONNX
    if os.path.exists(ONNX_MODEL_PATH):
        try:
            ort_sess = ort.InferenceSession(ONNX_MODEL_PATH, providers=["CPUExecutionProvider"])
            print(f"✅ Loaded ONNX model from {ONNX_MODEL_PATH}")
        except Exception as e:
            print(f"❌ Failed to load ONNX model: {e}")
            ort_sess = None
    else:
        print(f"⚠️ ONNX Model not found at {ONNX_MODEL_PATH}. Gaze will be mocked.")

    # 2. MediaPipe
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )
    mp_face_detection = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_client, db
    init_models()
    
    if MONGO_URI:
        try:
            db_client = AsyncIOMotorClient(MONGO_URI)
            db = db_client.get_default_database()
            await db_client.admin.command('ping')
            print("✅ Connected to MongoDB Successfully")
        except Exception as e:
            print(f"❌ MongoDB Connection Error: {e}")
    
    yield
    if db_client: db_client.close()
    if mp_face_mesh: mp_face_mesh.close()
    if mp_face_detection: mp_face_detection.close()

app = FastAPI(title="Gaze API", version="2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def analyze_gaze_real(frame, face_rect):
    h, w, _ = frame.shape
    x, y, fw, fh = face_rect
    
    center_x, center_y = x + fw // 2, y + fh // 2
    max_dim = max(fw, fh) * 1.2 
    x1 = int(max(0, center_x - max_dim // 2))
    y1 = int(max(0, center_y - max_dim // 2))
    x2 = int(min(w, center_x + max_dim // 2))
    y2 = int(min(h, center_y + max_dim // 2))
    
    face_roi = frame[y1:y2, x1:x2]
    pitch, yaw = 0.0, 0.0
    
    if ort_sess and face_roi.size > 0:
        model_input = preprocess_face_for_model(face_roi)
        if model_input is not None:
            output = ort_sess.run(None, {'input': model_input})[0][0]
            pitch, yaw = gaze_vector_to_pitch_yaw_deg(output)
            
    return pitch, yaw


def calculate_ear(landmarks, indices):
    # Eye Aspect Ratio
    # Vertical lines
    v1 = np.linalg.norm(np.array([landmarks[indices[1]].x, landmarks[indices[1]].y]) - 
                        np.array([landmarks[indices[5]].x, landmarks[indices[5]].y]))
    v2 = np.linalg.norm(np.array([landmarks[indices[2]].x, landmarks[indices[2]].y]) - 
                        np.array([landmarks[indices[4]].x, landmarks[indices[4]].y]))
    
    # Horizontal line
    h = np.linalg.norm(np.array([landmarks[indices[0]].x, landmarks[indices[0]].y]) - 
                       np.array([landmarks[indices[3]].x, landmarks[indices[3]].y]))
    
    ear = (v1 + v2) / (2.0 * h + 1e-6)
    return ear

def analyze_emotion_v2(landmarks):
    # Detailed Geometric Emotion
    # Happy: Mouth Width vs Jaw Width? Or Mouth Corners Up?
    
    # 1. Smile Check (Mouth corners vs Lip centers)
    # Left Corner 61, Right 291
    # Top Lip 13, Bottom 14
    
    lc = np.array([landmarks[61].x, landmarks[61].y])
    rc = np.array([landmarks[291].x, landmarks[291].y])
    
    top = np.array([landmarks[13].x, landmarks[13].y])
    bot = np.array([landmarks[14].x, landmarks[14].y])
    
    mouth_w = np.linalg.norm(lc - rc)
    mouth_h = np.linalg.norm(top - bot)
    
    # 2. Eye Aperture (Surprise/Stress) not implemented yet to keep simple
    
    # Ratio
    ratio = mouth_w / (mouth_h + 0.001)
    
    # Thresholds typically: Neutral ~1.2-2.0. Smile > 2.5-3.0
    if ratio > 2.8: 
        return "Happy"
        
    return "Neutral"

# -----------------------------
# HTTP Analysis Endpoint
# -----------------------------
@app.post("/api/analyze-frame")
async def analyze_frame_endpoint(req: ImageAnalysisRequest):
    session_id = req.session_id
    if session_id not in sessions:
        sessions[session_id] = BehaviorState(deque(maxlen=MAX_BUFFER), deque(maxlen=MAX_BUFFER))
    
    state = sessions[session_id]
    
    frame = decode_base64_image(req.image_data)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image data")
        
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # MediaPipe Mesh (High accuracy mode if needed, but keeping standard for speed)
    mesh_results = mp_face_mesh.process(rgb_frame)
    
    face_detected = False
    pitch, yaw = 0.0, 0.0
    
    if mesh_results.multi_face_landmarks:
        face_detected = True
        lms = mesh_results.multi_face_landmarks[0].landmark
        
        # 1. Detection (Needed for ONNX bbox)
        detection_results = mp_face_detection.process(rgb_frame)
        if detection_results.detections:
            d = detection_results.detections[0]
            box = d.location_data.relative_bounding_box
            x, y, fw, fh = int(box.xmin * w), int(box.ymin * h), int(box.width * w), int(box.height * h)
            
            # Gaze using ONNX
            raw_pitch, raw_yaw = analyze_gaze_real(frame, (x, y, fw, fh))
            
            # Smoothing (Exponential Moving Average)
            alpha = 0.2 # Smoothing factor (0.1 = very smooth/slow, 0.9 = responsive/jittery)
            pitch = alpha * raw_pitch + (1 - alpha) * state.last_pitch
            yaw = alpha * raw_yaw + (1 - alpha) * state.last_yaw
            
            # Update state
            state.last_pitch = pitch
            state.last_yaw = yaw
        
        # Smooth Logic could go here (Moving Average)
        
        # 2. Blink Detection (EAR)
        # Left Eye: 33, 160, 158, 133, 153, 144
        # Right Eye: 362, 385, 387, 263, 373, 380
        left_ear = calculate_ear(lms, [33, 160, 158, 133, 153, 144])
        right_ear = calculate_ear(lms, [362, 385, 387, 263, 373, 380])
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Blink Threshold (Typical 0.2 - 0.3)
        if avg_ear < 0.25:
            # Check if this is a new blink (debouncing simple)
            if time.time() - state.last_blink_time > 0.2: 
                state.blink_count += 1
                state.last_blink_time = time.time()
        
        # 3. Emotion
        state.gaze_metrics["dominant_emotion"] = analyze_emotion_v2(lms)
        
        # 4. Engagement (Head looking forward)
        # +/- 20 degrees tolerance
        if abs(pitch) < 20 and abs(yaw) < 20:
             state.bg_frames_detected_count += 1
             
        # Metrics Calculation
        state.total_frames += 1
        state.gaze_metrics["engagement_score"] = float(state.bg_frames_detected_count) / float(state.total_frames)
        # CPM = (Blinks / Frames) * FPS * 60... Approximating via time
        # Better: Blinks per minute relative to session start
        # Simplified:
        state.gaze_metrics["blink_rate"] = state.blink_count # Just sending count for raw data, or rate
        
    return {
        "type": "analysis_result",
        "session_id": session_id,
        "gaze_analysis": {"pitch": float(pitch), "yaw": float(yaw)},
        "behavior_analysis": state.gaze_metrics,
        "face_detected": face_detected,
        "timestamp": datetime.now().isoformat()
    }

# -----------------------------
# Resume & MongoDB Endpoints
# -----------------------------


# -----------------------------
# Endpoints
# -----------------------------
@app.websocket("/ws/analyze/{session_id}")
async def websocket_analysis(ws: WebSocket, session_id: str):
    await ws.accept()
    if session_id not in sessions:
        sessions[session_id] = BehaviorState(deque(maxlen=MAX_BUFFER), deque(maxlen=MAX_BUFFER))
    
    state = sessions[session_id]
    
    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)
            
            if msg.get("type") != "analyze_frame":
                continue
                
            frame = decode_base64_image(msg.get("image_data", ""))
            if frame is None: continue
            
            # 1. Detection
            # 1. Decode
            frame = decode_base64_image(msg.get("image_data", ""))
            if frame is None: continue
            
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # MediaPipe Mesh (High accuracy mode if needed, but keeping standard for speed)
            mesh_results = mp_face_mesh.process(rgb_frame)
            
            face_detected = False
            pitch, yaw = 0.0, 0.0
            
            if mesh_results.multi_face_landmarks:
                face_detected = True
                lms = mesh_results.multi_face_landmarks[0].landmark
                
                # 1. Detection (Needed for ONNX bbox)
                detection_results = mp_face_detection.process(rgb_frame)
                if detection_results.detections:
                    d = detection_results.detections[0]
                    box = d.location_data.relative_bounding_box
                    x, y, fw, fh = int(box.xmin * w), int(box.ymin * h), int(box.width * w), int(box.height * h)
                    
                    # Gaze using ONNX
                    raw_pitch, raw_yaw = analyze_gaze_real(frame, (x, y, fw, fh))
                    
                    # Smoothing (Exponential Moving Average)
                    alpha = 0.2
                    pitch = alpha * raw_pitch + (1 - alpha) * state.last_pitch
                    yaw = alpha * raw_yaw + (1 - alpha) * state.last_yaw
                    
                    # Update state
                    state.last_pitch = pitch
                    state.last_yaw = yaw
                
                # 2. Blink Detection (EAR)
                left_ear = calculate_ear(lms, [33, 160, 158, 133, 153, 144])
                right_ear = calculate_ear(lms, [362, 385, 387, 263, 373, 380])
                avg_ear = (left_ear + right_ear) / 2.0
                
                if avg_ear < 0.25:
                    if time.time() - state.last_blink_time > 0.2: 
                        state.blink_count += 1
                        state.last_blink_time = time.time()
                
                # 3. Emotion
                state.gaze_metrics["dominant_emotion"] = analyze_emotion_v2(lms)
                
                # 4. Engagement
                if abs(pitch) < 20 and abs(yaw) < 20:
                     state.bg_frames_detected_count += 1
                     
                state.gaze_metrics["engagement_score"] = float(state.bg_frames_detected_count) / float(state.total_frames + 1)
                state.gaze_metrics["blink_rate"] = state.blink_count # Raw count
            
            state.total_frames += 1
            
            # Response
            await ws.send_json({
                "type": "analysis_result",
                "session_id": session_id,
                "gaze_analysis": {"pitch": float(pitch), "yaw": float(yaw)},
                "behavior_analysis": state.gaze_metrics,
                "face_detected": face_detected,
                "timestamp": datetime.now().isoformat()
            })
            
    except WebSocketDisconnect:
        print(f"Client disconnected: {session_id}")
    except Exception as e:
        print(f"WS Error: {e}")

# -----------------------------
# LLM Utilities
# -----------------------------
from openai import AsyncOpenAI

# Use OpenRouter as seen in .env
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

@app.post("/api/evaluate-answer")
async def evaluate_answer_endpoint(req: AnswerEvaluationRequest):
    if not OPENROUTER_API_KEY:
        # Fallback if no key provided
        return {
            "evaluation": {
                "overall_score": 75,
                "detailed_feedback": "Mock evaluation (OpenAI Key missing)."
            }
        }

    prompt = f"""
    You are an expert technical interviewer. Evaluate the following candidate answer for the role of {req.job_role}.
    
    Question: {req.question}
    Answer: {req.answer}
    
    Return a strictly valid JSON object with:
    - technical_score (0-100)
    - communication_score (0-100)
    - relevance_score (0-100)
    - overall_score (average of above)
    - strengths (list of strings)
    - weaknesses (list of strings)
    - sentiment (positive/neutral/negative)
    - detailed_feedback (string, max 50 words)
    """

    try:
        response = await client.chat.completions.create(
            model="mistralai/mistral-7b-instruct:free",
            messages=[
                {"role": "system", "content": "You are a precise and fair technical interviewer. Output JSON only."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3
        )
        
        result_json = json.loads(response.choices[0].message.content)
        
        return {
            "question": req.question,
            "answer": req.answer,
            "job_role": req.job_role,
            "evaluation": result_json,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        print(f"LLM Error: {e}")
        raise HTTPException(status_code=500, detail="AI Evaluation Failed")
