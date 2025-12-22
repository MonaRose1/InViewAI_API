import os
import time
import math
import json
import base64
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from collections import deque
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import cv2
import onnxruntime as ort

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()
OUT_DIR = Path(os.getenv("OUT_DIR", "./outputs"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

ONNX_MODEL_PATH = os.getenv("ONNX_MODEL_PATH", "resnet18_gazevec_best1.onnx")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost,http://127.0.0.1").split(",")
PORT = int(os.getenv("PORT", "8002"))

# API Keys
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# Constants
ORT_INTRA_THREADS = int(os.getenv("ORT_INTRA_THREADS", "2"))
ORT_INTER_THREADS = int(os.getenv("ORT_INTER_THREADS", "1"))
MAX_BUFFER = int(os.getenv("MAX_BUFFER", "120"))
GAZE_SMOOTHING_WINDOW = int(os.getenv("GAZE_SMOOTHING_WINDOW", "5"))
EAR_THRESHOLD = float(os.getenv("EAR_THRESHOLD", "0.25"))
GAZE_THRESHOLD_DEG = float(os.getenv("GAZE_THRESHOLD_DEG", "15.0"))
FRAME_SAMPLING_INTERVAL_S = float(os.getenv("FRAME_SAMPLING_INTERVAL_S", "1.0")) 

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MODEL_INPUT_SIZE = (224, 224)
PITCH_CLAMP = (-60.0, 60.0)
YAW_CLAMP = (-60.0, 60.0)

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Gaze & Behavioral AI API", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o for o in ALLOWED_ORIGINS if o],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
# Session state
# -----------------------------
@dataclass
class BehaviorState:
    ear_series: deque
    timestamps: deque
    iris_dx: deque
    iris_dy: deque
    gaze_history: deque
    mar_series: deque
    furrow_series: deque
    blink_count: int = 0
    last_blink_time: float = 0.0
    total_frames: int = 0
    engaged_frames: int = 0
    gaze_metrics: dict = None
    last_seen: float = 0.0
    last_frame_ts: float = 0.0
    last_sample_time: float = 0.0
    sentiment: str = "Neutral"
    warnings: List[str] = None

    def __post_init__(self):
        if self.gaze_history is None: self.gaze_history = deque(maxlen=GAZE_SMOOTHING_WINDOW)
        if self.mar_series is None: self.mar_series = deque(maxlen=MAX_BUFFER)
        if self.furrow_series is None: self.furrow_series = deque(maxlen=MAX_BUFFER)
        if self.warnings is None: self.warnings = []
        if self.gaze_metrics is None:
            self.gaze_metrics = {
                "avg_gaze_deviation": 0.0,
                "eye_contact_ratio": 0.0,
                "blink_rate": 0.0,
                "engagement_score": 0.0,
                "speaking_ratio": 0.0,
                "stress_score": 0.0
            }

sessions: Dict[str, BehaviorState] = {}

# -----------------------------
# Globals
# -----------------------------
ort_sess: Optional[ort.InferenceSession] = None
mp_face_mesh = None
mp_face_detection = None

# -----------------------------
# Utilities
# -----------------------------
def safe_clip(val: float, minv: float, maxv: float) -> float:
    return max(minv, min(maxv, val))

def clamp_angles(pitch: float, yaw: float) -> Tuple[float, float]:
    return (safe_clip(pitch, PITCH_CLAMP[0], PITCH_CLAMP[1]),
            safe_clip(yaw, YAW_CLAMP[0], YAW_CLAMP[1]))

def decode_base64_image(data_uri_or_b64: str):
    try:
        if data_uri_or_b64.startswith("data:"):
            data = data_uri_or_b64.split(",")[1]
        else:
            data = data_uri_or_b64
        b = base64.b64decode(data)
        arr = np.frombuffer(b, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception:
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
        denom = math.sqrt(max(1e-8, x*x + z*z))
        yaw_rad = math.atan2(x, -z)
        pitch_rad = math.atan2(y, denom)
        return clamp_angles(math.degrees(pitch_rad), math.degrees(yaw_rad))
    except Exception:
        return 0.0, 0.0

# -----------------------------
# AI Logic
# -----------------------------
def init_ai():
    global ort_sess, mp_face_mesh, mp_face_detection
    import mediapipe as mp
    
    # ONNX
    if os.path.exists(ONNX_MODEL_PATH):
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = ORT_INTRA_THREADS
        sess_opts.inter_op_num_threads = ORT_INTER_THREADS
        ort_sess = ort.InferenceSession(ONNX_MODEL_PATH, sess_options=sess_opts, providers=["CPUExecutionProvider"])
    
    # MediaPipe
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=2, refine_landmarks=True)
    mp_face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)

def calculate_mar(landmarks, img_w, img_h) -> float:
    # Mouth landmarks: Top 13, Bottom 14, Left 78, Right 308
    def get_pt(idx): 
        l = landmarks.landmark[idx]
        return np.array([l.x * img_w, l.y * img_h])
    
    v = np.linalg.norm(get_pt(13) - get_pt(14))
    h = np.linalg.norm(get_pt(78) - get_pt(308))
    return v / max(1e-6, h)

def calculate_brow_furrow(landmarks, img_w, img_h) -> float:
    # Brows: 107, 336. Bridge: 168.
    def get_pt(idx): 
        l = landmarks.landmark[idx]
        return np.array([l.x * img_w, l.y * img_h])
        
    dist1 = np.linalg.norm(get_pt(107) - get_pt(168))
    dist2 = np.linalg.norm(get_pt(336) - get_pt(168))
    return (dist1 + dist2) / 2.0

def update_behavior(beh_state: BehaviorState, ear: float, mar: float, brow: float, pitch_deg: float, yaw_deg: float, ts: float):
    beh_state.last_seen = ts
    beh_state.total_frames += 1
    
    # Blink
    if ear < EAR_THRESHOLD:
        if ts - beh_state.last_blink_time > 0.25:
            beh_state.blink_count += 1
            beh_state.last_blink_time = ts

    beh_state.ear_series.append(ear)
    beh_state.mar_series.append(mar)
    beh_state.furrow_series.append(brow)
    beh_state.iris_dx.append(yaw_deg)
    beh_state.iris_dy.append(pitch_deg)

    if math.hypot(yaw_deg, pitch_deg) < GAZE_THRESHOLD_DEG:
        beh_state.engaged_frames += 1

    # Analysis
    duration = max(1.0, (ts - beh_state.timestamps[0])) if beh_state.timestamps else 1.0
    beh_state.timestamps.append(ts)
    
    speaking_ratio = sum(1 for m in beh_state.mar_series if m > 0.08) / len(beh_state.mar_series)
    eye_contact_ratio = beh_state.engaged_frames / beh_state.total_frames
    
    # Simple sentiment
    avg_mar = np.mean(beh_state.mar_series) if beh_state.mar_series else 0.0
    if avg_mar > 0.12: beh_state.sentiment = "Speaking/Expressive"
    elif brow < 15: beh_state.sentiment = "Stressed/Concentrated"
    else: beh_state.sentiment = "Neutral"

    beh_state.gaze_metrics.update({
        "avg_gaze_deviation": float(np.mean([math.hypot(dx, dy) for dx, dy in zip(beh_state.iris_dx, beh_state.iris_dy)])),
        "eye_contact_ratio": float(eye_contact_ratio),
        "blink_rate": float(beh_state.blink_count / duration),
        "speaking_ratio": float(speaking_ratio),
        "stress_score": float(1.0 - (brow / 30.0) if brow < 30 else 0.0),
        "engagement_score": float(eye_contact_ratio * 0.6 + (1.0 - min(1, abs((beh_state.blink_count / duration) - 0.3)/0.3)) * 0.4)
    })
    return beh_state.gaze_metrics

# -----------------------------
# API endpoints
# -----------------------------
@app.on_event("startup")
async def startup():
    init_ai()

@app.get("/health")
async def health():
    return {"status": "ok", "onnx": ort_sess is not None, "mediapipe": mp_face_mesh is not None}

@app.post("/api/analyze-frame")
async def analyze_frame(req: ImageAnalysisRequest):
    frame = decode_base64_image(req.image_data)
    if frame is None: raise HTTPException(400, "Invalid image")
    
    sid = req.session_id
    if sid not in sessions:
        sessions[sid] = BehaviorState(deque(maxlen=MAX_BUFFER), deque(maxlen=MAX_BUFFER),
                                      deque(maxlen=MAX_BUFFER), deque(maxlen=MAX_BUFFER),
                                      deque(maxlen=GAZE_SMOOTHING_WINDOW), deque(maxlen=MAX_BUFFER),
                                      deque(maxlen=MAX_BUFFER))
    beh_state = sessions[sid]
    ts = time.time()
    
    # 1. Anti-Cheat: Multiple faces
    warnings = []
    det_results = mp_face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if det_results.detections and len(det_results.detections) > 1:
        warnings.append("Multiple faces detected")
    
    # 2. Gaze Inference
    pitch, yaw = 0.0, 0.0
    if ort_sess and det_results.detections:
        # Simple extraction of the first face for ONNX
        d = det_results.detections[0].location_data.relative_bounding_box
        h, w, _ = frame.shape
        x1, y1 = int(d.xmin * w), int(d.ymin * h)
        x2, y2 = x1 + int(d.width * w), y1 + int(d.height * h)
        face_img = frame[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
        
        face_input = preprocess_face_for_model(face_img)
        if face_input is not None:
            outs = ort_sess.run(None, {'input': face_input})
            pitch, yaw = gaze_vector_to_pitch_yaw_deg(outs[0][0])

    # 3. Behavioral
    mesh_results = mp_face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    mar, brow = 0.0, 20.0
    if mesh_results.multi_face_landmarks:
        h, w, _ = frame.shape
        landmarks = mesh_results.multi_face_landmarks[0]
        mar = calculate_mar(landmarks, w, h)
        brow = calculate_brow_furrow(landmarks, w, h)
    
    # Check if candidate left
    if not det_results.detections and (ts - beh_state.last_seen > 5):
        warnings.append("Suspicious: Candidate left camera")

    metrics = update_behavior(beh_state, ear=0.3, mar=mar, brow=brow, pitch_deg=pitch, yaw_deg=yaw, ts=ts)
    beh_state.warnings = warnings

    return {
        "session_id": sid,
        "gaze": {"pitch": pitch, "yaw": yaw},
        "behavior": metrics,
        "sentiment": beh_state.sentiment,
        "warnings": warnings,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/evaluate-answer")
async def evaluate_answer(req: AnswerEvaluationRequest):
    if not OPENROUTER_API_KEY:
        raise HTTPException(500, "LLM Key not configured")
    
    try:
        prompt = f"""
        Analyze the following interview performance:
        Job Role: {req.job_role}
        Question: {req.question}
        Answer: {req.answer}

        Return a JSON object with:
        - technical_score (0-100)
        - soft_skills_score (0-100)
        - sentiment_label (Positive/Neutral/Negative)
        - keywords_matched (list of key technical terms found)
        - detailed_feedback (short paragraph)

        Ensure valid JSON only.
        """
        
        response = client.chat.completions.create(
            model="meta-llama/llama-3.1-8b-instruct:free",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return {
            "evaluation": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(500, f"LLM error: {str(e)}")

@app.get("/api/session-summary/{session_id}")
async def session_summary(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    beh_state = sessions[session_id]
    return {
        "session_id": session_id,
        "summary": beh_state.gaze_metrics,
        "last_sentiment": beh_state.sentiment,
        "warnings_history": beh_state.warnings,
        "total_frames": beh_state.total_frames,
        "timestamp": datetime.now().isoformat()
    }

@app.websocket("/ws/analyze/{session_id}")
async def websocket_endpoint(ws: WebSocket, session_id: str):
    await ws.accept()
    if session_id not in sessions:
        sessions[session_id] = BehaviorState(deque(maxlen=MAX_BUFFER), deque(maxlen=MAX_BUFFER),
                                      deque(maxlen=MAX_BUFFER), deque(maxlen=MAX_BUFFER),
                                      deque(maxlen=GAZE_SMOOTHING_WINDOW), deque(maxlen=MAX_BUFFER),
                                      deque(maxlen=MAX_BUFFER))
    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)
            if msg.get("type") == "analyze_frame":
                # Resuse endpoint logic or wrap it
                res = await analyze_frame(ImageAnalysisRequest(image_data=msg["image_data"], session_id=session_id))
                await ws.send_json(res)
    except WebSocketDisconnect:
        pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main1:app", host="0.0.0.0", port=PORT, log_level="info")
