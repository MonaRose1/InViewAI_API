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

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()
OUT_DIR = Path(os.getenv("OUT_DIR", "./outputs"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

ONNX_MODEL_PATH = os.getenv("ONNX_MODEL_PATH", "resnet18_gazevec_best1.onnx")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost,http://127.0.0.1").split(",")
PORT = int(os.getenv("PORT", "8002"))

ORT_INTRA_THREADS = int(os.getenv("ORT_INTRA_THREADS", "2"))
ORT_INTER_THREADS = int(os.getenv("ORT_INTER_THREADS", "1"))

MAX_BUFFER = int(os.getenv("MAX_BUFFER", "120"))
GAZE_SMOOTHING_WINDOW = int(os.getenv("GAZE_SMOOTHING_WINDOW", "5"))
EAR_THRESHOLD = float(os.getenv("EAR_THRESHOLD", "0.25"))
GAZE_THRESHOLD_DEG = float(os.getenv("GAZE_THRESHOLD_DEG", "15.0"))
ENGAGEMENT_WINDOW_S = int(os.getenv("ENGAGEMENT_WINDOW_S", "60"))
SESSION_TTL_S = int(os.getenv("SESSION_TTL_S", str(60*30)))  # 30 min
WS_FRAME_RATE_LIMIT = float(os.getenv("WS_FRAME_RATE_LIMIT", "15.0"))
FRAME_SAMPLING_INTERVAL_S = float(os.getenv("FRAME_SAMPLING_INTERVAL_S", "3.0"))  # sample every 3s

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MODEL_INPUT_SIZE = (224, 224)
PITCH_CLAMP = (-60.0, 60.0)
YAW_CLAMP = (-60.0, 60.0)

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Gaze API", version="1.0")
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
    blink_count: int = 0
    last_blink_time: float = 0.0
    total_frames: int = 0
    engaged_frames: int = 0
    gaze_metrics: dict = None
    calibration_offset: Tuple[float, float] = (0.0, 0.0)
    last_seen: float = 0.0
    last_frame_ts: float = 0.0
    last_sample_time: float = 0.0

    def __post_init__(self):
        if self.gaze_history is None:
            self.gaze_history = deque(maxlen=GAZE_SMOOTHING_WINDOW)
        if self.gaze_metrics is None:
            self.gaze_metrics = {
                "avg_gaze_deviation": 0.0,
                "eye_contact_ratio": 0.0,
                "blink_rate": 0.0,
                "engagement_score": 0.0
            }

sessions: Dict[str, BehaviorState] = {}

# -----------------------------
# Globals
# -----------------------------
ort_sess: Optional[ort.InferenceSession] = None
mp = None
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
# ONNX & MediaPipe init
# -----------------------------
def init_ort_session(path: str):
    if not os.path.exists(path):
        return None
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = ORT_INTRA_THREADS
    sess_opts.inter_op_num_threads = ORT_INTER_THREADS
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, sess_options=sess_opts, providers=["CPUExecutionProvider"])

def init_mediapipe():
    global mp, mp_face_mesh, mp_face_detection
    import mediapipe as _mp
    mp = _mp
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False,
                                                   max_num_faces=1,
                                                   refine_landmarks=True,
                                                   min_detection_confidence=0.5,
                                                   min_tracking_confidence=0.5)
    mp_face_detection = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# -----------------------------
# Behavior update
# -----------------------------
def update_behavior(beh_state: BehaviorState, ear: float, pitch_deg: float, yaw_deg: float, ts: float):
    # Sampling interval check
    if ts - beh_state.last_sample_time < FRAME_SAMPLING_INTERVAL_S:
        return beh_state.gaze_metrics
    beh_state.last_sample_time = ts

    beh_state.last_seen = ts
    beh_state.total_frames += 1
    beh_state.last_frame_ts = ts

    # Blink detection
    if ear < EAR_THRESHOLD:
        if ts - beh_state.last_blink_time > 0.25:
            beh_state.blink_count += 1
            beh_state.last_blink_time = ts

    beh_state.ear_series.append(ear)
    beh_state.iris_dx.append(yaw_deg)
    beh_state.iris_dy.append(pitch_deg)

    # Engagement score
    if math.hypot(yaw_deg, pitch_deg) < GAZE_THRESHOLD_DEG:
        beh_state.engaged_frames += 1

    duration = max(1.0, len(beh_state.timestamps))
    eye_contact_ratio = beh_state.engaged_frames / max(1, beh_state.total_frames)
    avg_gaze_dev = float(np.mean([math.hypot(dx, dy) for dx, dy in zip(beh_state.iris_dx, beh_state.iris_dy)])) if beh_state.iris_dx else 0.0
    blink_score = 1.0 - min(1.0, abs((beh_state.blink_count / duration) - 0.25) / 0.25)
    engagement_score = eye_contact_ratio * 0.7 + blink_score * 0.3

    beh_state.gaze_metrics.update({
        "avg_gaze_deviation": avg_gaze_dev,
        "eye_contact_ratio": float(eye_contact_ratio),
        "blink_rate": float(beh_state.blink_count / duration),
        "engagement_score": float(engagement_score)
    })

    return beh_state.gaze_metrics

# -----------------------------
# API startup
# -----------------------------
@app.on_event("startup")
async def startup():
    global ort_sess
    ort_sess = init_ort_session(ONNX_MODEL_PATH)
    init_mediapipe()

# -----------------------------
# Health endpoint
# -----------------------------
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": ort_sess is not None
    }

# -----------------------------
# Analyze frame endpoint
# -----------------------------
@app.post("/api/analyze-frame")
async def analyze_frame_endpoint(req: ImageAnalysisRequest):
    frame = decode_base64_image(req.image_data)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image")
    sid = req.session_id
    if sid not in sessions:
        sessions[sid] = BehaviorState(deque(maxlen=MAX_BUFFER),
                                      deque(maxlen=MAX_BUFFER),
                                      deque(maxlen=MAX_BUFFER),
                                      deque(maxlen=MAX_BUFFER),
                                      deque(maxlen=GAZE_SMOOTHING_WINDOW))
    beh_state = sessions[sid]

    ts = time.time()

    # Face detection
    face_results = mp_face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not face_results.detections:
        return {
            "session_id": sid,
            "gaze_analysis": {"pitch": 0.0, "yaw": 0.0},
            "behavior_analysis": beh_state.gaze_metrics,
            "face_detected": False,
            "timestamp": datetime.now().isoformat()
        }

    # Fallback gaze (no ONNX)
    pitch, yaw = 0.0, 0.0
    gaze_metrics = update_behavior(beh_state, ear=0.3, pitch_deg=pitch, yaw_deg=yaw, ts=ts)

    return {
        "session_id": sid,
        "gaze_analysis": {"pitch": pitch, "yaw": yaw},
        "behavior_analysis": gaze_metrics,
        "face_detected": True,
        "timestamp": datetime.now().isoformat()
    }

# -----------------------------
# Evaluate answer endpoint
# -----------------------------
@app.post("/api/evaluate-answer")
async def evaluate_answer_endpoint(req: AnswerEvaluationRequest):
    return {
        "question": req.question,
        "answer": req.answer,
        "job_role": req.job_role,
        "evaluation": {
            "relevance_score": 0.5,
            "completeness_score": 0.5,
            "technical_accuracy_score": 0.5,
            "communication_skills_score": 0.5,
            "overall_score": 0.5,
            "strengths": [],
            "improvement_areas": [],
            "sentiment_analysis": "neutral",
            "detailed_feedback": "Mock evaluation"
        },
        "timestamp": datetime.now().isoformat()
    }

# -----------------------------
# WebSocket endpoint
# -----------------------------
@app.websocket("/ws/analyze/{session_id}")
async def websocket_analysis(ws: WebSocket, session_id: str):
    await ws.accept()
    if session_id not in sessions:
        sessions[session_id] = BehaviorState(deque(maxlen=MAX_BUFFER),
                                             deque(maxlen=MAX_BUFFER),
                                             deque(maxlen=MAX_BUFFER),
                                             deque(maxlen=MAX_BUFFER),
                                             deque(maxlen=GAZE_SMOOTHING_WINDOW))
    beh_state = sessions[session_id]

    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)
            if msg.get("type") != "analyze_frame":
                continue

            ts = time.time()

            # Decode image
            frame = decode_base64_image(msg.get("image_data", ""))
            if frame is None:
                continue

            # Face detection
            face_results = mp_face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not face_results.detections:
                continue  # Skip frame if no face

            # Sampling check inside update_behavior
            pitch, yaw = 0.0, 0.0
            gaze_metrics = update_behavior(beh_state, ear=0.3, pitch_deg=pitch, yaw_deg=yaw, ts=ts)

            await ws.send_json({
                "type": "analysis_result",
                "session_id": session_id,
                "gaze_analysis": {"pitch": pitch, "yaw": yaw},
                "behavior_analysis": gaze_metrics,
                "timestamp": datetime.now().isoformat()
            })

    except WebSocketDisconnect:
        pass

# -----------------------------
# Session summary endpoint
# -----------------------------
@app.get("/api/session-summary/{session_id}")
async def session_summary(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    beh_state = sessions[session_id]
    return {
        "session_id": session_id,
        "summary": beh_state.gaze_metrics,
        "total_frames_processed": beh_state.total_frames,
        "timestamp": datetime.now().isoformat()
    }

# -----------------------------
# Run server
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, log_level="info")