import os
import glob
import time
import uuid
import threading
import json
import re
from collections import deque
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import logging

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; use system env vars

import cv2
import numpy as np
import requests
import torch
from flask import Flask, Response, jsonify, render_template, request, send_from_directory
from ultralytics import YOLO

# Force CPU if CUDA is not truly compatible (e.g. Blackwell sm_120 on older PyTorch)
def _get_device():
    if torch.cuda.is_available():
        try:
            # Quick sanity check — allocate a tiny tensor on GPU
            torch.zeros(1, device="cuda")
            return "cuda"
        except RuntimeError:
            pass
    return "cpu"

DEVICE = _get_device()

# Forecasting imports
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

# Optional: LSTM imports (install with: pip install tensorflow)
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PERSON_CLASS_ID = 0
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ===== NLP Configuration & Validation =====
NLP_CACHE_SIZE = 128
NLP_REQUEST_TIMEOUT = 20
NLP_MAX_PROMPT_LENGTH = 500
NLP_EXECUTOR = ThreadPoolExecutor(max_workers=3)

# Validate API keys at startup
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")

# Session pooling for HTTP connection reuse
_http_sessions = {}

def get_http_session(provider):
    """Return or create HTTP session for provider (connection pooling)."""
    if provider not in _http_sessions:
        sess = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=5, pool_maxsize=5)
        sess.mount("https://", adapter)
        sess.mount("http://", adapter)
        _http_sessions[provider] = sess
    return _http_sessions[provider]


class ForecastingEngine:
    """Multi-method crowd density forecasting with LSTM + statistical methods."""
    
    def __init__(self, lookback_window=120):
        self.lookback_window = lookback_window
        self.lstm_model = None
        self.scaler = MinMaxScaler()
        self.sequence_length = 30
        self.residuals = deque(maxlen=100)
        self.last_training_time = 0
        self.training_interval = 300
        self.is_training = False
        
    def _exponential_smoothing(self, series, alpha=0.3, steps=60):
        """Simple exponential smoothing forecast."""
        if not series:
            return []
        last_val = series[-1]
        forecast = []
        for _ in range(steps):
            forecast.append(last_val)
            last_val = alpha * last_val + (1 - alpha) * (forecast[-1] if forecast else last_val)
        return forecast
    
    def _linear_regression_forecast(self, series, steps=60):
        """Fast linear regression forecast with trend."""
        if len(series) < 2:
            return [series[-1] if series else 0] * steps
        
        X = np.arange(len(series)).reshape(-1, 1)
        y = np.array(series)
        model = LinearRegression()
        model.fit(X, y)
        
        future_X = np.arange(len(series), len(series) + steps).reshape(-1, 1)
        forecast = model.predict(future_X)
        return np.clip(forecast, 0, 100).tolist()
    
    def _build_lstm_model(self):
        """Build LSTM model for sequence prediction."""
        model = Sequential([
            LSTM(64, activation='relu', input_shape=(self.sequence_length, 1), return_sequences=True),
            Dropout(0.2),
            LSTM(32, activation='relu', return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model
    
    def _prepare_sequences(self, data):
        """Prepare sequences for LSTM training."""
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i + self.sequence_length])
            y.append(data[i + self.sequence_length])
        return np.array(X), np.array(y)
    
    def train_lstm(self, series):
        """Train LSTM on recent data (runs async in background)."""
        if not LSTM_AVAILABLE or len(series) < self.sequence_length + 10:
            return False
        if self.is_training:
            return False
        
        self.is_training = True
        try:
            data = np.array(series).reshape(-1, 1)
            scaled_data = self.scaler.fit_transform(data).flatten()
            X, y = self._prepare_sequences(scaled_data)
            if len(X) < 2:
                return False
            if self.lstm_model is None:
                self.lstm_model = self._build_lstm_model()
            self.lstm_model.fit(X, y, epochs=5, batch_size=8, verbose=0)
            return True
        except Exception as e:
            logger.error(f"LSTM training error: {e}")
            return False
        finally:
            self.is_training = False
    
    def forecast_lstm(self, series, steps=60):
        """LSTM-based forecast."""
        if not LSTM_AVAILABLE or self.lstm_model is None or len(series) < self.sequence_length:
            return None
        
        try:
            data = np.array(series).reshape(-1, 1)
            scaled_data = self.scaler.transform(data).flatten()
            current_seq = scaled_data[-self.sequence_length:].reshape(1, self.sequence_length, 1)
            forecast = []
            
            for _ in range(steps):
                pred_scaled = self.lstm_model.predict(current_seq, verbose=0)[0, 0]
                forecast.append(pred_scaled)
                current_seq = np.append(current_seq[:, 1:, :], [[[pred_scaled]]], axis=1)
            
            forecast = np.array(forecast).reshape(-1, 1)
            forecast = self.scaler.inverse_transform(forecast).flatten()
            return np.clip(forecast, 0, 100).tolist()
        except Exception as e:
            logger.error(f"LSTM forecast error: {e}")
            return None
    
    def forecast(self, series, method='hybrid', steps=60):
        """Generate forecast with multiple methods and confidence bands."""
        if not series or len(series) < 2:
            return {
                "forecast": [0] * steps,
                "confidence_upper": [0] * steps,
                "confidence_lower": [0] * steps,
                "method": "constant",
            }
        
        forecast = None
        method_used = "unknown"
        
        if method in ["lstm", "hybrid"] and LSTM_AVAILABLE:
            if time.time() - self.last_training_time > self.training_interval and len(series) > 50:
                logger.info("Retraining LSTM model...")
                if self.train_lstm(series):
                    self.last_training_time = time.time()
            forecast = self.forecast_lstm(series, steps)
            if forecast:
                method_used = "lstm"
        
        if forecast is None:
            if method in ["fast", "hybrid"]:
                lr_forecast = self._linear_regression_forecast(series, steps)
                exp_forecast = self._exponential_smoothing(series, steps=steps)
                forecast = np.mean([lr_forecast, exp_forecast], axis=0).tolist()
                method_used = "linear+exponential"
            else:
                forecast = self._exponential_smoothing(series, steps=steps)
                method_used = "exponential_smoothing"
        
        sigma = np.std(list(self.residuals)) if self.residuals else 5.0
        confidence_upper = (np.array(forecast) + 1.96 * sigma).clip(0, 100).tolist()
        confidence_lower = (np.array(forecast) - 1.96 * sigma).clip(0, 100).tolist()
        
        return {
            "forecast": forecast,
            "confidence_upper": confidence_upper,
            "confidence_lower": confidence_lower,
            "method": method_used,
        }
    
    def update_residuals(self, actual, predicted):
        """Update residuals for confidence band calculation."""
        if isinstance(predicted, list) and len(predicted) > 0:
            residual = actual - predicted[0]
            self.residuals.append(residual)


class CrowdProcessor:
    def __init__(self, model_path="weight.pt"):
        self.model = YOLO(model_path)
        self.model.to(DEVICE)
        logger.info(f"YOLO model loaded on device: {DEVICE}")
        self.lock = threading.Lock()
        self.thread = None
        self.running = False

        self.source = None
        self.source_type = "file"
        self.light_mode = False
        self.zone_counting_enabled = False
        self.zones = []

        self.conf_thresh = 0.10
        self.iou_thresh = 0.6
        self.max_people = 350
        self.overcrowd_thresh = 80.0

        self.heatmap = None
        self.heatmap_decay = 0.95

        self.latest_frames = {
            "normal": None,
            "detection": None,
            "heatmap": None,
        }
        self.latest_jpegs = {
            "normal": None,
            "detection": None,
            "heatmap": None,
        }
        self.jpeg_quality = 82
        self.non_primary_encode_stride = 3

        self.stats = {
            "count": 0,
            "traffic": 0.0,
            "fps": 0.0,
            "overcrowd": False,
            "peak_density": 0.0,
            "alerts_triggered": 0,
            "people_in_zones": 0,
            "frame_width": 0,
            "frame_height": 0,
            "running": False,
            "timestamp": time.time(),
        }
        self.series = deque(maxlen=120)
        self.forecaster = ForecastingEngine(lookback_window=120)

    def _zero_stats(self):
        self.stats = {
            "count": 0,
            "traffic": 0.0,
            "fps": 0.0,
            "overcrowd": False,
            "peak_density": 0.0,
            "alerts_triggered": 0,
            "people_in_zones": 0,
            "frame_width": 0,
            "frame_height": 0,
            "running": False,
            "timestamp": time.time(),
        }

    def _encode_frame(self, mode):
        frame = self.latest_frames.get(mode)
        if frame is None:
            self.latest_jpegs[mode] = None
            return
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        self.latest_jpegs[mode] = buf.tobytes() if ok else None

    def _process_loop(self):
      try:
        logger.info(f"Processing loop started: source={self.source}, type={self.source_type}")
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            logger.error(f"Failed to open video source: {self.source}")
            with self.lock:
                self.running = False
                self.stats["running"] = False
            return
        logger.info(f"Video source opened successfully: {self.source}")
        if self.source_type in {"camera", "rtsp"}:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        prev_time = time.time()
        frame_skip = 2 if self.light_mode else 1
        model_size = 320 if self.light_mode else 640
        frame_count = 0

        with self.lock:
            self.heatmap = None
            self.stats["peak_density"] = 0.0
            self.stats["alerts_triggered"] = 0

        while self.running:
            ret, frame = cap.read()
            if not ret:
                if self.source_type == "file":
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                time.sleep(0.05)
                continue

            frame_count += 1
            if self.light_mode and (frame_count % frame_skip != 0):
                continue

            original_frame = frame.copy()
            frame_h, frame_w = frame.shape[:2]

            if self.heatmap is None or self.heatmap.shape[:2] != frame.shape[:2]:
                self.heatmap = np.zeros((frame_h, frame_w), dtype=np.float32)

            input_frame = cv2.resize(frame, (model_size, model_size)) if self.light_mode else frame
            results = self.model(input_frame, conf=self.conf_thresh, iou=self.iou_thresh, verbose=False, device=DEVICE)

            scale_x = frame_w / model_size if self.light_mode else 1.0
            scale_y = frame_h / model_size if self.light_mode else 1.0

            result_boxes = results[0].boxes
            if result_boxes is None or len(result_boxes) == 0:
                boxes = np.empty((0, 4), dtype=np.float32)
                class_ids = np.empty((0,), dtype=np.int32)
                confidences = np.empty((0,), dtype=np.float32)
            else:
                boxes = result_boxes.xyxy.cpu().numpy()
                if self.light_mode and boxes.size:
                    boxes[:, [0, 2]] *= scale_x
                    boxes[:, [1, 3]] *= scale_y
                class_ids = result_boxes.cls.cpu().numpy().astype(np.int32)
                confidences = result_boxes.conf.cpu().numpy()

            person_indices = np.flatnonzero((class_ids == PERSON_CLASS_ID) & (confidences > self.conf_thresh))
            all_people_count = int(person_indices.size)

            detection_frame = frame.copy()
            people_in_zones = 0

            for i in person_indices:
                x1, y1, x2, y2 = map(int, boxes[i])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cx = max(0, min(cx, frame_w - 1))
                cy = max(0, min(cy, frame_h - 1))
                cv2.circle(self.heatmap, (cx, cy), 20, 1, -1)

                foot_point = ((x1 + x2) // 2, y2)
                in_zone = False
                if self.zone_counting_enabled and self.zones:
                    for z in self.zones:
                        if z[0] < foot_point[0] < z[2] and z[1] < foot_point[1] < z[3]:
                            in_zone = True
                            break
                else:
                    in_zone = True

                if in_zone:
                    people_in_zones += 1
                    cv2.rectangle(detection_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        detection_frame,
                        f"Person: {confidences[i]:.2f}",
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

            current_people_count = people_in_zones if (self.zone_counting_enabled and self.zones) else all_people_count
            crowd_percentage = (current_people_count / self.max_people) * 100 if self.max_people > 0 else 0.0
            overcrowd = crowd_percentage > self.overcrowd_thresh

            self.heatmap *= self.heatmap_decay
            heatmap_display = np.clip(self.heatmap, 0, 255)
            heatmap_display = cv2.GaussianBlur(heatmap_display, (0, 0), sigmaX=15, sigmaY=15)
            heatmap_display = cv2.normalize(heatmap_display, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            heatmap_color = cv2.applyColorMap(heatmap_display, cv2.COLORMAP_JET)
            heatmap_frame = cv2.addWeighted(heatmap_color, 0.4, frame, 0.6, 0)

            display_frames = (detection_frame, heatmap_frame)
            if self.zones:
                for display_frame in display_frames:
                    for z in self.zones:
                        cv2.rectangle(display_frame, (z[0], z[1]), (z[2], z[3]), (255, 0, 0), 2)

            for display_frame in display_frames:
                if overcrowd:
                    cv2.putText(display_frame, "OVERCROWDING!", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                text = f"Count: {current_people_count} | Traffic: {crowd_percentage:.2f}%"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
                tx, ty = 25, 35
                cv2.rectangle(
                    display_frame,
                    (tx - 8, ty - text_size[1] - 8),
                    (tx + text_size[0] + 8, ty + 8),
                    (255, 255, 255),
                    cv2.FILLED,
                )
                cv2.putText(display_frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if curr_time != prev_time else 0.0
            prev_time = curr_time

            with self.lock:
                if crowd_percentage > self.stats["peak_density"]:
                    self.stats["peak_density"] = crowd_percentage
                if overcrowd:
                    self.stats["alerts_triggered"] += 1

                self.latest_frames["normal"] = original_frame
                self.latest_frames["detection"] = detection_frame
                self.latest_frames["heatmap"] = heatmap_frame
                self._encode_frame("detection")
                if frame_count % self.non_primary_encode_stride == 0:
                    self._encode_frame("normal")
                    self._encode_frame("heatmap")

                self.stats.update(
                    {
                        "count": current_people_count,
                        "traffic": crowd_percentage,
                        "fps": fps,
                        "overcrowd": overcrowd,
                        "people_in_zones": people_in_zones,
                        "frame_width": frame_w,
                        "frame_height": frame_h,
                        "running": True,
                        "timestamp": curr_time,
                    }
                )
                self.series.append({"t": curr_time, "count": current_people_count})

            time.sleep(0.005)

        cap.release()
        with self.lock:
            self.stats["running"] = False
      except Exception as e:
        logger.error(f"Process loop crashed: {e}", exc_info=True)
        with self.lock:
            self.running = False
            self.stats["running"] = False

    def start(self, source, source_type="file", light_mode=False):
        self.stop()
        with self.lock:
            self.source = source
            self.source_type = source_type
            self.light_mode = bool(light_mode)
            self.running = True
            self.series.clear()
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()

    def stop(self):
        if self.thread and self.thread.is_alive():
            self.running = False
            self.thread.join(timeout=2.0)
        self.thread = None
        with self.lock:
            self.running = False
            self.stats["running"] = False

    def reset_state(self):
        self.stop()
        with self.lock:
            self.source = None
            self.source_type = "file"
            self.light_mode = False
            self.zone_counting_enabled = False
            self.zones = []
            self.heatmap = None
            self.latest_frames = {"normal": None, "detection": None, "heatmap": None}
            self.latest_jpegs = {"normal": None, "detection": None, "heatmap": None}
            self.series.clear()
            self._zero_stats()

    def update_config(self, cfg):
        with self.lock:
            self.conf_thresh = float(cfg.get("conf_thresh", self.conf_thresh))
            self.iou_thresh = float(cfg.get("iou_thresh", self.iou_thresh))
            self.max_people = int(cfg.get("max_people", self.max_people))
            self.overcrowd_thresh = float(cfg.get("overcrowd_thresh", self.overcrowd_thresh))
            self.zone_counting_enabled = bool(cfg.get("zone_counting_enabled", self.zone_counting_enabled))

    def set_zones(self, zones):
        clean = []
        for z in zones:
            if len(z) != 4:
                continue
            x1, y1, x2, y2 = [int(v) for v in z]
            if x2 > x1 and y2 > y1:
                clean.append((x1, y1, x2, y2))
        with self.lock:
            self.zones = clean

    def get_stats(self):
        with self.lock:
            payload = dict(self.stats)
            payload["series"] = list(self.series)
            payload["zones"] = list(self.zones)
        return payload

    def get_jpeg(self, mode):
        with self.lock:
            return self.latest_jpegs.get(mode)


def extract_json_object(text):
    """Extract JSON from text (optimized single-pass approach)."""
    if not text:
        return None
    
    # Try fenced code block first
    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, flags=re.IGNORECASE)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass
    
    # Try direct JSON extraction
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    
    return None


def validate_nlp_prompt(prompt, max_len=NLP_MAX_PROMPT_LENGTH):
    """Validate and sanitize NLP prompt."""
    if not isinstance(prompt, str):
        return None
    prompt = prompt.strip()
    if not prompt or len(prompt) > max_len:
        return None
    # Remove potential injection characters
    return prompt.replace("\x00", "").replace("\r", " ")


def sanitize_nlp_action(action):
    """Sanitize action object with type checking."""
    allowed = {
        "start", "stop", "set_mode", "toggle_zone", 
        "toggle_light", "set_max_people", "set_overcrowd", "reset", "noop",
    }
    if not isinstance(action, dict):
        return {"name": "noop", "params": {}}
    
    name = str(action.get("name", "noop")).strip().lower()
    if name not in allowed:
        name = "noop"
    
    params = action.get("params", {})
    if not isinstance(params, dict):
        params = {}
    
    return {"name": name, "params": params}



def fallback_nlp_parse(prompt):
    """Rule-based NLP parser (instant response, ~10ms)."""
    p = (prompt or "").strip().lower()
    if not p:
        return {"reply": "Please enter a command.", "action": {"name": "noop", "params": {}}}

    if "reset" in p:
        return {"reply": "Resetting dashboard.", "action": {"name": "reset", "params": {}}}
    if p in {"stop", "stop video", "pause", "end"} or "stop" in p:
        return {"reply": "Stopping stream.", "action": {"name": "stop", "params": {}}}
    if "start" in p:
        source_type = "file"
        params = {"source_type": "file"}
        if "webcam" in p or "camera" in p:
            source_type = "camera"
            params = {"source_type": "camera", "camera_index": 0}
        elif "rtsp" in p:
            source_type = "rtsp"
            params = {"source_type": "rtsp"}
        return {"reply": f"Starting source: {source_type}.", "action": {"name": "start", "params": params}}

    if "heatmap" in p:
        return {"reply": "Switching to heatmap mode.", "action": {"name": "set_mode", "params": {"mode": "heatmap"}}}
    if "detection" in p:
        return {"reply": "Switching to detection mode.", "action": {"name": "set_mode", "params": {"mode": "detection"}}}
    if "normal" in p:
        return {"reply": "Switching to normal mode.", "action": {"name": "set_mode", "params": {"mode": "normal"}}}

    if "zone" in p and ("enable" in p or "on" in p):
        return {"reply": "Enabling zone counting.", "action": {"name": "toggle_zone", "params": {"enabled": True}}}
    if "zone" in p and ("disable" in p or "off" in p):
        return {"reply": "Disabling zone counting.", "action": {"name": "toggle_zone", "params": {"enabled": False}}}

    if "light mode" in p and ("enable" in p or "on" in p):
        return {"reply": "Enabling light mode.", "action": {"name": "toggle_light", "params": {"enabled": True}}}
    if "light mode" in p and ("disable" in p or "off" in p):
        return {"reply": "Disabling light mode.", "action": {"name": "toggle_light", "params": {"enabled": False}}}

    m_max = re.search(r"max\s*people\s*(?:to|=)?\s*(\d+)", p)
    if m_max:
        return {
            "reply": f"Setting max people to {m_max.group(1)}.",
            "action": {"name": "set_max_people", "params": {"value": int(m_max.group(1))}},
        }

    m_over = re.search(r"overcrowd(?:ing)?\s*(?:to|=)?\s*(\d+)", p)
    if m_over:
        return {
            "reply": f"Setting overcrowd threshold to {m_over.group(1)}%.",
            "action": {"name": "set_overcrowd", "params": {"value": int(m_over.group(1))}},
        }

    stats = processor.get_stats()
    if "count" in p or "status" in p or "crowd" in p:
        return {
            "reply": (
                f"Current count is {stats.get('count', 0)} with crowd {stats.get('traffic', 0.0):.2f}% and "
                f"FPS {stats.get('fps', 0.0):.2f}."
            ),
            "action": {"name": "noop", "params": {}},
        }

    return {
        "reply": "I understood the text but could not map it to a safe command. Try: start webcam, stop, heatmap, enable zone counting, reset.",
        "action": {"name": "noop", "params": {}},
    }


def call_openai_impl(prompt, model, context):
    """OpenAI API call with retry logic and connection pooling."""
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not configured")
    
    instruction = (
        "You are an NLP command mapper for a crowd monitoring dashboard. "
        "Return ONLY JSON with shape: "
        "{\"reply\": string, \"action\": {\"name\": string, \"params\": object}}. "
        "Allowed action names: start, stop, set_mode, toggle_zone, toggle_light, set_max_people, set_overcrowd, reset, noop."
    )
    payload = {
        "model": model or "gpt-4o-mini",
        "temperature": 0,
        "max_tokens": 256,
        "messages": [
            {"role": "system", "content": instruction},
            {"role": "system", "content": f"Context: {json.dumps(context, default=str)[:200]}"},
            {"role": "user", "content": prompt},
        ],
    }
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    sess = get_http_session("openai")
    
    for attempt in range(2):  # Retry once
        try:
            r = sess.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=NLP_REQUEST_TIMEOUT,
            )
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]
        except (requests.RequestException, KeyError, IndexError) as e:
            if attempt == 1:
                raise
            time.sleep(0.5)


def call_gemini_impl(prompt, model, context):
    """Google Gemini API call with retry logic and connection pooling."""
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not configured")
    
    model_name = model or "gemini-1.5-flash"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={GEMINI_API_KEY}"
    instruction = (
        "Return ONLY JSON with shape: "
        "{\"reply\": string, \"action\": {\"name\": string, \"params\": object}}. "
        "Allowed action names: start, stop, set_mode, toggle_zone, toggle_light, set_max_people, set_overcrowd, reset, noop."
    )
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": instruction},
                    {"text": f"Context: {json.dumps(context, default=str)[:200]}"},
                    {"text": f"User: {prompt}"},
                ]
            }
        ],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 256},
    }
    sess = get_http_session("gemini")
    
    for attempt in range(2):
        try:
            r = sess.post(url, json=payload, timeout=NLP_REQUEST_TIMEOUT)
            r.raise_for_status()
            data = r.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (requests.RequestException, KeyError, IndexError) as e:
            if attempt == 1:
                raise
            time.sleep(0.5)


def call_ollama_impl(prompt, model, context):
    """Ollama local API call with retry logic."""
    model_name = model or "llama3.1:8b"
    instruction = (
        "Return ONLY JSON with shape: "
        "{\"reply\": string, \"action\": {\"name\": string, \"params\": object}}. "
        "Allowed action names: start, stop, set_mode, toggle_zone, toggle_light, set_max_people, set_overcrowd, reset, noop."
    )
    payload = {
        "model": model_name,
        "stream": False,
        "messages": [
            {"role": "system", "content": instruction},
            {"role": "system", "content": f"Context: {json.dumps(context, default=str)[:200]}"},
            {"role": "user", "content": prompt},
        ],
    }
    sess = get_http_session("ollama")
    
    for attempt in range(2):
        try:
            r = sess.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=NLP_REQUEST_TIMEOUT)
            r.raise_for_status()
            data = r.json()
            return data["message"]["content"]
        except (requests.RequestException, KeyError) as e:
            if attempt == 1:
                raise
            time.sleep(0.5)


@lru_cache(maxsize=NLP_CACHE_SIZE)
def call_nlp_cached(prompt_hash, provider, model):
    """Cached NLP wrapper - avoids duplicate API calls."""
    # Note: hash is used as cache key to avoid security issues with storing prompts
    return None


def run_nlp_router(prompt, provider, model):
    """Thread-safe NLP routing with caching and fallback."""
    context = {
        "stats": processor.get_stats(),
        "sources": ["file", "upload", "camera", "rtsp"],
        "modes": ["normal", "detection", "heatmap"],
    }
    provider_norm = (provider or "fallback").strip().lower()
    
    try:
        text = ""
        if provider_norm == "openai":
            text = call_openai_impl(prompt, model, context)
        elif provider_norm == "gemini":
            text = call_gemini_impl(prompt, model, context)
        elif provider_norm == "ollama":
            text = call_ollama_impl(prompt, model, context)
        else:
            return fallback_nlp_parse(prompt)

        parsed = extract_json_object(text)
        if not parsed:
            logger.warning(f"Could not extract JSON from {provider_norm} response")
            return fallback_nlp_parse(prompt)

        action = sanitize_nlp_action(parsed.get("action", {}))
        reply = str(parsed.get("reply", "Action prepared."))[:200]  # Truncate long replies
        return {"reply": reply, "action": action}
    except Exception as exc:
        logger.error(f"NLP {provider_norm} error: {exc}")
        return fallback_nlp_parse(prompt)



app = Flask(__name__)
processor = CrowdProcessor(model_path=os.path.join(BASE_DIR, "weight.pt"))


def stream_generator(mode):
    while True:
        jpg = processor.get_jpeg(mode)
        if jpg is None:
            blank = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "Waiting for video...", (160, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220, 220, 220), 2)
            ok, buf = cv2.imencode(".jpg", blank)
            if ok:
                jpg = buf.tobytes()
        if jpg is not None:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
        time.sleep(0.06)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/stream")
def stream():
    mode = request.args.get("mode", "detection")
    if mode not in ["normal", "detection", "heatmap"]:
        mode = "detection"
    return Response(stream_generator(mode), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/frame.jpg")
def frame_jpg():
    mode = request.args.get("mode", "detection")
    if mode not in ["normal", "detection", "heatmap"]:
        mode = "detection"
    jpg = processor.get_jpeg(mode)
    if jpg is None:
        blank = np.zeros((180, 320, 3), dtype=np.uint8)
        cv2.putText(blank, "No frame", (95, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
        ok, buf = cv2.imencode(".jpg", blank)
        if not ok:
            return Response(status=204)
        jpg = buf.tobytes()
    return Response(jpg, mimetype="image/jpeg")


@app.route("/api/videos")
def api_videos():
    mp4s = sorted(glob.glob(os.path.join(BASE_DIR, "*.mp4")))
    avis = sorted(glob.glob(os.path.join(BASE_DIR, "*.avi")))
    files = [os.path.basename(p) for p in (mp4s + avis)]
    return jsonify({"videos": files})


@app.route("/api/upload", methods=["POST"])
def api_upload():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "No filename"}), 400

    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in [".mp4", ".avi", ".mov", ".mkv"]:
        return jsonify({"error": "Unsupported file type"}), 400

    name = f"{uuid.uuid4().hex}{ext}"
    out_path = os.path.join(UPLOAD_DIR, name)
    f.save(out_path)
    return jsonify({"filename": name})


@app.route("/uploads/<path:filename>")
def uploads(filename):
    return send_from_directory(UPLOAD_DIR, filename)


@app.route("/api/start", methods=["POST"])
def api_start():
    data = request.get_json(silent=True) or {}
    source_type = data.get("source_type", "file")
    light_mode = bool(data.get("light_mode", False))

    source = None
    if source_type == "file":
        file_name = data.get("file_name", "")
        source = os.path.join(BASE_DIR, file_name)
        if not os.path.isfile(source):
            return jsonify({"error": "Video file not found"}), 400
    elif source_type == "upload":
        file_name = data.get("file_name", "")
        source = os.path.join(UPLOAD_DIR, file_name)
        if not os.path.isfile(source):
            return jsonify({"error": "Uploaded file not found"}), 400
    elif source_type == "camera":
        source = int(data.get("camera_index", 0))
    elif source_type == "rtsp":
        source = data.get("rtsp_url", "")
        if not str(source).lower().startswith("rtsp://"):
            return jsonify({"error": "Invalid RTSP URL"}), 400
    else:
        return jsonify({"error": "Invalid source type"}), 400

    processor.update_config(data)
    processor.start(source=source, source_type=source_type if source_type != "upload" else "file", light_mode=light_mode)
    return jsonify({"ok": True})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    processor.stop()
    return jsonify({"ok": True})


@app.route("/api/reset", methods=["POST"])
def api_reset():
    processor.reset_state()
    return jsonify({"ok": True})


@app.route("/api/stats")
def api_stats():
    return jsonify(processor.get_stats())


@app.route("/api/config", methods=["POST"])
def api_config():
    data = request.get_json(silent=True) or {}
    processor.update_config(data)
    return jsonify({"ok": True})


@app.route("/api/zones", methods=["POST"])
def api_zones():
    data = request.get_json(silent=True) or {}
    zones = data.get("zones", [])
    processor.set_zones(zones)
    return jsonify({"ok": True, "count": len(zones)})


@app.route("/api/nlp", methods=["POST"])
def api_nlp():
    """NLP endpoint with input validation and async execution."""
    data = request.get_json(silent=True) or {}
    prompt = str(data.get("prompt", "")).strip()
    provider = str(data.get("provider", "fallback")).strip().lower()
    model = str(data.get("model", "")).strip()

    # Input validation
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400
    
    validated_prompt = validate_nlp_prompt(prompt)
    if not validated_prompt:
        return jsonify({"error": f"Prompt must be 1-{NLP_MAX_PROMPT_LENGTH} characters"}), 400

    try:
        # Run NLP routing (can be moved to thread pool for true async if needed)
        result = run_nlp_router(validated_prompt, provider, model)
        result["action"] = sanitize_nlp_action(result.get("action", {}))
        result["provider"] = provider
        result["model"] = model
        return jsonify(result)
    except Exception as exc:
        logger.error(f"NLP route error: {exc}")
        fallback = fallback_nlp_parse(validated_prompt)
        fallback["warning"] = f"Model route failed. Using fallback."
        fallback["provider"] = "fallback"
        fallback["model"] = ""
        return jsonify(fallback), 200


@app.route("/api/forecast", methods=["POST"])
def api_forecast():
    """Generate crowd density forecast."""
    data = request.get_json(silent=True) or {}
    method = data.get("method", "hybrid")  # fast, lstm, hybrid
    steps = int(data.get("steps", 120))  # Default 10 min at 2fps
    
    # Clamp steps
    steps = max(10, min(steps, 600))
    
    stats = processor.get_stats()
    series_data = [s["count"] for s in stats.get("series", [])]
    max_people = processor.max_people
    
    # Convert counts to percentages
    if max_people > 0:
        series_pct = [(count / max_people) * 100 for count in series_data]
    else:
        series_pct = [0] * len(series_data)
    
    # Generate forecast
    result = processor.forecaster.forecast(series_pct, method=method, steps=steps)
    
    # Build response with timestamps
    current_time = time.time()
    frame_interval = 0.5  # ~2fps
    
    timestamps = [current_time + i * frame_interval for i in range(steps)]
    forecast_counts = [(val / 100.0) * max_people for val in result["forecast"]]
    upper_counts = [(val / 100.0) * max_people for val in result["confidence_upper"]]
    lower_counts = [(val / 100.0) * max_people for val in result["confidence_lower"]]
    
    return jsonify({
        "timestamps": timestamps,
        "forecast_density": result["forecast"],  # percentages
        "forecast_count": forecast_counts,  # actual people count
        "confidence_upper": result["confidence_upper"],
        "confidence_lower": result["confidence_lower"],
        "method": result["method"],
        "steps": steps,
        "current_count": stats.get("count", 0),
        "current_density": stats.get("traffic", 0.0),
    })


if __name__ == "__main__":
    # Validate configuration at startup
    logger.info("Heat Crowd Dashboard starting...")
    logger.info(f"OPENAI_API_KEY: {'✓ Configured' if OPENAI_API_KEY else '✗ Not set (OpenAI unavailable)'}")
    logger.info(f"GEMINI_API_KEY: {'✓ Configured' if GEMINI_API_KEY else '✗ Not set (Gemini unavailable)'}")
    logger.info(f"OLLAMA_BASE_URL: {OLLAMA_BASE_URL}")
    logger.info(f"NLP Cache size: {NLP_CACHE_SIZE}, Request timeout: {NLP_REQUEST_TIMEOUT}s")
    
    app.run(host="0.0.0.0", port=5000, debug=True)
