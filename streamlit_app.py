import os
import glob
import time
import threading
import json
import re
import urllib.request
from collections import deque

import cv2
import numpy as np
import requests
import pandas as pd
from ultralytics import YOLO

import streamlit as st
import folium
from streamlit_folium import st_folium
from PIL import Image

try:
    from streamlit_drawable_canvas import st_canvas
except ImportError:
    st_canvas = None

PERSON_CLASS_ID = 0
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class CrowdProcessor:
    def __init__(self, model_path="weight.pt"):
        self.model = YOLO(model_path)
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

        self.latest_frames = {"normal": None, "detection": None, "heatmap": None}
        
        self.stats = {}
        self.series = deque(maxlen=120)
        self._zero_stats()

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

    def _process_loop(self):
        cap = cv2.VideoCapture(self.source)
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
            results = self.model(input_frame, conf=self.conf_thresh, iou=self.iou_thresh, verbose=False)

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

                self.stats.update({
                    "count": current_people_count,
                    "traffic": crowd_percentage,
                    "fps": fps,
                    "overcrowd": overcrowd,
                    "people_in_zones": people_in_zones,
                    "frame_width": frame_w,
                    "frame_height": frame_h,
                    "running": True,
                    "timestamp": curr_time,
                })
                self.series.append({"t": curr_time, "count": current_people_count})

            time.sleep(0.005)

        cap.release()
        with self.lock:
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
            self.series.clear()
            self._zero_stats()

    def set_zones(self, zones):
        clean = []
        for z in zones:
            if len(z) == 4:
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

    def get_rgb_frame(self, mode):
        with self.lock:
            frame = self.latest_frames.get(mode)
            if frame is not None:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return None

# ==========================================
# NLP Parsing Logic
# ==========================================

def extract_json_object(text):
    if not text: return None
    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", text, flags=re.IGNORECASE)
    if fenced: text = fenced.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start: return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None

def sanitize_nlp_action(action):
    allowed = {"start", "stop", "set_mode", "toggle_zone", "toggle_light", "set_max_people", "set_overcrowd", "reset", "noop"}
    if not isinstance(action, dict): return {"name": "noop", "params": {}}
    name = str(action.get("name", "noop")).strip().lower()
    if name not in allowed: name = "noop"
    params = action.get("params", {})
    if not isinstance(params, dict): params = {}
    return {"name": name, "params": params}

def fallback_nlp_parse(prompt, processor):
    p = (prompt or "").strip().lower()
    if not p: return {"reply": "Please enter a command.", "action": {"name": "noop", "params": {}}}
    if "reset" in p: return {"reply": "Resetting dashboard.", "action": {"name": "reset", "params": {}}}
    if p in {"stop", "stop video", "pause", "end"} or "stop" in p: return {"reply": "Stopping stream.", "action": {"name": "stop", "params": {}}}
    
    if "start" in p:
        source_type = "file"
        params = {"source_type": "file"}
        if "webcam" in p or "camera" in p:
            params = {"source_type": "camera", "camera_index": 0}
        elif "rtsp" in p:
            params = {"source_type": "rtsp"}
        return {"reply": f"Starting.", "action": {"name": "start", "params": params}}

    if "heatmap" in p: return {"reply": "Switching to heatmap.", "action": {"name": "set_mode", "params": {"mode": "heatmap"}}}
    if "detection" in p: return {"reply": "Switching to detection.", "action": {"name": "set_mode", "params": {"mode": "detection"}}}
    if "normal" in p: return {"reply": "Switching to normal mode.", "action": {"name": "set_mode", "params": {"mode": "normal"}}}

    if "zone" in p and ("enable" in p or "on" in p): return {"reply": "Enabling zone counting.", "action": {"name": "toggle_zone", "params": {"enabled": True}}}
    if "zone" in p and ("disable" in p or "off" in p): return {"reply": "Disabling zone counting.", "action": {"name": "toggle_zone", "params": {"enabled": False}}}

    m_max = re.search(r"max\s*people\s*(?:to|=)?\s*(\d+)", p)
    if m_max: return {"reply": f"Setting max people to {m_max.group(1)}.", "action": {"name": "set_max_people", "params": {"value": int(m_max.group(1))}}}

    stats = processor.get_stats()
    if "count" in p or "status" in p or "crowd" in p:
        return {"reply": f"Current count is {stats.get('count', 0)} with crowd {stats.get('traffic', 0.0):.2f}%.", "action": {"name": "noop", "params": {}}}

    return {"reply": "Command not fully understood. Try: start webcam, stop, heatmap, reset.", "action": {"name": "noop", "params": {}}}

def call_openai(prompt, model, context):
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key: raise RuntimeError("OPENAI_API_KEY is not set")
    instruction = "Return ONLY JSON: {\"reply\": string, \"action\": {\"name\": string, \"params\": object}}. Allowed actions: start, stop, set_mode, toggle_zone, toggle_light, set_max_people, set_overcrowd, reset, noop."
    payload = {
        "model": model or "gpt-4o-mini",
        "temperature": 0,
        "messages": [{"role": "system", "content": instruction}, {"role": "system", "content": f"Context: {json.dumps(context)}"}, {"role": "user", "content": prompt}],
    }
    r = requests.post("https://api.openai.com/v1/chat/completions", headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, json=payload, timeout=25)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def call_gemini(prompt, model, context):
    api_key = os.getenv("GEMINI_API_KEY", "AIzaSyAjb2HZYbNld3wnztF47VFiFPwckq50p9Y").strip()
    if not api_key: raise RuntimeError("GEMINI_API_KEY is not set")
    model_name = model or "gemini-1.5-flash"
    url = f"https://generativelanguage.googleapis.com/v1/models/{model_name}:generateContent?key={api_key}"
    instruction = "Return ONLY JSON: {\"reply\": string, \"action\": {\"name\": string, \"params\": object}}. Allowed actions: start, stop, set_mode, toggle_zone, toggle_light, set_max_people, set_overcrowd, reset, noop."
    payload = {"contents": [{"parts": [{"text": instruction}, {"text": f"Context: {json.dumps(context)}"}, {"text": f"User: {prompt}"}]}]}
    r = requests.post(url, json=payload, timeout=25)
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]

def call_ollama(prompt, model, context):
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    model_name = model or "llama3.1:8b"
    instruction = "Return ONLY JSON: {\"reply\": string, \"action\": {\"name\": string, \"params\": object}}. Allowed actions: start, stop, set_mode, toggle_zone, toggle_light, set_max_people, set_overcrowd, reset, noop."
    payload = {"model": model_name, "stream": False, "messages": [{"role": "system", "content": instruction}, {"role": "system", "content": f"Context: {json.dumps(context)}"}, {"role": "user", "content": prompt}]}
    r = requests.post(f"{base_url}/api/chat", json=payload, timeout=25)
    r.raise_for_status()
    return r.json().get("message", {}).get("content", "")

def run_nlp_router(prompt, provider, model, processor):
    context = {"stats": processor.get_stats(), "sources": ["file", "upload", "camera", "rtsp"], "modes": ["normal", "detection", "heatmap"]}
    provider_norm = (provider or "fallback").strip().lower()
    text = ""
    try:
        if provider_norm == "openai": text = call_openai(prompt, model, context)
        elif provider_norm == "gemini": text = call_gemini(prompt, model, context)
        elif provider_norm == "ollama": text = call_ollama(prompt, model, context)
        else: return fallback_nlp_parse(prompt, processor)
    except Exception as e:
        res = fallback_nlp_parse(prompt, processor)
        res["reply"] += f" (Fallback used. Error: {str(e)})"
        return res

    parsed = extract_json_object(text)
    if not parsed: return fallback_nlp_parse(prompt, processor)
    return {"reply": str(parsed.get("reply", "Action prepared.")), "action": sanitize_nlp_action(parsed.get("action", {}))}

# ==========================================
# Streamlit UI
# ==========================================

st.set_page_config(page_title="Heat Crowd Streamlit", layout="wide")

# Custom CSS for gradients
st.markdown("""
<style>
    .stApp {
        background: radial-gradient(circle at 10% 10%, rgba(67, 179, 255, 0.1), transparent 34%),
                    radial-gradient(circle at 82% 22%, rgba(39, 211, 181, 0.08), transparent 32%),
                    linear-gradient(145deg, #08121f, #0f1f34 45%, #1a2e4d);
        color: #edf4ff;
    }
    .css-1d391kg, .st-emotion-cache-16txtl3 { 
        padding-top: 2rem; 
    }
</style>
""", unsafe_allow_html=True)

if 'processor' not in st.session_state:
    st.session_state.processor = CrowdProcessor()
if 'view_mode' not in st.session_state:
    st.session_state.view_mode = "detection"
if 'nlp_reply' not in st.session_state:
    st.session_state.nlp_reply = ""
if 'map_lat' not in st.session_state:
    st.session_state.map_lat = 28.6765
if 'map_lon' not in st.session_state:
    st.session_state.map_lon = 77.5005

processor = st.session_state.processor

def get_video_files():
    mp4s = sorted(glob.glob(os.path.join(BASE_DIR, "*.mp4")))
    avis = sorted(glob.glob(os.path.join(BASE_DIR, "*.avi")))
    return [os.path.basename(p) for p in (mp4s + avis)]

def execute_action(action):
    name = action.get("name")
    p = action.get("params", {})
    if name == "start":
        if "source_type" in p:
            st.session_state["source_type"] = p["source_type"]
        if "camera_index" in p:
            st.session_state["camera_index"] = p["camera_index"]
        
        # start video
        source = None
        current_src = st.session_state.get("source_type", "file")
        if current_src == "file":
            source = os.path.join(BASE_DIR, st.session_state.get("video_file", get_video_files()[0]))
        elif current_src == "camera":
            source = st.session_state.get("camera_index", 0)
        elif current_src == "rtsp":
            source = st.session_state.get("rtsp_url", "")
            
        processor.start(source=source, source_type=current_src, light_mode=st.session_state.get("light_mode", False))
    elif name == "stop":
        processor.stop()
    elif name == "reset":
        processor.reset_state()
        st.session_state.map_lat = 28.6765
        st.session_state.map_lon = 77.5005
    elif name == "set_mode":
        mode = p.get("mode")
        if mode in ["normal", "detection", "heatmap"]:
            st.session_state["view_mode"] = mode
    elif name == "toggle_zone":
        if "enabled" in p:
            st.session_state["zone_enabled"] = p["enabled"]
            processor.zone_counting_enabled = p["enabled"]
    elif name == "toggle_light":
        if "enabled" in p:
            st.session_state["light_mode"] = p["enabled"]
    elif name == "set_max_people":
        val = p.get("value")
        if val: 
            st.session_state["max_people"] = val
            processor.max_people = val
    elif name == "set_overcrowd":
        val = p.get("value")
        if val: 
            st.session_state["overcrowd"] = val
            processor.overcrowd_thresh = val

def handle_nlp():
    prompt = st.session_state.nlp_input
    provider = st.session_state.nlp_provider
    model = st.session_state.nlp_model
    if not prompt: return
    res = run_nlp_router(prompt, provider, model, processor)
    st.session_state.nlp_reply = res["reply"]
    if res["action"] and res["action"]["name"] != "noop":
        execute_action(res["action"])
    st.session_state.nlp_input = ""

st.title("Heat Crowd Streamlit Dashboard")

with st.sidebar:
    st.header("Source Controls")
    
    # Use session state default values safely
    src_type = st.selectbox("Source", ["file", "camera", "rtsp"], key="source_type_widget")
    
    if src_type == "file":
        st.selectbox("Video File", get_video_files(), key="video_file_widget")
    elif src_type == "camera":
        st.number_input("Camera Index", value=st.session_state.get("camera_index", 0), min_value=0, key="camera_index_widget")
    elif src_type == "rtsp":
        st.text_input("RTSP URL", key="rtsp_url_widget")

    st.checkbox("Light Mode", value=st.session_state.get("light_mode", False), key="light_mode_widget")
    zone_enabled = st.checkbox("Zone Counting", value=st.session_state.get("zone_enabled", False), key="zone_enabled_widget")
    
    st.number_input("Max People", value=st.session_state.get("max_people", 350), key="max_people_widget")
    st.number_input("Overcrowd %", value=st.session_state.get("overcrowd", 80), key="overcrowd_widget")
    
    # Sync widget values back to session_state and processor
    st.session_state.source_type = st.session_state.source_type_widget
    if src_type == "file":
        st.session_state.video_file = st.session_state.video_file_widget
    elif src_type == "camera":
        st.session_state.camera_index = st.session_state.camera_index_widget
    elif src_type == "rtsp":
        st.session_state.rtsp_url = st.session_state.rtsp_url_widget
        
    st.session_state.light_mode = st.session_state.light_mode_widget
    st.session_state.zone_enabled = st.session_state.zone_enabled_widget
    st.session_state.max_people = st.session_state.max_people_widget
    st.session_state.overcrowd = st.session_state.overcrowd_widget

    processor.max_people = st.session_state.max_people
    processor.overcrowd_thresh = st.session_state.overcrowd
    processor.zone_counting_enabled = st.session_state.zone_enabled

    col1, col2, col3 = st.columns(3)
    if col1.button("Start"):
        execute_action({"name": "start", "params": {}})
    if col2.button("Stop"):
        execute_action({"name": "stop", "params": {}})
    if col3.button("Reset"):
        execute_action({"name": "reset", "params": {}})
        
    if st.button("Clear Zones"):
        processor.set_zones([])
        
    st.markdown("---")
    st.header("Command Bar (NLP)")
    provider = st.selectbox("Provider", ["fallback", "openai", "gemini", "ollama"], key="nlp_provider")
    model_opts = []
    if provider == "gemini":
        model_opts = ["gemini-3.1-pro-preview", "gemini-3-flash", "gemini-2.5-pro", "gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro"]
    elif provider == "openai":
        model_opts = ["gpt-4o-mini", "gpt-4o"]
    elif provider == "ollama":
        model_opts = ["llama3.1:8b", "llama3:8b", "mistral"]
        
    if model_opts:
        st.selectbox("Model", model_opts, key="nlp_model")
    else:
        st.session_state.nlp_model = ""
        
    st.text_input("Command (e.g. 'start webcam', 'reset')", key="nlp_input", on_change=handle_nlp)
    if st.session_state.nlp_reply:
        st.info(st.session_state.nlp_reply)

st.radio("View Mode", ["normal", "detection", "heatmap"], horizontal=True, key="view_mode_widget")
st.session_state.view_mode = st.session_state.view_mode_widget

col_vid, col_map = st.columns([3, 1])

with col_vid:
    frame_placeholder = st.empty()
    if not processor.running and st.session_state.zone_enabled and st_canvas:
        st.write("Draw zones on this sample frame:")
        # Grab a sample frame to draw on
        cap = cv2.VideoCapture(st.session_state.get('camera_index', 0) if src_type == 'camera' else os.path.join(BASE_DIR, st.session_state.get('video_file', '')))
        ret, s_frame = cap.read()
        cap.release()
        if ret:
            s_frame_rgb = cv2.cvtColor(s_frame, cv2.COLOR_BGR2RGB)
            canvas_res = st_canvas(
                fill_color="rgba(30, 74, 120, 0.4)",
                stroke_width=2,
                stroke_color="#4da3ff",
                background_image=Image.fromarray(s_frame_rgb),
                height=s_frame_rgb.shape[0],
                width=s_frame_rgb.shape[1],
                drawing_mode="rect",
                key="canvas",
            )
            if canvas_res.json_data is not None:
                new_zones = []
                for obj in canvas_res.json_data["objects"]:
                    if obj["type"] == "rect":
                        x1 = int(obj["left"])
                        y1 = int(obj["top"])
                        x2 = int(obj["left"] + obj["width"])
                        y2 = int(obj["top"] + obj["height"])
                        new_zones.append([x1, y1, x2, y2])
                processor.set_zones(new_zones)
        else:
            st.warning("Could not read frame to draw zones. Start video first or ensure source is valid.")

    st.markdown("---")
    st.subheader("Command Bar (NLP)")
    c_prov, c_mod = st.columns(2)
    with c_prov:
        provider = st.selectbox("Provider", ["fallback", "openai", "gemini", "ollama"], key="nlp_provider")
    model_opts = []
    if provider == "gemini":
        model_opts = ["gemini-3.1-pro-preview", "gemini-3-flash", "gemini-2.5-pro", "gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro"]
    elif provider == "openai":
        model_opts = ["gpt-4o-mini", "gpt-4o"]
    elif provider == "ollama":
        model_opts = ["llama3.1:8b", "llama3:8b", "mistral"]
        
    with c_mod:
        if model_opts:
            st.selectbox("Model", model_opts, key="nlp_model")
        else:
            st.session_state.nlp_model = ""
            
    st.text_input("Command (e.g. 'start webcam', 'reset')", key="nlp_input", on_change=handle_nlp)
    if st.session_state.nlp_reply:
        st.info(st.session_state.nlp_reply)

with col_map:
    st.subheader("Live Stats")
    c1, c2, c3 = st.columns(3)
    count_ph = c1.empty()
    traffic_ph = c2.empty()
    fps_ph = c3.empty()
    c4, c5, c6 = st.columns(3)
    peak_ph = c4.empty()
    alerts_ph = c5.empty()
    zone_ph = c6.empty()

    chart_ph = st.empty()

    st.subheader("Map")
    m_lat = st.number_input("Latitude", value=st.session_state.map_lat, format="%.6f", key="inp_lat")
    m_lon = st.number_input("Longitude", value=st.session_state.map_lon, format="%.6f", key="inp_lon")
    m_width = st.number_input("Area Width (m)", value=100)
    
    def detect_ip():
        try:
            r = requests.get("http://ip-api.com/json/", timeout=3)
            d = r.json()
            st.session_state.map_lat = d.get("lat", m_lat)
            st.session_state.map_lon = d.get("lon", m_lon)
        except:
            st.warning("Failed to detect location.")
    
    st.button("Detect Live Location", on_click=detect_ip)
    
    st.session_state.map_lat = m_lat
    st.session_state.map_lon = m_lon

    m = folium.Map(location=[st.session_state.map_lat, st.session_state.map_lon], zoom_start=16)
    folium.Marker([st.session_state.map_lat, st.session_state.map_lon], draggable=True).add_to(m)
    folium.Circle(
        location=[st.session_state.map_lat, st.session_state.map_lon],
        radius=m_width/2,
        color="#ff6861",
        fill=True,
        fill_color="#ff6861"
    ).add_to(m)
    
    # We render the map outside the fast loop to prevent heavy UI lag
    st_folium(m, width=400, height=250, returned_objects=[])

if processor.running:
    while processor.running:
        rgb = processor.get_rgb_frame(st.session_state.view_mode)
        if rgb is not None:
            frame_placeholder.image(rgb, channels="RGB", use_container_width=True)
            
        stats = processor.get_stats()
        count_ph.metric("Count", stats.get("count", 0))
        traffic_ph.metric("Crowd", f"{stats.get('traffic', 0):.2f}%")
        fps_ph.metric("FPS", f"{stats.get('fps', 0):.2f}")
        peak_ph.metric("Peak Density", f"{stats.get('peak_density', 0):.2f}%")
        alerts_ph.metric("Alerts", stats.get("alerts_triggered", 0))
        zone_ph.metric("In Zones", stats.get("people_in_zones", 0))
        
        series = stats.get("series", [])
        if series:
            df = pd.DataFrame(series)
            if not df.empty:
                df['Time (s)'] = df['t'] - df['t'].iloc[0]
                chart_ph.line_chart(df.set_index('Time (s)')['count'], height=200)

        time.sleep(0.08)
else:
    # Render zeros when stopped
    count_ph.metric("Count", 0)
    traffic_ph.metric("Crowd", "0.00%")
    fps_ph.metric("FPS", "0.00")
    peak_ph.metric("Peak Density", "0.00%")
    alerts_ph.metric("Alerts", 0)
    zone_ph.metric("In Zones", 0)
    chart_ph.line_chart(pd.DataFrame(columns=['count']), height=200)
    frame_placeholder.info("Stream is stopped. Click Start to begin.")