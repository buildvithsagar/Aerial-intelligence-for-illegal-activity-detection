# requirements: PyQt5, opencv-python, numpy, ultralytics
import sys
import cv2
import numpy as np
import time
import os
import json
import urllib.error
import urllib.request
from collections import deque
from ultralytics import YOLO
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, 
                            QFileDialog, QHBoxLayout, QCheckBox, QGroupBox, QFormLayout, 
                            QLineEdit, QMessageBox, QDialog, QComboBox, QDialogButtonBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QUrl, QPoint, QRect
from PyQt5.QtGui import QImage, QPixmap, QFont, QCursor, QPainter, QPen
from PyQt5.QtWebEngineWidgets import QWebEngineView
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import datetime

PERSON_CLASS_ID = 0

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray, np.ndarray, np.ndarray, int, float, float, bool, float, int)

    def __init__(self, video_path, conf_thresh=0.10, iou_thresh=0.6, max_people=350, overcrowd_thresh=80.0, light_mode=False, zones=None, zone_counting_enabled=False):
        super().__init__()
        self.video_path = video_path
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.max_people = max_people
        self.overcrowd_thresh = overcrowd_thresh
        self._run_flag = True
        self.model = YOLO('weight.pt')
        self.heatmap = None
        self.heatmap_decay = 0.95
        self.peak_density = 0.0
        self.alerts_triggered = 0
        self.light_mode = light_mode
        self.zones = zones or []
        self.zone_counting_enabled = zone_counting_enabled

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        prev_time = time.time()
        frame_skip = 2 if self.light_mode else 1
        model_size = 320 if self.light_mode else 640
        frame_count = 0
        
        while self._run_flag:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart video
                continue
                
            frame_count += 1
            if self.light_mode and (frame_count % frame_skip != 0):
                continue
                
            # Store original frame for normal mode
            original_frame = frame.copy()
            
            if self.heatmap is None:
                self.heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)

            # Resize for light mode
            input_frame = cv2.resize(frame, (model_size, model_size)) if self.light_mode else frame
            results = self.model(input_frame, conf=self.conf_thresh, iou=self.iou_thresh, verbose=False)
            
            # Map boxes back if resized
            scale_x = frame.shape[1] / model_size if self.light_mode else 1.0
            scale_y = frame.shape[0] / model_size if self.light_mode else 1.0

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
            
            # Create detection frame with bounding boxes
            detection_frame = frame.copy()

            # --- Zone-based counting logic ---
            people_in_zones = 0
            all_people_count = int(person_indices.size)
            
            # Update heatmap based on ALL people for a complete thermal view
            frame_h, frame_w = frame.shape[:2]
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
                    in_zone = True  # If zone counting disabled, count all people

                if in_zone:
                    people_in_zones += 1
                    cv2.rectangle(detection_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"Person: {confidences[i]:.2f}"
                    cv2.putText(detection_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            current_people_count = people_in_zones if self.zone_counting_enabled and self.zones else all_people_count
            crowd_percentage = (current_people_count / self.max_people) * 100 if self.max_people > 0 else 0
            
            # Update peak density
            if crowd_percentage > self.peak_density:
                self.peak_density = crowd_percentage
            
            self.heatmap *= self.heatmap_decay
            
            # Create heatmap frame
            heatmap_display = np.clip(self.heatmap, 0, 255)
            heatmap_display = cv2.GaussianBlur(heatmap_display, (0, 0), sigmaX=15, sigmaY=15)
            heatmap_display = cv2.normalize(heatmap_display, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            heatmap_color = cv2.applyColorMap(heatmap_display, cv2.COLORMAP_JET)
            
            overlay_alpha = 0.4
            heatmap_frame = cv2.addWeighted(heatmap_color, overlay_alpha, frame, 1 - overlay_alpha, 0)
            
            # Add count and stats to all frames
            overcrowd = crowd_percentage > self.overcrowd_thresh
            if overcrowd:
                self.alerts_triggered += 1
                
            # Draw zones on frames
            if self.zones:
                for display_frame in [detection_frame, heatmap_frame]:
                    for z in self.zones:
                        cv2.rectangle(display_frame, (z[0], z[1]), (z[2], z[3]), (255, 0, 0), 2)

            # Add stats text to detection and heatmap frames
            for display_frame in [detection_frame, heatmap_frame]:
                if overcrowd:
                    cv2.putText(display_frame, 'OVERCROWDING!', (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                
                text = f'Count: {current_people_count} | Traffic: {crowd_percentage:.2f}%'
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                text_x, text_y = 50, 30
                
                cv2.rectangle(display_frame, (text_x - 10, text_y - text_size[1] - 10),
                            (text_x + text_size[0] + 10, text_y + 10), (255, 255, 255), cv2.FILLED)
                cv2.putText(display_frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            # FPS calculation
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if curr_time != prev_time else 0
            prev_time = curr_time
            
            # Send all three frames to GUI
            self.change_pixmap_signal.emit(original_frame, detection_frame, heatmap_frame, 
                                         current_people_count, crowd_percentage, fps, overcrowd, 
                                         self.peak_density, self.alerts_triggered)
            self.msleep(10)
            
        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

class ClickableLabel(QLabel):
    clicked = pyqtSignal()
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setCursor(QCursor(Qt.PointingHandCursor))
        self.default_style = "border: 2px solid #666666; background-color: #333333;"
        self.hover_style = "border: 2px solid #4CAF50; background-color: #444444;"
        self.setStyleSheet(self.default_style)
    def enterEvent(self, event):
        self.setStyleSheet(self.hover_style)
        super().enterEvent(event)
    def leaveEvent(self, event):
        self.setStyleSheet(self.default_style)
        super().leaveEvent(event)
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)

class DrawableLabel(QLabel):
    new_zone = pyqtSignal(QRect)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_drawing = False
        self._start_point = QPoint()
        self._end_point = QPoint()
        self.setCursor(Qt.CrossCursor)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._is_drawing = True
            self._start_point = event.pos()
            self._end_point = event.pos()
            self.update()

    def mouseMoveEvent(self, event):
        if self._is_drawing:
            self._end_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._is_drawing:
            self._is_drawing = False
            rect = QRect(self._start_point, self._end_point).normalized()
            if rect.width() > 5 and rect.height() > 5:  # Threshold for a valid zone
                self.new_zone.emit(rect)
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._is_drawing:
            painter = QPainter(self)
            pen = QPen(Qt.blue, 2, Qt.DashLine)
            painter.setPen(pen)
            rect = QRect(self._start_point, self._end_point).normalized()
            painter.drawRect(rect)

class CrowdCountingApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Crowd Counting & Density Detection')
        self.setGeometry(100, 100, 1400, 900)
        self.setStyleSheet("""
            QWidget {
                background-color: #111111;
                font-family: Arial, sans-serif;
                color: #f0f0f0;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #333333;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
                background-color: #222222;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #f0f0f0;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #333333;
            }
            QRadioButton {
                font-weight: bold;
                spacing: 5px;
                color: #f0f0f0;
            }
            QLabel {
                color: #f0f0f0;
            }
        """)
        # Logo at the top (now for side-by-side heading)
        self.logo_label = QLabel(self)
        self.logo_label.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        self.logo_label.setFixedHeight(100)
        try:
            logo_pixmap = QPixmap('CIAA.png')
            if not logo_pixmap.isNull():
                self.logo_label.setPixmap(logo_pixmap.scaledToHeight(90, Qt.SmoothTransformation))
        except Exception:
            self.logo_label.setText('')
        # Heading label
        self.heading_label = QLabel('Crowd Detection with Heatmap', self)
        self.heading_label.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        self.heading_label.setFont(QFont('Arial', 22, QFont.Bold))
        self.heading_label.setStyleSheet('color: #f0f0f0; margin-left: 20px;')
        # Main video display
        self.video_label = DrawableLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(800, 600)
        self.video_label.setStyleSheet("border: 2px solid #333333; background-color: black;")
        self.video_label.new_zone.connect(self.add_zone)
        
        # Mode preview thumbnails
        self.normal_thumb = ClickableLabel()
        self.detection_thumb = ClickableLabel()
        self.heatmap_thumb = ClickableLabel()
        
        for thumb in [self.normal_thumb, self.detection_thumb, self.heatmap_thumb]:
            thumb.setFixedSize(200, 150)
            thumb.setAlignment(Qt.AlignCenter)
            # Style is set in ClickableLabel
        self.normal_thumb.clicked.connect(lambda: self.set_mode(0))
        self.detection_thumb.clicked.connect(lambda: self.set_mode(1))
        self.heatmap_thumb.clicked.connect(lambda: self.set_mode(2))
        
        # Control buttons
        self.open_btn = QPushButton('Open Video')
        self.start_btn = QPushButton('Start')
        self.stop_btn = QPushButton('Stop')
        self.stop_btn.setEnabled(False)
        
        self.fps_toggle = QCheckBox('Light Mode (FPS Optimized)')
        self.fps_toggle.setChecked(False)
        self.live_btn = QPushButton('Live Mode')
        self.live_btn.clicked.connect(self.live_mode)
        
        # Stats display
        self.stats_label = QLabel('Stats will appear here')
        self.stats_label.setAlignment(Qt.AlignLeft)
        
        # Live Stats Dashboard
        self.dashboard = QGroupBox('Live Stats Dashboard')
        self.count_label = QLabel('Current Count: 0')
        self.peak_label = QLabel('Peak Density: 0.00%')
        self.alerts_label = QLabel('Alerts Triggered: 0')
        self.zone_count_label = QLabel('People in Zones: 0')

        # --- Time-Series Chart ---
        self.figure = Figure(figsize=(3, 2.5))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title('People Count vs Time')
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Count')
        self.max_points = 120
        self.time_series_data = deque(maxlen=self.max_points)
        self.chart_window_seconds = 10
        self.chart_update_interval = 0.5
        self.last_chart_update = 0.0
        self.thumbnail_stride = 3
        self.thumbnail_frame_counter = 0
        
        # --- Map Settings ---
        self.map_settings_group = QGroupBox('Map Settings')
        self.lat_input = QLineEdit('40.7128')  # Default to NYC
        self.lon_input = QLineEdit('-74.0060')
        self.width_input = QLineEdit('100') # Default width in meters
        self.update_map_btn = QPushButton('Update Map')
        self.detect_location_btn = QPushButton('Detect Live Location')
        self.update_map_btn.clicked.connect(self.update_map_coverage)
        self.detect_location_btn.clicked.connect(self.detect_live_location)
        
        map_settings_layout = QFormLayout()
        map_settings_layout.addRow('Center Latitude:', self.lat_input)
        map_settings_layout.addRow('Center Longitude:', self.lon_input)
        map_settings_layout.addRow('Area Width (m):', self.width_input)
        map_settings_layout.addRow(self.detect_location_btn)
        map_settings_layout.addRow(self.update_map_btn)
        self.map_settings_group.setLayout(map_settings_layout)

        # --- Zone Controls ---
        self.zone_controls_group = QGroupBox('Zone Controls')
        self.enable_zone_checkbox = QCheckBox('Enable Zone Counting')
        self.clear_zones_btn = QPushButton('Clear All Zones')
        self.clear_zones_btn.clicked.connect(self.clear_zones)
        
        zone_layout = QVBoxLayout()
        zone_layout.addWidget(self.enable_zone_checkbox)
        zone_layout.addWidget(self.clear_zones_btn)
        self.zone_controls_group.setLayout(zone_layout)

        self.minimap_view = QWebEngineView()
        self.minimap_view.setFixedSize(500, 350)

        dash_layout = QFormLayout()
        dash_layout.addRow(self.count_label)
        dash_layout.addRow(self.peak_label)
        dash_layout.addRow(self.alerts_label)
        dash_layout.addRow(self.zone_count_label)
        dash_layout.addRow(self.map_settings_group)
        dash_layout.addRow(self.zone_controls_group)
        dash_layout.addRow(QLabel("Live Density Map:"))
        dash_layout.addRow(self.minimap_view)
        dash_layout.addRow(self.canvas)  # Move chart to the bottom
        self.dashboard.setLayout(dash_layout)
        
        # Connect signals
        self.open_btn.clicked.connect(self.open_video)
        self.start_btn.clicked.connect(self.start_video)
        self.stop_btn.clicked.connect(self.stop_video)
        # Remove mode_group.buttonClicked.connect(self.mode_changed)
        
        self.setup_layout()
        
        # Initialize variables
        self.video_path = None
        self.thread = None
        self.current_frames = [None, None, None]  # normal, detection, heatmap
        self.current_mode = 1  # Default to detection mode
        self.zones = []
        # Loading animation/message state
        self.loading_timer = None
        self.loading_shown = False
        self.first_frame_received = False

    def setup_layout(self):
        # Main layout
        main_layout = QHBoxLayout()
        # Left side - video and controls
        left_layout = QVBoxLayout()
        # Top bar with logo and heading
        top_bar = QHBoxLayout()
        top_bar.addWidget(self.logo_label, 0)
        top_bar.addWidget(self.heading_label, 1)
        top_bar.addStretch()
        left_layout.addLayout(top_bar)
        left_layout.addWidget(self.video_label)
        
        # Remove mode selection layout
        # Thumbnail layout
        thumb_layout = QHBoxLayout()
        thumb_layout.setAlignment(Qt.AlignHCenter)
        thumb_layout.setSpacing(30)
        # Add vertical spacer above thumbnails to move them lower
        thumbline_spacer = QVBoxLayout()
        thumbline_spacer.addSpacing(30)  # Adjust value for more/less space
        thumbline_spacer.addLayout(thumb_layout)
        
        # Normal thumbnail with label
        normal_container = QVBoxLayout()
        normal_container.addWidget(self.normal_thumb)
        normal_label = QLabel("Normal")
        normal_label.setAlignment(Qt.AlignCenter)
        normal_label.setFont(QFont("Arial", 12, QFont.Bold))
        normal_container.addWidget(normal_label)
        
        # Detection thumbnail with label
        detection_container = QVBoxLayout()
        detection_container.addWidget(self.detection_thumb)
        detection_label = QLabel("Detection")
        detection_label.setAlignment(Qt.AlignCenter)
        detection_label.setFont(QFont("Arial", 12, QFont.Bold))
        detection_container.addWidget(detection_label)
        
        # Heatmap thumbnail with label
        heatmap_container = QVBoxLayout()
        heatmap_container.addWidget(self.heatmap_thumb)
        heatmap_label = QLabel("Heatmap")
        heatmap_label.setAlignment(Qt.AlignCenter)
        heatmap_label.setFont(QFont("Arial", 12, QFont.Bold))
        heatmap_container.addWidget(heatmap_label)
        
        thumb_layout.addLayout(normal_container)
        thumb_layout.addLayout(detection_container)
        thumb_layout.addLayout(heatmap_container)
        
        # Add thumbnails below the top bar and video
        left_layout.addLayout(thumbline_spacer)
        left_layout.addWidget(self.stats_label)
        # Restore control buttons layout
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.open_btn)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        btn_layout.addWidget(self.fps_toggle)
        btn_layout.addWidget(self.live_btn)
        btn_layout.addStretch()
        left_layout.addLayout(btn_layout)
        
        # Right side - dashboard
        main_layout.addLayout(left_layout, 3)
        main_layout.addWidget(self.dashboard, 1)
        
        self.setLayout(main_layout)

    def set_mode(self, mode_idx):
        self.current_mode = mode_idx
        self.update_main_display()

    def open_video(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open Video File', '', 
                                             'Video Files (*.mp4 *.avi *.mov *.mkv)')
        if fname:
            self.video_path = fname
            self.stats_label.setText(f'Loaded: {fname}')
            self.start_video()  # Auto-start after selecting video

    def live_mode(self):
        self.detect_live_location(silent=True)
        # Use custom dialog for source selection
        dialog = LiveSourceDialog(self)
        choice, ok = dialog.get_choice()
        if not ok or not choice:
            return
        if choice == "Webcam":
            # Try webcam indices 0, 1, 2
            found = False
            for idx in range(3):
                cap = cv2.VideoCapture(idx)
                ret, _ = cap.read()
                cap.release()
                if ret:
                    self.video_path = idx
                    self.stats_label.setText(f'Live: Webcam (index {idx})')
                    self.start_video()
                    found = True
                    break
            if not found:
                QMessageBox.warning(self, "No Webcam Found", "No available webcam was detected.")
        elif choice == "RTSP Stream":
            while True:
                dialog2 = RTSPInputDialog(self)
                url, ok2 = dialog2.get_url()
                if not ok2 or not url:
                    return
                if url.lower().startswith("rtsp://"):
                    self.video_path = url
                    self.stats_label.setText(f'Live: RTSP {url}')
                    self.start_video()
                    break
                else:
                    QMessageBox.warning(self, "Invalid RTSP URL", "Please enter a valid RTSP URL starting with rtsp://")

    def _fetch_location_from_ip(self):
        endpoints = [
            ("https://ipapi.co/json/", lambda d: (d.get('latitude'), d.get('longitude'))),
            ("http://ip-api.com/json/", lambda d: (d.get('lat'), d.get('lon'))),
        ]

        headers = {
            'User-Agent': 'HeatCrowd/1.0'
        }

        for url, parser in endpoints:
            req = urllib.request.Request(url, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=4) as response:
                    if response.status != 200:
                        continue
                    payload = response.read().decode('utf-8', errors='replace')
                    data = json.loads(payload)
                    lat, lon = parser(data)
                    if lat is None or lon is None:
                        continue
                    return float(lat), float(lon)
            except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError):
                continue

        return None

    def detect_live_location(self, silent=False):
        location = self._fetch_location_from_ip()
        if location is None:
            if not silent:
                QMessageBox.warning(self, 'Location Detection Failed', 'Could not detect live location. Please enter latitude and longitude manually.')
            return False

        lat, lon = location
        self.lat_input.setText(f'{lat:.6f}')
        self.lon_input.setText(f'{lon:.6f}')
        self.stats_label.setText(f'Live location detected: {lat:.6f}, {lon:.6f}')

        if not silent:
            self.update_map_coverage()
        return True

    def start_video(self):
        if self.video_path is None or self.video_path == '':
            self.stats_label.setText('Please select a video file or live source first!')
            return

        # --- Leaflet Map Setup ---
        try:
            lat = float(self.lat_input.text())
            lon = float(self.lon_input.text())
            width_meters = float(self.width_input.text())
        except ValueError:
            self.stats_label.setText('Invalid map settings! Please enter numbers.')
            return

        # Simple conversion from meters to degrees (approximation)
        meters_per_degree = 111320 
        width_deg = width_meters / meters_per_degree
        
        # Get video aspect ratio to calculate height
        cap = cv2.VideoCapture(self.video_path)
        vid_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        vid_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        cap.release()
        aspect_ratio = vid_h / vid_w if vid_w > 0 else 1.0
        height_deg = width_deg * aspect_ratio

        half_w = width_deg / 2
        half_h = height_deg / 2
        
        south = lat - half_h
        north = lat + half_h
        west = lon - half_w
        east = lon + half_w
        
        bounds = [[south, west], [north, east]]
        
        map_path = os.path.abspath('leaflet_map.html')
        self.minimap_view.setUrl(QUrl.fromLocalFile(map_path))
        
        try:
            self.minimap_view.loadFinished.disconnect()
        except TypeError:
            pass

        # Use a lambda to avoid issues with loop variables in closures
        self.minimap_view.loadFinished.connect(
            lambda: self.setup_map_and_start_thread(lat, lon, bounds)
        )

    def setup_map_and_start_thread(self, lat, lon, bounds):
        # Show the coverage circle after initializing the map
        self.minimap_view.page().runJavaScript(f"initializeMap({lat}, {lon}, 17);")
        try:
            width_meters = float(self.width_input.text())
        except ValueError:
            width_meters = 100
        self.minimap_view.page().runJavaScript(f"showCoverageCircle({lat}, {lon}, {width_meters});")
        # Disconnect the signal to prevent it from running again on reloads
        try:
            self.minimap_view.loadFinished.disconnect()
        except TypeError:
            pass # Already disconnected
        self.first_frame_received = False
        self.loading_shown = False
        if self.loading_timer:
            self.loading_timer.stop()
        self.loading_timer = QTimer(self)
        self.loading_timer.setSingleShot(True)
        self.loading_timer.timeout.connect(self.show_loading_message)
        self.loading_timer.start(500)  # 0.5 seconds
        light_mode = self.fps_toggle.isChecked()
        zones_enabled = self.enable_zone_checkbox.isChecked()
        self.thread = VideoThread(self.video_path, light_mode=light_mode, zones=self.zones, zone_counting_enabled=zones_enabled)
        self.thread.change_pixmap_signal.connect(self.update_frames)
        self.thread.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.open_btn.setEnabled(False)
        self.fps_toggle.setEnabled(False)

    def show_loading_message(self):
        if not self.first_frame_received:
            self.loading_shown = True
            self.video_label.setText('Video Loading...')
            self.video_label.setAlignment(Qt.AlignCenter)
            self.video_label.setStyleSheet("border: 2px solid #333333; background-color: black; color: #f0f0f0; font-size: 32px;")

    def stop_video(self):
        if self.thread:
            self.thread.stop()
            self.thread = None
        if self.loading_timer:
            self.loading_timer.stop()
            
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.open_btn.setEnabled(True)
        self.fps_toggle.setEnabled(True)
        # Clear map on stop
        self.minimap_view.setUrl(QUrl("about:blank"))

    def closeEvent(self, event):
        if self.thread:
            self.thread.stop()
        event.accept()

    def add_zone(self, rect: QRect):
        # The rect from DrawableLabel is in widget coordinates.
        # It needs to be scaled to the original video frame coordinates.
        if self.current_frames[0] is None:
            return
        frame_h, frame_w, _ = self.current_frames[0].shape
        label_w = self.video_label.width()
        label_h = self.video_label.height()
        pixmap = self.video_label.pixmap()
        if pixmap is None or pixmap.isNull():
            return
        pixmap_w = pixmap.width()
        pixmap_h = pixmap.height()
        offset_x = (label_w - pixmap_w) / 2
        offset_y = (label_h - pixmap_h) / 2
        # Translate from label coordinates to pixmap coordinates
        px1 = rect.left() - offset_x
        py1 = rect.top() - offset_y
        px2 = rect.right() - offset_x
        py2 = rect.bottom() - offset_y
        # Scale from pixmap coordinates to original frame coordinates
        scale_w = frame_w / pixmap_w
        scale_h = frame_h / pixmap_h
        fx1 = int(px1 * scale_w)
        fy1 = int(py1 * scale_h)
        fx2 = int(px2 * scale_w)
        fy2 = int(py2 * scale_h)
        # Clamp to frame dimensions and add if valid
        fx1, fx2 = sorted((max(0, min(fx1, frame_w)), max(0, min(fx2, frame_w))))
        fy1, fy2 = sorted((max(0, min(fy1, frame_h)), max(0, min(fy2, frame_h))))
        if fx2 > fx1 and fy2 > fy1:
            self.zones.append((fx1, fy1, fx2, fy2))
        # Restart video to update zone-based counting
        self.restart_video_with_zones()

    def clear_zones(self):
        self.zones.clear()
        # Restart video to update zone-based counting
        self.restart_video_with_zones()

    def restart_video_with_zones(self):
        # Stop and restart the video thread with the current zones
        if self.thread:
            self.thread.stop()
            self.thread = None
        if self.video_path:
            self.start_video()

    def update_frames(self, normal_frame, detection_frame, heatmap_frame, count, percent, fps, overcrowd, peak_density, alerts_triggered):
        # First frame received, stop loading message
        self.first_frame_received = True
        if self.loading_timer:
            self.loading_timer.stop()
        if self.loading_shown:
            self.video_label.setText("")
            self.loading_shown = False
        # Store all frames
        self.current_frames = [normal_frame, detection_frame, heatmap_frame]
        # Update thumbnails less frequently to reduce GUI overhead
        self.thumbnail_frame_counter += 1
        if self.thumbnail_frame_counter % self.thumbnail_stride == 0:
            self.update_thumbnail(self.normal_thumb, normal_frame)
            self.update_thumbnail(self.detection_thumb, detection_frame)
            self.update_thumbnail(self.heatmap_thumb, heatmap_frame)
        # Update main display based on current mode
        self.update_main_display()
        # Update stats
        alert = ' | OVERCROWDING!' if overcrowd else ''
        self.stats_label.setText(f'Count: {count} | Traffic: {percent:.2f}% | FPS: {fps:.2f}{alert}')
        self.count_label.setText(f'Current Count: {count}')
        self.peak_label.setText(f'Peak Density: {peak_density:.2f}%')
        self.alerts_label.setText(f'Alerts Triggered: {alerts_triggered}')
        # Update zone count label
        if self.enable_zone_checkbox.isChecked() and self.zones:
            self.zone_count_label.setText(f'People in Zones: {count}')
        else:
            self.zone_count_label.setText('People in Zones: -')
        # Update time-series chart
        now = datetime.datetime.now()
        self.time_series_data.append((now, count))
        # Keep only the most recent window for plotting
        cutoff = now - datetime.timedelta(seconds=self.chart_window_seconds)
        while self.time_series_data and self.time_series_data[0][0] < cutoff:
            self.time_series_data.popleft()

        if not self.time_series_data:
            return

        now_ts = time.time()
        if now_ts - self.last_chart_update >= self.chart_update_interval:
            times, counts = zip(*self.time_series_data)
            x_vals = [(t - times[0]).total_seconds() for t in times]
            self.ax.clear()
            self.ax.plot(x_vals, counts, color='tab:blue', linewidth=2)
            self.ax.set_title('People Count vs Time (last 10s)')
            self.ax.set_xlabel('Seconds')
            self.ax.set_ylabel('Count')
            self.figure.tight_layout()
            self.canvas.draw_idle()
            self.last_chart_update = now_ts
        # No minimap overlay update needed

    def update_thumbnail(self, label, cv_img):
        if cv_img is not None:
            rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(qt_img).scaled(
                label.width(), label.height(), Qt.KeepAspectRatio))

    def update_main_display(self):
        if self.current_frames[self.current_mode] is not None:
            cv_img = self.current_frames[self.current_mode].copy()
            # --- Add semi-transparent black overlay for zones if enabled ---
            if self.enable_zone_checkbox.isChecked() and self.zones:
                overlay = cv_img.copy()
                for z in self.zones:
                    x1, y1, x2, y2 = z
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), thickness=-1)
                alpha = 0.35  # Transparency factor
                cv_img = cv2.addWeighted(overlay, alpha, cv_img, 1 - alpha, 0)
            rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qt_img).scaled(
                self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio))

    def update_map_coverage(self):
        try:
            lat = float(self.lat_input.text())
            lon = float(self.lon_input.text())
            width_meters = float(self.width_input.text())
        except ValueError:
            return
        self.minimap_view.page().runJavaScript(f"initializeMap({lat}, {lon}, 17);")
        self.minimap_view.page().runJavaScript(f"showCoverageCircle({lat}, {lon}, {width_meters});")

class LiveSourceDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Live Mode")
        self.setMinimumWidth(350)
        self.setMinimumHeight(180)
        layout = QVBoxLayout(self)
        label = QLabel("Select source:")
        label.setStyleSheet("font-size: 16px; margin-bottom: 10px;")
        layout.addWidget(label)
        self.combo = QComboBox()
        self.combo.addItems(["Webcam", "RTSP Stream"])
        self.combo.setStyleSheet("font-size: 15px; padding: 4px;")
        layout.addWidget(self.combo)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
    def get_choice(self):
        if self.exec_() == QDialog.Accepted:
            return self.combo.currentText(), True
        return None, False

class RTSPInputDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("RTSP Stream")
        self.setMinimumWidth(420)
        self.setMinimumHeight(180)
        layout = QVBoxLayout(self)
        label = QLabel("Enter RTSP URL:")
        label.setStyleSheet("font-size: 16px; margin-bottom: 10px;")
        layout.addWidget(label)
        self.input = QLineEdit()
        self.input.setPlaceholderText("rtsp://...")
        self.input.setStyleSheet("font-size: 15px; padding: 4px;")
        layout.addWidget(self.input)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
    def get_url(self):
        if self.exec_() == QDialog.Accepted:
            return self.input.text().strip(), True
        return None, False

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = CrowdCountingApp()
    win.show()
    sys.exit(app.exec_())
