import sys
import cv2
import torch
import numpy as np
from PIL import Image
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, 
                            QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, 
                            QGroupBox, QComboBox, QSlider, QGridLayout, QSplitter, QTextEdit, QCheckBox)
from PyQt5.QtCore import Qt, QTimer, pyqtSlot
from PyQt5.QtGui import QPixmap, QImage, QFont, QIcon
from ultralytics import YOLO
import torchvision.transforms as transforms
import torchvision.models as models
from torch import nn
import os
import time
from bytetrack import ByteTrack, Detection

class InsectDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Setup UI
        self.setWindowTitle("Smart Insect Monitoring: Real-Time Detection, Classification, and Risk Assessment")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet("background-color: #f0f0f0;")
        
        # Set window icon if file exists
        icon_path = "logo.png"
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        
        # Initialize variables
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.is_webcam_active = False
        self.is_video_active = False
        self.current_video_path = None
        self.process_every = 3
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.fps_value = 0.0
        self.detection_results = []
        
        # ByteTrack
        self.use_tracking = True
        self.tracker = ByteTrack(high_threshold=0.3, low_threshold=0.01, max_time_lost=60)
        self.tracking_results = []
        
        # Track colors for visualization
        self.track_colors = {}
        
        # Load models
        self.load_models()
        
        # Create UI components
        self.create_ui()
        
        # Set initial status information
        self.tracking_status_label.setText("Detection: waiting | Tracking: waiting")
        self.fps_label.setText("FPS: 0.0")
        
    def load_models(self):
        # GPU checking
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")
        
        # Load YOLOv8 model for detection
        self.detection_model = YOLO('../weights/yolo/best.pt')
        
        # Load the VGG16 classification model
        self.NUM_CLASSES = 6
        self.classification_model, self.class_names = self.load_vgg_model(self.NUM_CLASSES, '../weights/vgg16/best_vgg_insect_model.pth')
        
        # Preprocessing for VGG16
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # For tracking detection results
        self.last_results = None
        self.last_boxes = []
        self.last_labels = []
        
    def load_vgg_model(self, num_classes, model_path):
        # Load model file
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create model structure
        model = models.vgg16(pretrained=False)
        model.classifier[6] = nn.Linear(4096, num_classes)
        
        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        return model, checkpoint['class_names']
    
    def create_ui(self):
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        
        # Create splitter for resizable areas
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel (video display)
        self.left_panel = QWidget()
        left_layout = QVBoxLayout()
        
        # Video display
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #000000; border-radius: 8px;")
        self.video_label.setMinimumSize(640, 480)
        
        # Status bar
        self.status_label = QLabel("Status: Waiting")
        self.status_label.setStyleSheet("color: #606060; font-size: 14px; margin-top: 5px;")
        
        # Add to left layout
        left_layout.addWidget(self.video_label, 1)
        left_layout.addWidget(self.status_label)
        self.left_panel.setLayout(left_layout)
        
        # Right panel (controls and results)
        self.right_panel = QWidget()
        right_layout = QVBoxLayout()
        
        # Labels for FPS and Status information
        self.fps_label = QLabel("FPS: 0.0")
        self.fps_label.setStyleSheet("color: #0055aa; font-weight: bold; font-size: 16px;")
        self.tracking_status_label = QLabel("Detection: waiting | Tracking: waiting")
        self.tracking_status_label.setStyleSheet("color: #0055aa; font-size: 14px;")
        
        # Box containing FPS and Status information
        stats_group = QGroupBox("Statistics")
        stats_group.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #cccccc; border-radius: 6px; margin-top: 12px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }")
        stats_layout = QVBoxLayout()
        stats_layout.addWidget(self.fps_label)
        stats_layout.addWidget(self.tracking_status_label)
        stats_group.setLayout(stats_layout)
        
        # Controls group
        controls_group = QGroupBox("Controls")
        controls_group.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #cccccc; border-radius: 6px; margin-top: 12px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }")
        controls_layout = QVBoxLayout()
        
        # Buttons layout
        buttons_layout = QHBoxLayout()
        
        # Webcam button
        self.webcam_button = QPushButton("Webcam")
        self.webcam_button.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; border-radius: 5px; padding: 10px; font-weight: bold; } QPushButton:hover { background-color: #45a049; }")
        self.webcam_button.clicked.connect(self.toggle_webcam)
        
        # Video button
        self.video_button = QPushButton("Open Video")
        self.video_button.setStyleSheet("QPushButton { background-color: #2196F3; color: white; border-radius: 5px; padding: 10px; font-weight: bold; } QPushButton:hover { background-color: #0b7dda; }")
        self.video_button.clicked.connect(self.open_video)
        
        # Image classification button
        self.image_button = QPushButton("Classify Image")
        self.image_button.setStyleSheet("QPushButton { background-color: #9c27b0; color: white; border-radius: 5px; padding: 10px; font-weight: bold; } QPushButton:hover { background-color: #7b1fa2; }")
        self.image_button.clicked.connect(self.classify_image)
        
        # Stop button
        self.stop_button = QPushButton("Stop")
        self.stop_button.setStyleSheet("QPushButton { background-color: #f44336; color: white; border-radius: 5px; padding: 10px; font-weight: bold; } QPushButton:hover { background-color: #da190b; }")
        self.stop_button.clicked.connect(self.stop_video)
        self.stop_button.setEnabled(False)
        
        # Add buttons to layout
        buttons_layout.addWidget(self.webcam_button)
        buttons_layout.addWidget(self.video_button)
        buttons_layout.addWidget(self.image_button)
        buttons_layout.addWidget(self.stop_button)
        
        # Settings layout
        settings_layout = QGridLayout()
        
        # Process every frames setting
        settings_layout.addWidget(QLabel("Detection Frequency:"), 0, 0)
        self.process_slider = QSlider(Qt.Horizontal)
        self.process_slider.setMinimum(1)
        self.process_slider.setMaximum(10)
        self.process_slider.setValue(self.process_every)
        self.process_slider.setTickPosition(QSlider.TicksBelow)
        self.process_slider.setTickInterval(1)
        self.process_slider.valueChanged.connect(self.update_process_every)
        settings_layout.addWidget(self.process_slider, 0, 1)
        self.process_value_label = QLabel(f"{self.process_every}")
        settings_layout.addWidget(self.process_value_label, 0, 2)
        
        # Confidence threshold
        settings_layout.addWidget(QLabel("Confidence Threshold:"), 1, 0)
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setMinimum(10)
        self.confidence_slider.setMaximum(95)
        self.confidence_slider.setValue(50)
        self.confidence_slider.setTickPosition(QSlider.TicksBelow)
        self.confidence_slider.setTickInterval(10)
        settings_layout.addWidget(self.confidence_slider, 1, 1)
        self.confidence_value_label = QLabel("0.50")
        self.confidence_slider.valueChanged.connect(self.update_confidence_label)
        settings_layout.addWidget(self.confidence_value_label, 1, 2)
        
        # Tracking checkbox
        settings_layout.addWidget(QLabel("Object Tracking:"), 2, 0)
        self.tracking_checkbox = QCheckBox("Active")
        self.tracking_checkbox.setChecked(self.use_tracking)
        self.tracking_checkbox.stateChanged.connect(self.toggle_tracking)
        settings_layout.addWidget(self.tracking_checkbox, 2, 1)
        
        # Add layouts to controls group
        controls_layout.addLayout(buttons_layout)
        controls_layout.addLayout(settings_layout)
        controls_group.setLayout(controls_layout)
        
        # Results group
        results_group = QGroupBox("Detection Results")
        results_group.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #cccccc; border-radius: 6px; margin-top: 12px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }")
        results_layout = QVBoxLayout()
        
        # Results list - using QTextEdit instead of QLabel for scrolling
        self.results_label = QTextEdit()
        self.results_label.setReadOnly(True)
        self.results_label.setStyleSheet("padding: 10px; background-color: white; border-radius: 5px;")
        self.results_label.setMinimumHeight(300)
        self.results_label.setText("No detection results yet")
        
        results_layout.addWidget(self.results_label)
        results_group.setLayout(results_layout)
        
        # Risk Assessment group
        risk_group = QGroupBox("Risk Assessment")
        risk_group.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #cccccc; border-radius: 6px; margin-top: 12px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }")
        risk_layout = QVBoxLayout()
        
        # Risk assessment display
        self.risk_label = QTextEdit()
        self.risk_label.setReadOnly(True)
        self.risk_label.setStyleSheet("padding: 10px; background-color: white; border-radius: 5px;")
        self.risk_label.setMinimumHeight(150)
        self.risk_label.setText("No risk assessment available")
        
        risk_layout.addWidget(self.risk_label)
        risk_group.setLayout(risk_layout)
        
        # Add groups to right layout
        right_layout.addWidget(stats_group)  # FPS ve durum bilgilerini ekledik
        right_layout.addWidget(controls_group)
        right_layout.addWidget(results_group)
        right_layout.addWidget(risk_group)  # Risk değerlendirme grubunu ekledik
        right_layout.addStretch()
        self.right_panel.setLayout(right_layout)
        
        # Add panels to splitter
        splitter.addWidget(self.left_panel)
        splitter.addWidget(self.right_panel)
        splitter.setSizes([700, 500])  # Initial sizes
        
        # Add splitter to main layout
        main_layout.addWidget(splitter)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
    
    def update_process_every(self, value):
        self.process_every = value
        self.process_value_label.setText(f"{value}")
        
    def update_confidence_label(self, value):
        self.confidence_value_label.setText(f"{value/100:.2f}")
        # When confidence value changes, update the tracker
        if self.use_tracking:
            # Use fixed thresholds for tracking
            self.tracker = ByteTrack(
                high_threshold=0.3, 
                low_threshold=0.01,
                max_time_lost=60
            )
            self.track_colors = {}
        
    def toggle_tracking(self, state):
        self.use_tracking = state == Qt.Checked
        if self.use_tracking:
            # Reset tracker when enabling
            self.tracker = ByteTrack(
                high_threshold=0.3, 
                low_threshold=0.01,
                max_time_lost=60
            )
            self.track_colors = {}
            # Reset all IDs
            from bytetrack import STrack
            STrack.reset_id()
        else:
            # Clear tracking results
            self.tracking_results = []
        
    def toggle_webcam(self):
        if not self.is_webcam_active:
            # Stop video if running
            if self.is_video_active:
                self.stop_video()
            
            # Open webcam
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.is_webcam_active = True
                self.webcam_button.setText("Stop Webcam")
                self.webcam_button.setStyleSheet("QPushButton { background-color: #ff9800; color: white; border-radius: 5px; padding: 10px; font-weight: bold; } QPushButton:hover { background-color: #e68a00; }")
                self.video_button.setEnabled(False)
                self.stop_button.setEnabled(True)
                self.timer.start(30)  # Update every 30ms (approx. 33 fps)
                self.status_label.setText("Status: Webcam Active")
                
                # Reset frame counter and tracker
                self.frame_count = 0
                if self.use_tracking:
                    self.tracker = ByteTrack(
                        high_threshold=0.3, 
                        low_threshold=0.01,
                        max_time_lost=60
                    )
                    self.track_colors = {}
                    # Reset all IDs
                    from bytetrack import STrack
                    STrack.reset_id()
            else:
                self.status_label.setText("Status: Webcam not opened!")
        else:
            self.stop_video()
    
    def open_video(self):
        # Get video file path
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Video File", "", "Video Files (*.mp4 *.avi *.mov *.wmv)")
        
        if file_path:
            # Stop webcam/video if running
            if self.is_webcam_active or self.is_video_active:
                self.stop_video()
            
            # Open video
            self.cap = cv2.VideoCapture(file_path)
            if self.cap.isOpened():
                self.is_video_active = True
                self.current_video_path = file_path
                self.video_button.setText("Change Video")
                self.webcam_button.setEnabled(False)
                self.stop_button.setEnabled(True)
                self.timer.start(30)  # Update every 30ms (approx. 33 fps)
                self.status_label.setText(f"Status: Playing Video - {os.path.basename(file_path)}")
                
                # Reset frame counter and tracker
                self.frame_count = 0
                if self.use_tracking:
                    self.tracker = ByteTrack(
                        high_threshold=0.3, 
                        low_threshold=0.01,
                        max_time_lost=60
                    )
                    self.track_colors = {}
                    # Reset all IDs
                    from bytetrack import STrack
                    STrack.reset_id()
            else:
                self.status_label.setText("Status: Video not opened!")
    
    def stop_video(self):
        # Stop timer
        self.timer.stop()
        
        # Release video capture
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        # Reset UI
        self.is_webcam_active = False
        self.is_video_active = False
        self.webcam_button.setText("Webcam")
        self.webcam_button.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; border-radius: 5px; padding: 10px; font-weight: bold; } QPushButton:hover { background-color: #45a049; }")
        self.webcam_button.setEnabled(True)
        self.video_button.setText("Open Video")
        self.video_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        
        # Clear video display
        self.video_label.clear()
        self.video_label.setText("Video or Webcam Waiting")
        self.video_label.setStyleSheet("background-color: #000000; color: white; border-radius: 8px;")
        
        # Update status
        self.status_label.setText("Status: Waiting")
        
        # Clear results
        self.results_label.setHtml("<html><body style='font-family: Arial; font-size: 14px;'><p>No detection results yet</p></body></html>")
        self.risk_label.setHtml("<html><body style='font-family: Arial; font-size: 14px;'><p>No risk assessment available</p></body></html>")
        self.detection_results = []
        self.tracking_results = []
    
    def classify_crop(self, crop_img):
        """Classify cropped image using VGG16 model"""
        img = Image.fromarray(crop_img)
        img_tensor = self.transform(img).unsqueeze(0)
        
        with torch.no_grad():
            img_tensor = img_tensor.to(self.device)
            outputs = self.classification_model(img_tensor)
            _, predicted = torch.max(outputs, 1)
            
        return self.class_names[predicted.item()]
    
    def get_track_color(self, track_id):
        """Get a persistent color for a track ID"""
        if track_id not in self.track_colors:
            # Generate a new random color for this track
            r = np.random.randint(0, 255)
            g = np.random.randint(0, 255)
            b = np.random.randint(0, 255)
            self.track_colors[track_id] = (r, g, b)
        return self.track_colors[track_id]
    
    def process_frame(self, frame):
        """Process a single frame: detect, classify, and track insects"""
        frame_copy = frame.copy()
        
        # Process every N frames for detection
        if self.frame_count % self.process_every == 0 or self.last_results is None:
            # Detection
            detection_start_time = time.time()
            results = self.detection_model(frame)
            detection_time = (time.time() - detection_start_time) * 1000  # ms
            self.last_results = results
            
            # Store boxes and labels
            self.last_boxes = []
            self.last_labels = []
            self.detection_results = []
            
            # For summary stats
            class_counts = {}
            
            # Risk assessment data
            risk_data = {}
            
            # Process each detection
            confidence_threshold = self.confidence_slider.value() / 100
            detections = []  # For tracker
            
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Get confidence
                    confidence = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    
                    # Ensure coordinates are within frame bounds
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                    
                    # Add to detections for tracking - add ALL detections
                    detections.append(Detection([x1, y1, x2, y2], confidence))
                    
                    # Skip low confidence detections for detailed display
                    if confidence < confidence_threshold:
                        continue
                    
                    # Crop detected object
                    crop = frame[y1:y2, x1:x2]
                    
                    if crop.size == 0:
                        continue
                        
                    # Classify cropped image
                    class_name = self.classify_crop(crop)
                    
                    # Store box information
                    self.last_boxes.append((x1, y1, x2, y2))
                    self.last_labels.append((cls, confidence, class_name))
                    
                    # Add to detection results with position info
                    detection_info = {
                        "type": self.detection_model.names[cls],
                        "confidence": confidence,
                        "class": class_name,
                        "position": f"X:{(x1+x2)//2}, Y:{(y1+y2)//2}",
                        "size": f"{x2-x1}x{y2-y1}",
                        "area": (x2-x1) * (y2-y1),
                        "bbox": [x1, y1, x2, y2]
                    }
                    self.detection_results.append(detection_info)
                    
                    # Update class counts for summary
                    if class_name in class_counts:
                        class_counts[class_name] += 1
                    else:
                        class_counts[class_name] = 1
                        
                    # Get risk assessment info
                    if class_name not in risk_data:
                        risk_level, risk_description = self.get_risk_assessment(class_name)
                        risk_data[class_name] = {
                            "level": risk_level,
                            "description": risk_description,
                            "count": 1
                        }
                    else:
                        risk_data[class_name]["count"] += 1
            
            # Update tracking if enabled - VERY IMPORTANT: Here we send all detected insects to tracking
            if self.use_tracking:
                tracking_start_time = time.time()
                
                # Send detected insects to tracking if any
                if len(detections) > 0:
                    self.tracking_results = self.tracker.update(detections)
                    
                tracking_time = (time.time() - tracking_start_time) * 1000  # ms
                
                # Store class info for each track ID
                for det in self.detection_results:
                    pos_x, pos_y = map(int, det["position"].replace("X:", "").replace("Y:", "").split(", "))
                    # Find matching track for this detection
                    for track in self.tracking_results:
                        tx1, ty1, tx2, ty2 = track.tlbr.astype(int)
                        # If detection center is inside the track box or significant overlap
                        if tx1 <= pos_x <= tx2 and ty1 <= pos_y <= ty2:
                            # Calculate IoU between detection and track
                            iou = self.calculate_iou(det["bbox"], [tx1, ty1, tx2, ty2])
                            if iou > 0.25:  # Daha düşük bir IoU değeri
                                # Store class in track properties (if not exists)
                                if not hasattr(track, 'insect_class') or track.insect_class is None:
                                    track.insect_class = det["class"]
                                    track.insect_type = det["type"]
                                    track.confidence = det["confidence"]
                                elif track.confidence < det["confidence"]:
                                    # Update with higher confidence detection
                                    track.insect_class = det["class"]
                                    track.insect_type = det["type"]
                                    track.confidence = det["confidence"]
                                break
            
            # Update results display
            if self.detection_results:
                # Sort by detection confidence
                sorted_results = sorted(self.detection_results, key=lambda x: x["confidence"], reverse=True)
                
                # Create summary text
                total_insects = len(sorted_results)
                active_tracks = len(self.tracking_results) if self.tracking_results else 0
                
                summary_text = f"<p><b>Total {total_insects} insects detected</b></p>"
                
                # Add class summary 
                if class_counts:
                    class_summary = ", ".join([f"{count} {cls}" for cls, count in class_counts.items()])
                    summary_text += f"<p>{class_summary}</p>"
                
                # Add tracking info if enabled
                if self.use_tracking:
                    summary_text += f"<p>Active tracked insect count: {active_tracks}</p>"
                
                # Add detection time
                summary_text += f"<p>Detection time: {detection_time:.1f} ms</p>"
                
                # Create detailed results
                results_text = ""
                for i, result in enumerate(sorted_results):
                    cls_type = result["type"]
                    confidence = result["confidence"]
                    cls_name = result["class"]
                    position = result["position"]
                    size = result["size"]
                    
                    results_text += f"""
                    <div style='margin-bottom: 10px; padding: 5px; background-color: #f8f8f8; border-radius: 5px;'>
                        <b>{i+1}. {cls_type} - {cls_name}</b><br>
                        Confidence: <span style='color: {"green" if confidence > 0.7 else "orange" if confidence > 0.5 else "red"};'>{confidence:.2f}</span><br>
                        Position: {position}<br>
                        Size: {size}
                    </div>
                    """
                
                self.results_label.setHtml(f"""
                <html>
                <body style='font-family: Arial; font-size: 14px;'>
                {summary_text}
                <hr>
                <p><b>Detailed detection results:</b></p>
                {results_text}
                </body>
                </html>
                """)
                
                # Update risk assessment display
                if risk_data:
                    risk_html = "<p><b>Risk Assessment Summary:</b></p>"
                    
                    for class_name, data in risk_data.items():
                        level = data["level"]
                        level_color = "green" if level == "Low" else "orange" if level == "Medium" else "red"
                        
                        risk_html += f"""
                        <div style='margin-bottom: 15px; padding: 10px; background-color: #f8f8f8; border-radius: 5px; border-left: 5px solid {level_color};'>
                            <b>{class_name} ({data["count"]})</b><br>
                            Risk Level: <span style='color: {level_color}; font-weight: bold;'>{level}</span><br>
                            <p>{data["description"]}</p>
                        </div>
                        """
                    
                    self.risk_label.setHtml(f"""
                    <html>
                    <body style='font-family: Arial; font-size: 14px;'>
                    {risk_html}
                    </body>
                    </html>
                    """)
                else:
                    self.risk_label.setHtml("<html><body style='font-family: Arial; font-size: 14px;'><p>No risk assessment available</p></body></html>")
            else:
                active_tracks = len(self.tracking_results) if self.tracking_results else 0
                if self.use_tracking and active_tracks > 0:
                    self.results_label.setHtml(f"<html><body style='font-family: Arial; font-size: 14px;'><p>No insects detected, but {active_tracks} insects being tracked.</p></body></html>")
                    
                    # Also update risk assessment for tracked insects
                    risk_data = {}
                    for track in self.tracking_results:
                        if hasattr(track, 'insect_class') and track.insect_class is not None:
                            class_name = track.insect_class
                            if class_name not in risk_data:
                                risk_level, risk_description = self.get_risk_assessment(class_name)
                                risk_data[class_name] = {
                                    "level": risk_level,
                                    "description": risk_description,
                                    "count": 1
                                }
                            else:
                                risk_data[class_name]["count"] += 1
                    
                    if risk_data:
                        risk_html = "<p><b>Risk Assessment for Tracked Insects:</b></p>"
                        
                        for class_name, data in risk_data.items():
                            level = data["level"]
                            level_color = "green" if level == "Low" else "orange" if level == "Medium" else "red"
                            
                            risk_html += f"""
                            <div style='margin-bottom: 15px; padding: 10px; background-color: #f8f8f8; border-radius: 5px; border-left: 5px solid {level_color};'>
                                <b>{class_name} ({data["count"]})</b><br>
                                Risk Level: <span style='color: {level_color}; font-weight: bold;'>{level}</span><br>
                                <p>{data["description"]}</p>
                            </div>
                            """
                        
                        self.risk_label.setHtml(f"""
                        <html>
                        <body style='font-family: Arial; font-size: 14px;'>
                        {risk_html}
                        </body>
                        </html>
                        """)
                    else:
                        self.risk_label.setHtml("<html><body style='font-family: Arial; font-size: 14px;'><p>No risk assessment available for tracked insects</p></body></html>")
                else:
                    self.results_label.setHtml("<html><body style='font-family: Arial; font-size: 14px;'><p>No insects detected</p></body></html>")
                    self.risk_label.setHtml("<html><body style='font-family: Arial; font-size: 14px;'><p>No risk assessment available</p></body></html>")
        
        # If tracking is enabled, ALWAYS draw tracked objects
        active_tracks = 0
        if self.use_tracking and self.tracking_results:
            for track in self.tracking_results:
                active_tracks += 1
                # Get track color
                color = self.get_track_color(track.track_id)
                
                # Get box coordinates
                x1, y1, x2, y2 = track.tlbr.astype(int)
                
                # Draw bounding box
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
                
                # Draw track ID and class info if available
                if hasattr(track, 'insect_class') and track.insect_class is not None:
                    track_text = f"ID: {track.track_id} - {track.insect_class}"
                else:
                    track_text = f"ID: {track.track_id}"
                
                cv2.putText(frame_copy, track_text, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw track history
                if len(track.history) > 1:
                    for i in range(1, len(track.history)):
                        # Get current and previous points
                        pt1 = (int(track.history[i-1][0] + (track.history[i-1][2] - track.history[i-1][0])/2), 
                            int(track.history[i-1][1] + (track.history[i-1][3] - track.history[i-1][1])/2))
                        pt2 = (int(track.history[i][0] + (track.history[i][2] - track.history[i][0])/2), 
                            int(track.history[i][1] + (track.history[i][3] - track.history[i][1])/2))
                        
                        # Draw line between points
                        cv2.line(frame_copy, pt1, pt2, color, 1)
        else:
            # Draw detection boxes if not tracking
            for i, (x1, y1, x2, y2) in enumerate(self.last_boxes):
                cls, confidence, class_name = self.last_labels[i]
                
                # Draw bounding box
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Show detection confidence and classification result
                label = f"{self.detection_model.names[cls]} {confidence:.2f} - {class_name}"
                cv2.putText(frame_copy, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Calculate and show FPS
        if self.frame_count % 10 == 0:
            current_time = time.time()
            elapsed = current_time - self.fps_start_time
            if elapsed > 0:
                self.fps_value = 10 / elapsed
                self.fps_start_time = current_time
                
                # UI'da FPS değerini güncelle
                self.fps_label.setText(f"FPS: {self.fps_value:.1f}")
                
        # Show detection and tracking status in UI
        if self.use_tracking:
            status_text = f"Detection: {'ACTIVE' if self.frame_count % self.process_every == 0 else 'waiting'} | Tracking: ACTIVE ({active_tracks})"
        else:
            status_text = f"Detection: {'ACTIVE' if self.frame_count % self.process_every == 0 else 'waiting'} | Tracking: waiting"
            
        # Update tracking status in UI
        self.tracking_status_label.setText(status_text)
        
        return frame_copy
    
    def calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes."""
        # Get intersection coordinates
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        # Calculate areas
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)
        intersection = w * h
        
        # Calculate union
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection
        
        # Return IoU
        return intersection / max(union, 1e-6)
    
    def update_frame(self):
        # Read frame
        ret, frame = self.cap.read()
        
        if not ret:
            # Video ended or frame reading failed
            if self.is_video_active:
                # Try to restart the video
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if not ret:
                    self.stop_video()
                    return
            else:
                self.stop_video()
                return
        
        # Process frame
        self.frame_count += 1
        processed_frame = self.process_frame(frame)
        
        # Convert to QImage and display
        h, w, ch = processed_frame.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(processed_frame.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        p = QPixmap.fromImage(convert_to_Qt_format)
        
        # Scale pixmap while maintaining aspect ratio
        self.video_label.setPixmap(p.scaled(self.video_label.width(), self.video_label.height(), 
                                           Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def get_risk_assessment(self, class_name):
        """Get risk level and description for an insect class"""
        if class_name == "spider":
            risk_level = "Medium"
            risk_description = "Some spiders can be venomous and may pose a moderate threat, especially in living areas."
            
        elif class_name == "lepi":
            risk_level = "Low"
            risk_description = "Butterflies are generally harmless and pose no significant risk to humans or environments."
            
        elif class_name == "grasshopper":
            risk_level = "Medium"
            risk_description = "Grasshoppers can damage crops and plants, posing a moderate agricultural risk."
            
        elif class_name == "fly":
            risk_level = "Medium"
            risk_description = "Flies can carry bacteria and contaminate food, posing a moderate hygiene risk."
            
        elif class_name == "bee":
            risk_level = "High"
            risk_description = "Bees can sting and cause allergic reactions, especially dangerous for sensitive individuals."
            
        elif class_name == "scorpion":
            risk_level = "High"
            risk_description = "Scorpions are venomous and potentially dangerous, especially in indoor or populated areas."
        
        else:
            risk_level = "Unknown"
            risk_description = "Risk assessment not available for this insect type."
            
        return risk_level, risk_description

    def classify_image(self):
        """Classify insects in a single image using both detection and classification models"""
        # Stop any running video/webcam
        if self.is_webcam_active or self.is_video_active:
            self.stop_video()
        
        # Open file dialog to select an image
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Image Files (*.jpg *.jpeg *.png *.bmp)")
        
        if file_path:
            try:
                # Load image
                image = cv2.imread(file_path)
                if image is None:
                    self.status_label.setText("Status: Failed to load image!")
                    return
                
                # Convert from BGR to RGB for display and processing
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_copy = image_rgb.copy()
                
                # Use YOLO to detect insects in the image
                confidence_threshold = self.confidence_slider.value() / 100
                results = self.detection_model(image)
                
                # Reset detection results
                self.detection_results = []
                
                # Class counts for summary
                class_counts = {}
                
                # Risk assessment data
                risk_data = {}
                
                # Process each detection
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        # Get confidence
                        confidence = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        
                        # Skip low confidence detections
                        if confidence < confidence_threshold:
                            continue
                        
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        
                        # Ensure coordinates are within frame bounds
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
                        
                        # Crop detected object
                        crop = image_rgb[y1:y2, x1:x2]
                        
                        if crop.size == 0:
                            continue
                            
                        # Classify cropped image
                        class_name = self.classify_crop(crop)
                        
                        # Add to detection results with position info
                        detection_info = {
                            "type": self.detection_model.names[cls],
                            "confidence": confidence,
                            "class": class_name,
                            "position": f"X:{(x1+x2)//2}, Y:{(y1+y2)//2}",
                            "size": f"{x2-x1}x{y2-y1}",
                            "area": (x2-x1) * (y2-y1),
                            "bbox": [x1, y1, x2, y2]
                        }
                        self.detection_results.append(detection_info)
                        
                        # Update class counts for summary
                        if class_name in class_counts:
                            class_counts[class_name] += 1
                        else:
                            class_counts[class_name] = 1
                            
                        # Get risk assessment info
                        if class_name not in risk_data:
                            risk_level, risk_description = self.get_risk_assessment(class_name)
                            risk_data[class_name] = {
                                "level": risk_level,
                                "description": risk_description,
                                "count": 1
                            }
                        else:
                            risk_data[class_name]["count"] += 1
                        
                        # Draw bounding box on the image
                        cv2.rectangle(image_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Show detection confidence and classification result
                        label = f"{self.detection_model.names[cls]} {confidence:.2f} - {class_name}"
                        cv2.putText(image_copy, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Display processed image with bounding boxes
                h, w, ch = image_copy.shape
                bytes_per_line = ch * w
                convert_to_Qt_format = QImage(image_copy.data, w, h, bytes_per_line, QImage.Format_RGB888)
                p = QPixmap.fromImage(convert_to_Qt_format)
                
                # Scale pixmap while maintaining aspect ratio
                self.video_label.setPixmap(p.scaled(self.video_label.width(), self.video_label.height(), 
                                           Qt.KeepAspectRatio, Qt.SmoothTransformation))
                
                # Display results
                self.status_label.setText(f"Status: Image analyzed - {os.path.basename(file_path)}")
                
                # Update detection results display
                if self.detection_results:
                    # Sort by detection confidence
                    sorted_results = sorted(self.detection_results, key=lambda x: x["confidence"], reverse=True)
                    
                    # Create summary text
                    total_insects = len(sorted_results)
                    
                    summary_text = f"<p><b>Total {total_insects} insects detected</b></p>"
                    
                    # Add class summary 
                    if class_counts:
                        class_summary = ", ".join([f"{count} {cls}" for cls, count in class_counts.items()])
                        summary_text += f"<p>{class_summary}</p>"
                    
                    # Create detailed results
                    results_text = ""
                    for i, result in enumerate(sorted_results):
                        cls_type = result["type"]
                        confidence = result["confidence"]
                        cls_name = result["class"]
                        position = result["position"]
                        size = result["size"]
                        
                        results_text += f"""
                        <div style='margin-bottom: 10px; padding: 5px; background-color: #f8f8f8; border-radius: 5px;'>
                            <b>{i+1}. {cls_type} - {cls_name}</b><br>
                            Confidence: <span style='color: {"green" if confidence > 0.7 else "orange" if confidence > 0.5 else "red"};'>{confidence:.2f}</span><br>
                            Position: {position}<br>
                            Size: {size}
                        </div>
                        """
                    
                    self.results_label.setHtml(f"""
                    <html>
                    <body style='font-family: Arial; font-size: 14px;'>
                    {summary_text}
                    <hr>
                    <p><b>Detailed detection results:</b></p>
                    {results_text}
                    </body>
                    </html>
                    """)
                    
                    # Update risk assessment display
                    if risk_data:
                        risk_html = "<p><b>Risk Assessment Summary:</b></p>"
                        
                        for class_name, data in risk_data.items():
                            level = data["level"]
                            level_color = "green" if level == "Low" else "orange" if level == "Medium" else "red"
                            
                            risk_html += f"""
                            <div style='margin-bottom: 15px; padding: 10px; background-color: #f8f8f8; border-radius: 5px; border-left: 5px solid {level_color};'>
                                <b>{class_name} ({data["count"]})</b><br>
                                Risk Level: <span style='color: {level_color}; font-weight: bold;'>{level}</span><br>
                                <p>{data["description"]}</p>
                            </div>
                            """
                        
                        self.risk_label.setHtml(f"""
                        <html>
                        <body style='font-family: Arial; font-size: 14px;'>
                        {risk_html}
                        </body>
                        </html>
                        """)
                else:
                    # Fallback to direct VGG16 classification if no insects are detected by YOLO
                    self.results_label.setHtml("<html><body style='font-family: Arial; font-size: 14px;'><p>No insects detected by YOLO model. Attempting whole image classification...</p></body></html>")
                    
                    # Classify whole image
                    img_pil = Image.fromarray(image_rgb)
                    img_tensor = self.transform(img_pil).unsqueeze(0)
                    
                    with torch.no_grad():
                        img_tensor = img_tensor.to(self.device)
                        outputs = self.classification_model(img_tensor)
                        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                        _, predicted = torch.max(outputs, 1)
                        
                    # Get classification result
                    class_name = self.class_names[predicted.item()]
                    probability = probabilities[predicted.item()].item()
                    
                    # Get risk assessment
                    risk_level, risk_description = self.get_risk_assessment(class_name)
                    
                    # Create formatted HTML result
                    classification_html = f"""
                    <html>
                    <body style='font-family: Arial; font-size: 14px;'>
                    <h3>Full Image Classification Result:</h3>
                    <div style='margin: 15px; padding: 15px; background-color: #f0f0f0; border-radius: 10px; border-left: 5px solid #2196F3;'>
                        <p><b>Identified as:</b> {class_name}</p>
                        <p><b>Confidence:</b> {probability:.2%}</p>
                        <p><i>Note: No insects were detected by the object detection model. This is a classification of the entire image.</i></p>
                    </div>
                    </body>
                    </html>
                    """
                    
                    self.results_label.setHtml(classification_html)
                    
                    # Display risk assessment
                    risk_color = "green" if risk_level == "Low" else "orange" if risk_level == "Medium" else "red"
                    risk_html = f"""
                    <html>
                    <body style='font-family: Arial; font-size: 14px;'>
                    <h3>Risk Assessment:</h3>
                    <div style='margin: 15px; padding: 15px; background-color: #f8f8f8; border-radius: 10px; border-left: 5px solid {risk_color};'>
                        <p><b>Insect:</b> {class_name}</p>
                        <p><b>Risk Level:</b> <span style='color: {risk_color}; font-weight: bold;'>{risk_level}</span></p>
                        <p><b>Description:</b> {risk_description}</p>
                    </div>
                    </body>
                    </html>
                    """
                    
                    self.risk_label.setHtml(risk_html)
                
                # Update status labels
                self.tracking_status_label.setText("Detection: Image mode | Tracking: not applicable")
                self.fps_label.setText("Mode: Image Analysis")
                
            except Exception as e:
                self.status_label.setText(f"Status: Error analyzing image - {str(e)}")
                print(f"Error: {str(e)}")

def main():
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show the application
    window = InsectDetectionApp()
    window.show()
    
    # Run the application
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 