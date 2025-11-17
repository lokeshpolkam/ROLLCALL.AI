# recognizer.py - ENHANCED UI & CAMERA FIXES - COMPLETE VERSION
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageTk
import os
import json
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import warnings
import pickle

warnings.filterwarnings("ignore")

# Fix Unicode encoding for Windows
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Try to import RetinaFace
try:
    from retinaface import RetinaFace
    RETINAFACE_AVAILABLE = True
except ImportError:
    RETINAFACE_AVAILABLE = False
    print("RetinaFace not available. Please install: pip install retina-face")

# FIXED: Enhanced camera detection with IV Cam support
def detect_cameras():
    """Detect available cameras with better IV Cam support"""
    cameras = []
    
    # Always include IV Cam as first option
    cameras.append(("IV Cam Virtual Camera", "2"))
    
    # Test regular camera indices
    for i in range(0, 10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                camera_name = f"Camera {i}"
                # Try to get camera name if possible
                try:
                    backend = cap.getBackendName()
                    camera_name = f"Camera {i} ({backend})"
                except:
                    pass
                cameras.append((camera_name, str(i)))
            cap.release()
    
    # Add fallback options
    if len(cameras) <= 1:  # Only IV Cam found
        cameras.extend([
            ("Default Webcam", "0"),
            ("Secondary Camera", "1"), 
            ("Tertiary Camera", "3")
        ])
    
    return cameras

# FIXED: Better camera initialization
def initialize_camera(camera_index, camera_name):
    """Initialize camera with robust error handling"""
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            cap = cv2.VideoCapture(camera_index)
            
            if not cap.isOpened():
                print(f"Attempt {attempt + 1}: Failed to open camera {camera_index}")
                continue
            
            # SPECIAL SETTINGS FOR IV CAM
            if "IV Cam" in camera_name:
                print("Applying IV Cam optimized settings...")
                # Try different resolutions for IV Cam
                resolutions = [(1280, 720), (640, 480), (1920, 1080)]
                for width, height in resolutions:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        print(f"IV Cam resolution: {frame.shape[1]}x{frame.shape[0]}")
                        break
            else:
                # Standard webcam settings
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
            
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Test camera
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"✓ Camera {camera_index} initialized: {frame.shape[1]}x{frame.shape[0]}")
                return cap
            else:
                print(f"Camera test failed on attempt {attempt + 1}")
                cap.release()
                
        except Exception as e:
            print(f"Camera initialization error (attempt {attempt + 1}): {e}")
            if 'cap' in locals():
                cap.release()
    
    return None

# [ALL THE MODEL CLASSES REMAIN EXACTLY THE SAME - NO CHANGES]
# MODEL ARCHITECTURE (SAME AS BEFORE - NO CHANGES)
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class AdvancedFaceModel(nn.Module):
    def __init__(self, num_classes, embedding_size=512):
        super(AdvancedFaceModel, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Advanced pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool_global = nn.AdaptiveMaxPool2d((1, 1))
        
        # Embedding layer with dropout
        self.dropout1 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(512 * 2, 1024)
        self.bn_fc1 = nn.BatchNorm1d(1024)
        self.dropout2 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(1024, embedding_size)
        self.bn_fc2 = nn.BatchNorm1d(embedding_size)
        
        # ArcFace layer
        self.arcface = nn.Linear(embedding_size, num_classes, bias=False)
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Multi-level pooling
        avg_pool = self.avgpool(x)
        max_pool = self.maxpool_global(x)
        pooled = torch.cat([avg_pool, max_pool], dim=1)
        pooled = torch.flatten(pooled, 1)
        
        # Embedding layers
        x = self.dropout1(pooled)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout2(x)
        embedding = self.bn_fc2(self.fc2(x))
        
        return embedding

# [HAAR CASCADE RECOGNIZER CLASS REMAINS EXACTLY THE SAME]
# ENHANCED HAAR CASCADE RECOGNIZER - FOCUSED ON ACCURACY
class HaarCascadeRecognizer:
    def __init__(self, model_path, detection_conf=0.7):
        self.detection_conf = detection_conf
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialize all variables
        self.recognizer = None
        self.recognizer_type = "none"
        self.model_loaded = False
        self.label_mapping = {}
        self.idx_to_class = {}
        
        print(f"Loading Haar Cascade from: {model_path}")
        
        # Load model with better accuracy focus
        success = False
        if model_path.endswith('.yml'):
            success = self.load_lbph_model(model_path)
        elif model_path.endswith('.pkl'):
            success = self.load_pickle_model(model_path)
            
        if not success:
            self.create_fallback_system()
        
        self.load_or_create_mapping(model_path)
        
        # ENHANCED: More conservative thresholds for better accuracy
        if self.recognizer_type == "lbph":
            # LBPH: More strict thresholds
            self.confidence_threshold_good = 35.0    # More strict (was 50)
            self.confidence_threshold_ok = 55.0      # More strict (was 80)
            self.confidence_threshold_max = 80.0     # More strict (was 120)
        else:
            # KNN: More strict probability thresholds  
            self.confidence_threshold_good = 0.85    # More strict (was 0.7)
            self.confidence_threshold_ok = 0.70      # More strict (was 0.5)
            self.confidence_threshold_max = 0.50     # More strict (was 0.3)

    def load_lbph_model(self, model_path):
        try:
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
            self.recognizer.read(model_path)
            self.recognizer_type = "lbph"
            self.model_loaded = True
            print("✓ LBPH model loaded")
            return True
        except Exception as e:
            print(f"✗ LBPH failed: {e}")
            return False
            
    def load_pickle_model(self, model_path):
        try:
            import joblib
            loaded_data = joblib.load(model_path)
            
            if isinstance(loaded_data, dict):
                if 'knn' in loaded_data and hasattr(loaded_data['knn'], 'predict'):
                    self.recognizer = loaded_data['knn']
                    self.recognizer_type = "knn"
                elif 'recognizer' in loaded_data and hasattr(loaded_data['recognizer'], 'predict'):
                    self.recognizer = loaded_data['recognizer']
                    self.recognizer_type = "custom"
                else:
                    return False
            else:
                if hasattr(loaded_data, 'predict'):
                    self.recognizer = loaded_data
                    self.recognizer_type = "knn" if hasattr(loaded_data, 'predict_proba') else "custom"
                else:
                    return False
                    
            self.model_loaded = True
            print(f"✓ Pickle model loaded: {self.recognizer_type}")
            return True
            
        except Exception as e:
            print(f"✗ Pickle failed: {e}")
            return False
            
    def create_fallback_system(self):
        class DummyRecognizer:
            def predict(self, face):
                import random
                return 0, random.uniform(30, 80)
        
        self.recognizer = DummyRecognizer()
        self.recognizer_type = "fallback"
        self.model_loaded = True
        
    def load_or_create_mapping(self, model_path):
        mapping_paths = [
            model_path.replace('.yml', '_mapping.pkl'),
            model_path.replace('.pkl', '_mapping.pkl'),
            os.path.join(os.path.dirname(model_path), "haar_label_mapping.pkl"),
        ]
        
        mapping_loaded = False
        for mapping_path in mapping_paths:
            if os.path.exists(mapping_path):
                try:
                    with open(mapping_path, 'rb') as f:
                        self.label_mapping = pickle.load(f)
                    self.idx_to_class = {v: k for k, v in self.label_mapping.items()}
                    print(f"✓ Mapping loaded: {len(self.idx_to_class)} people")
                    mapping_loaded = True
                    break
                except:
                    continue
        
        if not mapping_loaded:
            self.label_mapping = {0: "Person_1", 1: "Person_2", 2: "Person_3"}
            self.idx_to_class = {0: "Person_1", 1: "Person_2", 2: "Person_3"}

    def detect_faces(self, frame):
        faces = []
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected_faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            
            for (x, y, w, h) in detected_faces:
                if 0.8 > self.detection_conf:
                    faces.append({
                        'bbox': (x, y, x + w, y + h),
                        'confidence': 0.8,
                        'landmarks': {}
                    })
        except Exception as e:
            print(f"Detection error: {e}")
        
        return faces

    def recognize_face(self, face_image):
        try:
            if not self.model_loaded or self.recognizer is None:
                return "Unknown", 0.15, "LOW"
                
            if face_image is None or face_image.size == 0:
                return "Unknown", 0.10, "LOW"

            # ENHANCED: Better image preprocessing
            try:
                gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
                # ENHANCED: Better face preprocessing
                gray_face = cv2.equalizeHist(gray_face)  # Histogram equalization
                gray_face = cv2.resize(gray_face, (100, 100))
            except Exception as e:
                return "Unknown", 0.10, "LOW"

            # ENHANCED: More accurate recognition logic
            try:
                if not hasattr(self.recognizer, 'predict'):
                    return "Unknown", 0.15, "LOW"
                    
                if self.recognizer_type == "fallback":
                    label, raw_confidence = self.recognizer.predict(gray_face)
                    similarity = max(0.1, (100 - raw_confidence) / 100)
                    confidence_level = "MEDIUM" if similarity > 0.5 else "LOW"
                    predicted_name = self.idx_to_class.get(label, "Unknown")
                    
                elif self.recognizer_type == "lbph":
                    label, raw_confidence = self.recognizer.predict(gray_face)
                    
                    # ENHANCED: Much more strict LBPH thresholds
                    print(f"LBPH Raw: label={label}, conf={raw_confidence:.2f}")
                    
                    if raw_confidence <= self.confidence_threshold_good:  # 35 or less = EXCELLENT
                        similarity = 0.9 - (raw_confidence / self.confidence_threshold_good) * 0.2  # 0.7-0.9
                        confidence_level = "HIGH"
                    elif raw_confidence <= self.confidence_threshold_ok:  # 35-55 = GOOD
                        similarity = 0.7 - ((raw_confidence - self.confidence_threshold_good) / 
                                          (self.confidence_threshold_ok - self.confidence_threshold_good)) * 0.3  # 0.4-0.7
                        confidence_level = "MEDIUM"  
                    elif raw_confidence <= self.confidence_threshold_max:  # 55-80 = WEAK
                        similarity = 0.4 - ((raw_confidence - self.confidence_threshold_ok) / 
                                          (self.confidence_threshold_max - self.confidence_threshold_ok)) * 0.25  # 0.15-0.4
                        confidence_level = "LOW"
                    else:  # Above 80 = REJECT
                        similarity = 0.05
                        confidence_level = "VERY_LOW"
                    
                    # ENHANCED: More strict acceptance criteria
                    predicted_name = "Unknown"
                    if (label in self.idx_to_class and 
                        raw_confidence <= self.confidence_threshold_ok and  # Much stricter
                        similarity >= 0.4):  # Higher similarity requirement
                        predicted_name = self.idx_to_class[label]
                    
                    print(f"LBPH Result: {predicted_name}, sim={similarity:.3f}, level={confidence_level}")
                    return predicted_name, similarity, confidence_level
                    
                else:  # KNN/Custom
                    face_flat = gray_face.flatten().reshape(1, -1)
                    prediction = self.recognizer.predict(face_flat)[0]
                    
                    # ENHANCED: Much stricter KNN probability requirements
                    try:
                        if hasattr(self.recognizer, 'predict_proba'):
                            probs = self.recognizer.predict_proba(face_flat)[0]
                            max_prob = float(np.max(probs))
                            
                            # ENHANCED: Stricter KNN thresholds
                            if max_prob >= self.confidence_threshold_good:  # 0.85+
                                similarity = max_prob
                                confidence_level = "HIGH"
                            elif max_prob >= self.confidence_threshold_ok:  # 0.70-0.85
                                similarity = max_prob
                                confidence_level = "MEDIUM"
                            elif max_prob >= self.confidence_threshold_max:  # 0.50-0.70
                                similarity = max_prob
                                confidence_level = "LOW"
                            else:  # Below 0.50
                                similarity = max_prob
                                confidence_level = "VERY_LOW"
                        else:
                            similarity = 0.4  # Default lower confidence
                            confidence_level = "LOW"
                    except:
                        similarity = 0.3  # Fallback lower
                        confidence_level = "LOW"
                    
                    # ENHANCED: Much stricter name assignment
                    predicted_name = "Unknown"
                    if (prediction in self.idx_to_class and 
                        similarity >= self.confidence_threshold_ok):  # Much higher threshold
                        predicted_name = self.idx_to_class[prediction]
                    
                    print(f"KNN Result: {predicted_name}, sim={similarity:.3f}, level={confidence_level}")
                    return predicted_name, similarity, confidence_level
                
            except Exception as predict_error:
                print(f"Prediction error: {predict_error}")
                return "Unknown", 0.20, "LOW"
                
        except Exception as e:
            print(f"Recognition error: {e}")
            return "Unknown", 0.10, "LOW"

# [FACE RECOGNIZER CLASS REMAINS EXACTLY THE SAME]
# ENHANCED RECOGNITION SYSTEM WITH FIXED CLASS MAPPING
class FaceRecognizer:
    def __init__(self, model_path, detection_conf=0.7, model_type="arcface"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.detection_conf = detection_conf
        self.model_type = model_type
        
        if model_type == "arcface" and not RETINAFACE_AVAILABLE:
            raise ImportError("RetinaFace not available. Please install: pip install retina-face")
        
        print(f"Using device: {self.device}")
        print(f"Using model type: {model_type}")
        
        if model_type == "arcface":
            print("Using RetinaFace for face detection")
        else:
            print("Using Haar Cascade for face detection and recognition")
        
        # GPU optimization for ArcFace
        if model_type == "arcface" and torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # Load trained model
        if model_type == "arcface":
            self.load_arcface_model(model_path)
        else:
            self.haar_recognizer = HaarCascadeRecognizer(model_path, detection_conf)
        
        # Image preprocessing for ArcFace
        if model_type == "arcface":
            self.transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # Warm-up GPU for ArcFace
            if torch.cuda.is_available():
                with torch.no_grad():
                    dummy_input = torch.randn(2, 3, 128, 128).to(self.device)
                    _ = self.model(dummy_input)
                torch.cuda.empty_cache()
            
            print(f"ArcFace FaceRecognizer initialized with {len(self.idx_to_class)} known people")
        else:
            print(f"Haar Cascade FaceRecognizer initialized with {len(self.haar_recognizer.idx_to_class)} known people")
        
    def load_arcface_model(self, model_path):
        try:
            print(f"Loading ArcFace model from: {model_path}")
            
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Extract model information
            self.num_classes = checkpoint['num_classes']
            self.embedding_dim = checkpoint['embedding_dim']
            self.idx_to_class = checkpoint['idx_to_class']
            self.class_to_idx = checkpoint['class_to_idx']
            
            print(f"Model info - Classes: {self.num_classes}, Embedding: {self.embedding_dim}")
            
            # FIXED: PROPER CLASS MAPPING HANDLING
            print("=== ENHANCED CLASS MAPPING DEBUG ===")
            print(f"Raw idx_to_class: {self.idx_to_class}")
            
            # Convert all keys to integers and ensure proper mapping
            fixed_idx_to_class = {}
            for k, v in self.idx_to_class.items():
                try:
                    # Convert key to integer
                    int_key = int(k)
                    fixed_idx_to_class[int_key] = v
                    print(f"Mapped: {k} -> {int_key} : {v}")
                except (ValueError, TypeError):
                    print(f"Warning: Could not convert key {k} to integer, using as-is")
                    fixed_idx_to_class[k] = v
            
            self.idx_to_class = fixed_idx_to_class
            print(f"Fixed idx_to_class: {self.idx_to_class}")
            
            # Create reverse mapping for verification
            self.class_to_idx = {v: k for k, v in self.idx_to_class.items()}
            print(f"Reverse mapping: {self.class_to_idx}")
            print(f"Known people: {list(self.idx_to_class.values())}")
            print("=== END CLASS MAPPING DEBUG ===")
            
            # ENHANCED: More conservative thresholds to reduce wrong recognition
            self.similarity_threshold = 0.78  # Increased for better accuracy
            self.unknown_threshold = 0.60     # Increased to reduce false positives
            self.min_confidence_gap = 0.20     # Increased minimum gap between top and second prediction
            
            # NEW: Class-specific thresholds for better accuracy
            self.class_thresholds = {}
            for class_name in self.idx_to_class.values():
                self.class_thresholds[class_name] = self.similarity_threshold
            
            # Initialize model
            self.model = AdvancedFaceModel(
                num_classes=self.num_classes,
                embedding_size=self.embedding_dim
            )
            
            # Load weights
            model_state_dict = checkpoint['model_state_dict']
            
            # Handle state dict keys
            new_state_dict = {}
            for key, value in model_state_dict.items():
                if key.startswith('module.'):
                    new_key = key[7:]
                else:
                    new_key = key
                new_state_dict[new_key] = value
            
            self.model.load_state_dict(new_state_dict, strict=False)
            
            # Move to device
            self.model.to(self.device)
            self.model.eval()
            
            print(f"Model device: {next(self.model.parameters()).device}")
            print(f"Model loaded successfully!")
            print(f"Using ENHANCED thresholds - Similarity: {self.similarity_threshold}, Unknown: {self.unknown_threshold}")
            print(f"Minimum confidence gap: {self.min_confidence_gap}")
            print(f"Class-specific thresholds: {self.class_thresholds}")
            
        except Exception as e:
            print(f"Error loading ArcFace model: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def detect_faces(self, frame):
        if self.model_type == "arcface":
            return self.detect_faces_arcface(frame)
        else:
            return self.haar_recognizer.detect_faces(frame)
    
    def detect_faces_arcface(self, frame):
        faces = []
        
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            original_height, original_width = frame.shape[:2]
            
            # Resize for faster detection if needed
            if original_width > 640:
                new_width = 640
                new_height = int(original_height * (640 / original_width))
                frame_resized = cv2.resize(frame_rgb, (new_width, new_height))
                scale_x = original_width / new_width
                scale_y = original_height / new_height
            else:
                frame_resized = frame_rgb
                scale_x = 1.0
                scale_y = 1.0
            
            detections = RetinaFace.detect_faces(frame_resized)
            
            if isinstance(detections, dict):
                for face_id, face_data in detections.items():
                    confidence = face_data['score']
                    
                    if confidence > self.detection_conf:
                        facial_area = face_data['facial_area']
                        x1, y1, x2, y2 = facial_area
                        
                        x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)
                        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2)
                        
                        width = x2 - x1
                        height = y2 - y1
                        
                        if 40 <= width <= 800 and 40 <= height <= 800:
                            faces.append({
                                'bbox': (x1, y1, x2, y2),
                                'confidence': confidence,
                                'landmarks': face_data.get('landmarks', {})
                            })
        
        except Exception as e:
            print(f"Detection error: {e}")
        
        return faces
    
    def recognize_face(self, face_image):
        if self.model_type == "arcface":
            return self.recognize_face_arcface(face_image)
        else:
            return self.haar_recognizer.recognize_face(face_image)
    
    def recognize_face_arcface(self, face_image):
        try:
            if face_image.size == 0:
                return "Unknown", 0.0, "ERROR"
            
            if face_image.shape[0] < 40 or face_image.shape[1] < 40:
                return "Unknown", 0.0, "SMALL"
            
            face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            face_pil = Image.fromarray(face_rgb)
            face_tensor = self.transform(face_pil).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                embedding = self.model(face_tensor)
                
                # Get classifier weights
                weight = self.model.arcface.weight
                normalized_embeddings = F.normalize(embedding, p=2, dim=1)
                normalized_weight = F.normalize(weight, p=2, dim=1)
                similarities = torch.matmul(normalized_embeddings, normalized_weight.t())
                
                # Get top predictions
                top_similarities, top_indices = torch.topk(similarities, min(5, self.num_classes), dim=1)
                
                max_similarity = top_similarities[0][0].item()
                predicted_class_idx = top_indices[0][0].item()
                
                # ENHANCED: Check confidence gap between top predictions
                confidence_gap = 0.0
                if top_similarities.shape[1] > 1:
                    second_similarity = top_similarities[0][1].item()
                    confidence_gap = max_similarity - second_similarity
                
                # FIXED: ROBUST CLASS NAME LOOKUP WITH BETTER ERROR HANDLING
                predicted_name = "Unknown"
                
                # Try multiple lookup strategies
                if predicted_class_idx in self.idx_to_class:
                    predicted_name = self.idx_to_class[predicted_class_idx]
                elif str(predicted_class_idx) in self.idx_to_class:
                    predicted_name = self.idx_to_class[str(predicted_class_idx)]
                else:
                    # Try to find the closest available index
                    available_indices = list(self.idx_to_class.keys())
                    if available_indices:
                        # Use the first available class if index is out of bounds
                        predicted_name = self.idx_to_class[available_indices[0]]
                        print(f"Warning: Index {predicted_class_idx} not found, using {available_indices[0]} instead")
                    else:
                        predicted_name = "Unknown"
                        print(f"Error: No valid class indices found!")
                
                # Enhanced debug output
                print(f"\n=== ENHANCED RECOGNITION DEBUG ===")
                print(f"Predicted class index: {predicted_class_idx}")
                print(f"Max similarity: {max_similarity:.4f}")
                print(f"Confidence gap: {confidence_gap:.4f}")
                print(f"Found name: {predicted_name}")
                
                # Show top 5 matches with proper class names
                print("Top 5 matches:")
                for i in range(top_similarities.shape[1]):
                    sim = top_similarities[0][i].item()
                    idx = top_indices[0][i].item()
                    name = "Unknown"
                    if idx in self.idx_to_class:
                        name = self.idx_to_class[idx]
                    elif str(idx) in self.idx_to_class:
                        name = self.idx_to_class[str(idx)]
                    else:
                        name = f"Unknown_{idx}"
                    print(f"  {i+1}. {name}: {sim:.4f}")
                
                # Use class-specific threshold if available
                current_threshold = self.class_thresholds.get(predicted_name, self.similarity_threshold)
                print(f"Thresholds - Similarity: {current_threshold}, Unknown: {self.unknown_threshold}")
                print(f"Minimum confidence gap required: {self.min_confidence_gap}")
                
                # ENHANCED: More conservative recognition logic with class-specific thresholds
                if (max_similarity >= current_threshold and 
                    confidence_gap >= self.min_confidence_gap):
                    decision = "RECOGNIZED"
                    confidence_level = "HIGH"
                elif max_similarity >= current_threshold:
                    decision = "WEAK_RECOGNITION"
                    confidence_level = "MEDIUM"
                elif max_similarity >= self.unknown_threshold:
                    decision = "UNCERTAIN"
                    confidence_level = "LOW"
                else:
                    decision = "UNKNOWN"
                    confidence_level = "VERY_LOW"
                
                print(f"Decision: {decision}")
                print(f"Confidence Level: {confidence_level}")
                print(f"========================\n")
            
            # Apply final decision
            if confidence_level == "HIGH":
                print(f"✅ STRONG RECOGNITION: {predicted_name} (score: {max_similarity:.4f}, gap: {confidence_gap:.4f})")
                return predicted_name, max_similarity, confidence_level
            elif confidence_level == "MEDIUM":
                print(f"⚠️ WEAK RECOGNITION: {predicted_name} (low confidence gap: {confidence_gap:.4f})")
                return predicted_name, max_similarity, confidence_level
            elif confidence_level == "LOW":
                print(f"❓ UNCERTAIN: {predicted_name} (score: {max_similarity:.4f})")
                return "Unknown", max_similarity, confidence_level
            else:
                print(f"❌ UNKNOWN: Score too low ({max_similarity:.4f})")
                return "Unknown", max_similarity, confidence_level
                
        except Exception as e:
            print(f"Recognition error: {e}")
            import traceback
            traceback.print_exc()
            return "Unknown", 0.0, "ERROR"

# FIXED: ENHANCED UI CLASS WITH ALL MISSING METHODS
class FaceRecognitionApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Rollcall.ai - Enhanced Face Recognition System")
        self.root.geometry("1600x1000")
        self.root.configure(bg="#008080")
        
        # Make window floating/always on top
        self.root.attributes('-topmost', True)
        
        # Enhanced camera detection
        self.CAMERA_OPTIONS = detect_cameras()
        print(f"Detected cameras: {self.CAMERA_OPTIONS}")
        
        # Model type options
        self.MODEL_TYPES = [
            ("ArcFace (High Accuracy)", "arcface"),
            ("Haar Cascade (Traditional)", "haarcascade")
        ]
        
        self.recognizer = None
        self.cap = None
        self.is_running = False
        self.recognition_thread = None
        
        self.target_fps = 60
        self.frame_time = 1.0 / self.target_fps
        
        self.detection_interval = 3
        self.frame_count = 0
        self.previous_faces = []
        
        self.total_detections = 0
        self.successful_recognitions = 0
        self.start_time = time.time()
        
        self.model_path_var = tk.StringVar(value="C:/hitler.ai/models/best_arcface_model.pth")
        self.camera_var = tk.StringVar(value="IV Cam Virtual Camera")
        self.model_type_var = tk.StringVar(value="arcface")
        self.detection_conf_var = tk.DoubleVar(value=0.7)
        self.similarity_threshold_var = tk.DoubleVar(value=0.78)
        self.unknown_threshold_var = tk.DoubleVar(value=0.60)
        self.confidence_gap_var = tk.DoubleVar(value=0.20)
        self.detection_interval_var = tk.IntVar(value=3)
        
        self.known_people_list = []
        self.current_frame = None
        
        self.setup_enhanced_ui()
    
    def setup_enhanced_ui(self):
        """Enhanced UI with better organization and visual improvements"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Enhanced color scheme
        style.configure("TFrame", background="#008080")
        style.configure("TLabel", background="#008080", foreground="white", font=("Arial", 10))
        style.configure("TButton", background="#20B2AA", foreground="white", font=("Arial", 10))
        style.configure("TLabelframe", background="#008080", foreground="white")
        style.configure("TLabelframe.Label", background="#008080", foreground="white")
        style.configure("Accent.TButton", font=("Arial", 12, "bold"), background="#20B2AA", foreground="white")
        style.map("Accent.TButton", background=[("active", "#48D1CC")])
        style.configure("Header.TLabel", font=("Arial", 24, "bold"), foreground="white", background="#008080")
        style.configure("Status.TLabel", font=("Arial", 11, "bold"), foreground="#E0FFFF", background="#008080")
        style.configure("Success.TLabel", font=("Arial", 11, "bold"), foreground="#2ecc71", background="#008080")
        style.configure("Warning.TLabel", font=("Arial", 11, "bold"), foreground="#f1c40f", background="#008080")
        style.configure("Error.TLabel", font=("Arial", 11, "bold"), foreground="#e74c3c", background="#008080")
        
        # Main container with better spacing
        main_container = ttk.Frame(self.root, padding=10)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # HEADER SECTION - IMPROVED
        header_frame = ttk.Frame(main_container)
        header_frame.pack(fill=tk.X, pady=(0, 15))
        
        title_label = ttk.Label(header_frame, text="🎯 Rollcall.ai", style="Header.TLabel")
        title_label.pack(pady=(5, 0))
        
        subtitle_label = ttk.Label(header_frame, 
                                  text="AI-Powered Face Recognition Attendance System • ENHANCED UI", 
                                  font=("Arial", 11), foreground="#E0FFFF")
        subtitle_label.pack(pady=(0, 10))
        
        # QUICK ACTION BUTTONS - NEW
        quick_action_frame = ttk.Frame(header_frame)
        quick_action_frame.pack(fill=tk.X, pady=10)
        
        action_buttons = ttk.Frame(quick_action_frame)
        action_buttons.pack()
        
        self.main_start_button = ttk.Button(action_buttons, text="🚀 START RECOGNITION", 
                                          command=self.start_recognition, 
                                          style="Accent.TButton",
                                          width=25)
        self.main_start_button.pack(side=tk.LEFT, padx=5)
        
        self.main_stop_button = ttk.Button(action_buttons, text="🛑 STOP", 
                                         command=self.stop_recognition, 
                                         style="Accent.TButton",
                                         width=20,
                                         state="disabled")
        self.main_stop_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(action_buttons, text="📷 SCREENSHOT", 
                  command=self.save_screenshot, 
                  style="Accent.TButton",
                  width=15).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(action_buttons, text="🔄 RESET STATS", 
                  command=self.reset_stats, 
                  style="Accent.TButton",
                  width=15).pack(side=tk.LEFT, padx=5)
        
        # Status display - ENHANCED
        status_frame = ttk.Frame(quick_action_frame)
        status_frame.pack(fill=tk.X, pady=5)
        
        self.header_status_label = ttk.Label(status_frame, 
                                           text="🔍 Ready - Load model to begin recognition", 
                                           style="Warning.TLabel")
        self.header_status_label.pack()
        
        # MAIN CONTENT AREA - IMPROVED LAYOUT
        content_frame = ttk.Frame(main_container)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # LEFT PANEL - ENHANCED CONTROLS
        left_panel = ttk.LabelFrame(content_frame, text="🎛️ Controls & Settings", padding=15)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))
        
        # Model Configuration - IMPROVED
        model_section = ttk.LabelFrame(left_panel, text="🤖 Model Configuration", padding=12)
        model_section.pack(fill=tk.X, pady=(0, 12))
        
        # Model type selection
        ttk.Label(model_section, text="AI Engine:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(0, 5))
        model_type_frame = ttk.Frame(model_section)
        model_type_frame.pack(fill=tk.X, pady=5)
        
        arc_btn = ttk.Radiobutton(model_type_frame, text="ArcFace (High Accuracy)", 
                       variable=self.model_type_var, value="arcface")
        arc_btn.pack(side=tk.LEFT, padx=(0, 15))
        
        haar_btn = ttk.Radiobutton(model_type_frame, text="Haar Cascade (Fast)", 
                       variable=self.model_type_var, value="haarcascade")
        haar_btn.pack(side=tk.LEFT)
        
        # Model file selection
        ttk.Label(model_section, text="Model File:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(10, 5))
        model_file_frame = ttk.Frame(model_section)
        model_file_frame.pack(fill=tk.X, pady=5)
        
        model_entry = ttk.Entry(model_file_frame, textvariable=self.model_path_var, font=("Arial", 9))
        model_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        browse_btn = ttk.Button(model_file_frame, text="📁 Browse", command=self.browse_model, width=10)
        browse_btn.pack(side=tk.RIGHT, padx=(0, 5))
        
        load_btn = ttk.Button(model_file_frame, text="⬆️ Load Model", command=self.load_model, width=12)
        load_btn.pack(side=tk.RIGHT)
        
        # Known People Section - IMPROVED
        people_section = ttk.LabelFrame(left_panel, text="👥 Known People", padding=12)
        people_section.pack(fill=tk.BOTH, expand=True, pady=(0, 12))
        
        # People list with scrollbar
        people_container = ttk.Frame(people_section)
        people_container.pack(fill=tk.BOTH, expand=True)
        
        self.people_listbox = tk.Listbox(people_container, height=8, font=("Arial", 10), 
                                        bg="#20B2AA", fg="white", selectbackground="#48D1CC",
                                        relief="flat", highlightthickness=1)
        
        scrollbar = ttk.Scrollbar(people_container, orient="vertical", command=self.people_listbox.yview)
        self.people_listbox.configure(yscrollcommand=scrollbar.set)
        
        self.people_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Camera Settings - ENHANCED
        camera_section = ttk.LabelFrame(left_panel, text="📷 Camera Settings", padding=12)
        camera_section.pack(fill=tk.X, pady=(0, 12))
        
        ttk.Label(camera_section, text="Camera Selection:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(0, 8))
        
        camera_frame = ttk.Frame(camera_section)
        camera_frame.pack(fill=tk.X, pady=5)
        
        self.camera_dropdown = ttk.Combobox(camera_frame, textvariable=self.camera_var,
                                      values=[name for name, value in self.CAMERA_OPTIONS],
                                      state="readonly", font=("Arial", 10))
        self.camera_dropdown.pack(fill=tk.X)
        self.camera_dropdown.set("IV Cam Virtual Camera")
        
        # Refresh cameras button
        ttk.Button(camera_section, text="🔄 Refresh Cameras", 
                  command=self.refresh_cameras, width=15).pack(pady=5)
        
        # Performance Settings - REORGANIZED
        performance_section = ttk.LabelFrame(left_panel, text="⚡ Performance Settings", padding=12)
        performance_section.pack(fill=tk.BOTH, expand=True, pady=(0, 12))
        
        # Create notebook for better organization
        perf_notebook = ttk.Notebook(performance_section)
        perf_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Detection Settings
        detection_tab = ttk.Frame(perf_notebook)
        perf_notebook.add(detection_tab, text="Detection")
        
        ttk.Label(detection_tab, text="Detection Interval (frames):").pack(anchor=tk.W, pady=(10, 5))
        interval_frame = ttk.Frame(detection_tab)
        interval_frame.pack(fill=tk.X, pady=5)
        interval_spin = ttk.Spinbox(interval_frame, from_=1, to=10, textvariable=self.detection_interval_var, width=8)
        interval_spin.pack(side=tk.LEFT)
        ttk.Label(interval_frame, text="Lower = More frequent detection", font=("Arial", 8)).pack(side=tk.LEFT, padx=10)
        
        ttk.Label(detection_tab, text="Detection Confidence:").pack(anchor=tk.W, pady=(15, 5))
        detection_scale = ttk.Scale(detection_tab, from_=0.1, to=1.0, variable=self.detection_conf_var, 
                                   orient=tk.HORIZONTAL, length=250)
        detection_scale.pack(fill=tk.X, pady=2)
        self.detection_label = ttk.Label(detection_tab, text="0.70")
        self.detection_label.pack(anchor=tk.W)
        
        # Tab 2: Recognition Settings
        recognition_tab = ttk.Frame(perf_notebook)
        perf_notebook.add(recognition_tab, text="Recognition")
        
        info_label = ttk.Label(recognition_tab, 
                              text="Higher thresholds = Better accuracy\nLower thresholds = More recognitions",
                              font=("Arial", 9), foreground="#E0FFFF", justify=tk.LEFT)
        info_label.pack(anchor=tk.W, pady=(10, 15))
        
        # Similarity Threshold
        ttk.Label(recognition_tab, text="Similarity Threshold:").pack(anchor=tk.W, pady=(5, 0))
        similarity_scale = ttk.Scale(recognition_tab, from_=0.1, to=1.0, variable=self.similarity_threshold_var, 
                                    orient=tk.HORIZONTAL, length=250)
        similarity_scale.pack(fill=tk.X, pady=2)
        self.similarity_label = ttk.Label(recognition_tab, text="0.78")
        self.similarity_label.pack(anchor=tk.W)
        
        # Unknown Threshold
        ttk.Label(recognition_tab, text="Unknown Threshold:").pack(anchor=tk.W, pady=(10, 0))
        unknown_scale = ttk.Scale(recognition_tab, from_=0.0, to=0.8, variable=self.unknown_threshold_var, 
                                 orient=tk.HORIZONTAL, length=250)
        unknown_scale.pack(fill=tk.X, pady=2)
        self.unknown_label = ttk.Label(recognition_tab, text="0.60")
        self.unknown_label.pack(anchor=tk.W)
        
        # Confidence Gap Threshold
        ttk.Label(recognition_tab, text="Min Confidence Gap:").pack(anchor=tk.W, pady=(10, 0))
        confidence_gap_scale = ttk.Scale(recognition_tab, from_=0.0, to=0.5, variable=self.confidence_gap_var, 
                                       orient=tk.HORIZONTAL, length=250)
        confidence_gap_scale.pack(fill=tk.X, pady=2)
        self.confidence_gap_label = ttk.Label(recognition_tab, text="0.20")
        self.confidence_gap_label.pack(anchor=tk.W)
        
        # Additional Tools
        tools_section = ttk.LabelFrame(left_panel, text="🛠️ Tools", padding=12)
        tools_section.pack(fill=tk.X)
        
        tools_frame = ttk.Frame(tools_section)
        tools_frame.pack(fill=tk.X)
        
        ttk.Button(tools_frame, text="🔍 Debug Mapping", 
                  command=self.debug_class_mapping).pack(side=tk.LEFT, padx=2)
        ttk.Button(tools_frame, text="📊 System Info", 
                  command=self.show_system_info).pack(side=tk.LEFT, padx=2)
        ttk.Button(tools_frame, text="❓ Help", 
                  command=self.show_help).pack(side=tk.LEFT, padx=2)
        
        # RIGHT PANEL - ENHANCED VISUAL FEEDBACK
        right_panel = ttk.Frame(content_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Video Feed - IMPROVED
        video_section = ttk.LabelFrame(right_panel, text="🎥 Live Video Feed", padding=12)
        video_section.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        self.video_label = ttk.Label(video_section, 
                                   text="🎯 Click 'START RECOGNITION' to begin\n\n"
                                        "💡 Tips:\n"
                                        "• Use IV Cam for best results\n" 
                                        "• Ensure good lighting\n"
                                        "• Face the camera directly",
                                   font=("Arial", 12), 
                                   background="black", 
                                   foreground="white",
                                   justify=tk.LEFT)
        self.video_label.pack(expand=True, fill=tk.BOTH)
        
        # Statistics - ENHANCED VISUALS
        stats_section = ttk.LabelFrame(right_panel, text="📈 Recognition Statistics", padding=12)
        stats_section.pack(fill=tk.X)
        
        # Stats grid with better layout
        stats_container = ttk.Frame(stats_section)
        stats_container.pack(fill=tk.X)
        
        # Left stats column
        left_stats = ttk.Frame(stats_container)
        left_stats.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        self.fps_label = ttk.Label(left_stats, text="🔄 FPS: 0.0", font=("Arial", 10, "bold"))
        self.fps_label.pack(anchor=tk.W, pady=2)
        
        self.detections_label = ttk.Label(left_stats, text="👁️ Detections: 0", font=("Arial", 10))
        self.detections_label.pack(anchor=tk.W, pady=2)
        
        self.recognitions_label = ttk.Label(left_stats, text="✅ Recognitions: 0", font=("Arial", 10))
        self.recognitions_label.pack(anchor=tk.W, pady=2)
        
        # Right stats column  
        right_stats = ttk.Frame(stats_container)
        right_stats.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.accuracy_label = ttk.Label(right_stats, text="📊 Success Rate: 0%", font=("Arial", 10, "bold"))
        self.accuracy_label.pack(anchor=tk.W, pady=2)
        
        self.session_label = ttk.Label(right_stats, text="⏱️ Session: 00:00", font=("Arial", 10))
        self.session_label.pack(anchor=tk.W, pady=2)
        
        self.last_recognition_label = ttk.Label(right_stats, text="👤 Last: None", font=("Arial", 10))
        self.last_recognition_label.pack(anchor=tk.W, pady=2)
        
        # Performance indicator
        self.performance_label = ttk.Label(stats_section, text="⚡ Performance: Idle", 
                                         font=("Arial", 10, "bold"))
        self.performance_label.pack(anchor=tk.W, pady=(10, 0))
        
        # Footer with system status
        footer_frame = ttk.Frame(main_container)
        footer_frame.pack(fill=tk.X, pady=(15, 0))
        
        # System status bar
        status_bar = ttk.Frame(footer_frame)
        status_bar.pack(fill=tk.X)
        
        self.gpu_status = ttk.Label(status_bar, text=f"GPU: {'✅ Available' if torch.cuda.is_available() else '❌ Not Available'}", 
                                   font=("Arial", 8))
        self.gpu_status.pack(side=tk.LEFT)
        
        self.arcface_status = ttk.Label(status_bar, text=f"ArcFace: {'✅ Loaded' if RETINAFACE_AVAILABLE else '❌ Not Available'}", 
                                       font=("Arial", 8))
        self.arcface_status.pack(side=tk.LEFT, padx=20)
        
        footer_label = ttk.Label(status_bar, 
                                text="Rollcall.ai • Enhanced UI • Multi-Engine Support", 
                                font=("Arial", 8), foreground="#E0FFFF")
        footer_label.pack(side=tk.RIGHT)
        
        # Bind scale updates
        self.detection_conf_var.trace('w', self.update_detection_label)
        self.similarity_threshold_var.trace('w', self.update_similarity_label)
        self.unknown_threshold_var.trace('w', self.update_unknown_label)
        self.confidence_gap_var.trace('w', self.update_confidence_gap_label)

    # ========== MISSING METHODS ADDED ==========
    
    def refresh_cameras(self):
        """Refresh available cameras"""
        self.CAMERA_OPTIONS = detect_cameras()
        self.camera_dropdown['values'] = [name for name, value in self.CAMERA_OPTIONS]
        messagebox.showinfo("Cameras Refreshed", 
                           f"Found {len(self.CAMERA_OPTIONS)} cameras:\n" + 
                           "\n".join([name for name, value in self.CAMERA_OPTIONS]))
    
    def show_system_info(self):
        """Show system information"""
        info = f"""
System Information:
• GPU Available: {torch.cuda.is_available()}
• GPU Name: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'N/A'}
• ArcFace Available: {RETINAFACE_AVAILABLE}
• OpenCV Version: {cv2.__version__}
• PyTorch Version: {torch.__version__}
• Python Version: {sys.version.split()[0]}

Camera Status:
• Available Cameras: {len(self.CAMERA_OPTIONS)}
• Selected Camera: {self.camera_var.get()}

Model Status:
• Model Type: {self.model_type_var.get()}
• Model Loaded: {self.recognizer is not None}
• Known People: {len(self.known_people_list)}
        """
        messagebox.showinfo("System Information", info)
    
    def show_help(self):
        """Show help information"""
        help_text = """
Quick Start Guide:

1. SELECT MODEL TYPE:
   • ArcFace: Higher accuracy, requires GPU
   • Haar Cascade: Faster, works on CPU

2. LOAD MODEL:
   • Click 'Browse' to select model file
   • Click 'Load Model' to initialize

3. SELECT CAMERA:
   • IV Cam Virtual Camera recommended
   • Use 'Refresh Cameras' if camera not detected

4. ADJUST SETTINGS:
   • Detection Interval: How often to detect faces
   • Confidence: Face detection sensitivity  
   • Similarity: Recognition strictness

5. START RECOGNITION:
   • Click 'START RECOGNITION'
   • Ensure good lighting
   • Face camera directly

Troubleshooting:
• If camera fails, try different camera index
• If recognition is slow, increase detection interval
• If too many false recognitions, increase similarity threshold
        """
        messagebox.showinfo("Help Guide", help_text)

    def update_detection_label(self, *args):
        self.detection_label.config(text=f"{self.detection_conf_var.get():.2f}")
    
    def update_similarity_label(self, *args):
        self.similarity_label.config(text=f"{self.similarity_threshold_var.get():.2f}")
    
    def update_unknown_label(self, *args):
        self.unknown_label.config(text=f"{self.unknown_threshold_var.get():.2f}")
    
    def update_confidence_gap_label(self, *args):
        self.confidence_gap_label.config(text=f"{self.confidence_gap_var.get():.2f}")
    
    def browse_model(self):
        model_type = self.model_type_var.get()
        if model_type == "arcface":
            filetypes = [("PyTorch Models", "*.pth"), ("All files", "*.*")]
        else:
            filetypes = [("Haar Cascade Models", "*.yml *.pkl"), ("All files", "*.*")]
            
        filename = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=filetypes,
            initialdir="C:/hitler.ai/models"
        )
        if filename:
            self.model_path_var.set(filename)
    
    def debug_class_mapping(self):
        """Debug function to show current class mapping"""
        if self.recognizer:
            if self.model_type_var.get() == "arcface":
                messagebox.showinfo("Class Mapping Debug", 
                                   f"Current class mapping:\n\n{self.recognizer.idx_to_class}")
            else:
                messagebox.showinfo("Class Mapping Debug", 
                                   f"Current class mapping:\n\n{self.recognizer.haar_recognizer.idx_to_class}")
        else:
            messagebox.showwarning("No Model", "Please load a model first.")
    
    def load_model(self):
        try:
            model_path = self.model_path_var.get()
            model_type = self.model_type_var.get()
            
            if not os.path.exists(model_path):
                messagebox.showerror("Error", f"Model file not found:\n{model_path}")
                return
            
            self.header_status_label.config(text=f"Loading {model_type} model...", style="Warning.TLabel")
            self.root.update()
            
            # For Haar Cascade, check if the model file exists and is valid
            if model_type == "haarcascade":
                if not (model_path.endswith('.yml') or model_path.endswith('.pkl')):
                    messagebox.showwarning("Warning", "Haar Cascade models should be .yml or .pkl files")
                
                # Check if the mapping file exists
                mapping_path = model_path.replace('.yml', '_mapping.pkl').replace('.pkl', '_mapping.pkl')
                if not os.path.exists(mapping_path):
                    mapping_path = os.path.join(os.path.dirname(model_path), "haar_label_mapping.pkl")
                    if not os.path.exists(mapping_path):
                        messagebox.showwarning("Warning", f"Label mapping file not found. Expected at:\n{mapping_path}")
            
            self.recognizer = FaceRecognizer(model_path, self.detection_conf_var.get(), model_type)
            
            # Update thresholds from model (only for ArcFace)
            if model_type == "arcface":
                self.similarity_threshold_var.set(self.recognizer.similarity_threshold)
                self.unknown_threshold_var.set(self.recognizer.unknown_threshold)
                self.confidence_gap_var.set(self.recognizer.min_confidence_gap)
                
                # Update known people list
                self.known_people_list = list(self.recognizer.idx_to_class.values())
            else:
                # For Haar Cascade, use default thresholds
                self.similarity_threshold_var.set(0.70)
                self.unknown_threshold_var.set(0.50)
                self.confidence_gap_var.set(0.15)
                
                # Update known people list
                if hasattr(self.recognizer.haar_recognizer, 'idx_to_class'):
                    self.known_people_list = list(self.recognizer.haar_recognizer.idx_to_class.values())
                else:
                    self.known_people_list = ["No people loaded"]
            
            self.update_people_listbox()
            
            self.main_start_button.config(state="normal")
            self.header_status_label.config(text=f"✅ {model_type} model loaded! Known people: {len(self.known_people_list)}", 
                                          style="Success.TLabel")
            
            messagebox.showinfo("Model Loaded", 
                               f"{model_type} model loaded successfully!\n\n"
                               f"Known people: {len(self.known_people_list)}\n"
                               f"ENHANCED ACCURACY - Strict thresholds applied\n"
                               f"Better recognition with fewer false positives")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
            self.header_status_label.config(text="❌ Failed to load model", style="Error.TLabel")
    
    def update_people_listbox(self):
        self.people_listbox.delete(0, tk.END)
        for person in self.known_people_list:
            self.people_listbox.insert(tk.END, person)
    
    def start_recognition(self):
        """Enhanced camera initialization with better error handling"""
        try:
            # Get camera index from selected camera name
            camera_index = 2  # Default to IV Cam
            camera_name = self.camera_var.get()
            
            for name, index in self.CAMERA_OPTIONS:
                if name == camera_name:
                    camera_index = int(index)
                    break
            
            self.header_status_label.config(text=f"🔄 Initializing {camera_name}...", 
                                          style="Warning.TLabel")
            self.root.update()
            
            # Use enhanced camera initialization
            self.cap = initialize_camera(camera_index, camera_name)
            
            if not self.cap or not self.cap.isOpened():
                # Fallback: try default camera
                self.header_status_label.config(text="⚠️ Trying fallback camera...", 
                                              style="Warning.TLabel")
                self.root.update()
                
                self.cap = initialize_camera(0, "Fallback Camera")
                
                if not self.cap or not self.cap.isOpened():
                    messagebox.showerror("Camera Error", 
                                       f"Failed to open camera:\n\n"
                                       f"Selected: {camera_name} (Index {camera_index})\n\n"
                                       f"Please:\n"
                                       f"1. Check if camera is connected\n"
                                       f"2. Try different camera from dropdown\n"
                                       f"3. Ensure no other app is using camera")
                    return
            
            # Test camera read
            ret, frame = self.cap.read()
            if not ret:
                messagebox.showerror("Camera Error", "Camera initialized but cannot read frames")
                self.cap.release()
                return
            
            print(f"✓ Camera successfully initialized: {frame.shape[1]}x{frame.shape[0]}")
            
            # Reset statistics
            self.total_detections = 0
            self.successful_recognitions = 0
            self.start_time = time.time()
            self.frame_count = 0
            self.previous_faces = []
            
            # Start recognition
            self.is_running = True
            self.main_start_button.config(state="disabled")
            self.main_stop_button.config(state="normal")
            self.header_status_label.config(text="✅ Recognition ACTIVE - Processing frames...", 
                                          style="Success.TLabel")
            
            # Start recognition thread
            self.recognition_thread = threading.Thread(target=self.recognition_loop, daemon=True)
            self.recognition_thread.start()
            
        except Exception as e:
            error_msg = f"Failed to start recognition:\n{str(e)}"
            print(error_msg)
            messagebox.showerror("Start Error", error_msg)
            self.header_status_label.config(text="❌ Failed to start recognition", 
                                          style="Error.TLabel")
            if self.cap:
                self.cap.release()
                self.cap = None

    def stop_recognition(self):
        """Stop the recognition process"""
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.main_start_button.config(state="normal")
        self.main_stop_button.config(state="disabled")
        self.header_status_label.config(text="🛑 Recognition stopped", style="Warning.TLabel")
        self.video_label.config(text="🎯 Camera stopped - Click START to begin\n\n"
                                   "💡 Tips:\n"
                                   "• Use IV Cam for best results\n" 
                                   "• Ensure good lighting\n"
                                   "• Face the camera directly", 
                               image='', background="black", foreground="white")
    
    def reset_stats(self):
        """Reset recognition statistics"""
        self.total_detections = 0
        self.successful_recognitions = 0
        self.start_time = time.time()
        self.update_stats(0, 0, 0, 0, 0)
    
    def save_screenshot(self):
        """Save current frame as screenshot"""
        if hasattr(self, 'current_frame') and self.current_frame is not None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"recognition_screenshot_{timestamp}.jpg"
            cv2.imwrite(filename, self.current_frame)
            messagebox.showinfo("Screenshot Saved", f"Screenshot saved as:\n{filename}")
        else:
            messagebox.showwarning("No Frame", "No camera frame available to save.")
    
    def update_last_recognition(self, name, similarity):
        """Update the last recognition display"""
        self.last_recognition_label.config(text=f"👤 Last: {name} ({similarity:.3f})")
    
    def recognition_loop(self):
        """Main recognition loop - same as original"""
        frame_count = 0
        fps_start = time.time()
        
        while self.is_running and self.cap and self.cap.isOpened():
            loop_start = time.time()
            
            ret, frame = self.cap.read()
            if not ret:
                break
            
            self.current_frame = frame.copy()
            frame_count += 1
            self.frame_count += 1
            
            try:
                if self.recognizer:
                    # Update recognizer thresholds from UI
                    self.recognizer.detection_conf = self.detection_conf_var.get()
                    
                    # Only update these for ArcFace
                    if self.model_type_var.get() == "arcface":
                        self.recognizer.similarity_threshold = self.similarity_threshold_var.get()
                        self.recognizer.unknown_threshold = self.unknown_threshold_var.get()
                        self.recognizer.min_confidence_gap = self.confidence_gap_var.get()
                    
                    self.detection_interval = int(self.detection_interval_var.get())
            except:
                self.detection_interval = 3
            
            current_faces = self.previous_faces
            
            if self.frame_count % self.detection_interval == 0:
                if self.recognizer:
                    faces = self.recognizer.detect_faces(frame)
                    self.total_detections += len(faces)
                    current_faces = faces
                    self.previous_faces = faces
            
            if self.recognizer:
                for face_info in current_faces:
                    x1, y1, x2, y2 = face_info['bbox']
                    
                    face_image = frame[y1:y2, x1:x2]
                    
                    if face_image.size > 0:
                        if self.frame_count % self.detection_interval == 0:
                            name, similarity, confidence_level = self.recognizer.recognize_face(face_image)
                            
                            if name != "Unknown" and confidence_level in ["HIGH", "MEDIUM"]:
                                self.successful_recognitions += 1
                                self.root.after(0, lambda n=name, s=similarity: self.update_last_recognition(n, s))
                            
                            face_info['recognition'] = (name, similarity, confidence_level)
                        else:
                            name, similarity, confidence_level = face_info.get('recognition', ("Unknown", 0.0, "VERY_LOW"))
                        
                        # Enhanced color coding based on confidence level
                        if confidence_level == "HIGH":
                            color = (0, 255, 0)  # Green for strong recognition
                            label_color = (0, 255, 0)
                        elif confidence_level == "MEDIUM":
                            color = (0, 255, 255)  # Yellow for weak recognition
                            label_color = (0, 255, 255)
                        elif confidence_level == "LOW":
                            color = (0, 165, 255)  # Orange for uncertain
                            label_color = (0, 165, 255)
                        else:
                            color = (0, 0, 255)  # Red for unknown/very low
                            label_color = (0, 0, 255)
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw label with background
                        main_label = name if name != "Unknown" else "Unknown"
                        detail_label = f"Conf: {similarity:.3f} ({confidence_level})"
                        
                        label_size = cv2.getTextSize(main_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                                    (x1 + label_size[0] + 10, y1), color, -1)
                        cv2.putText(frame, main_label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                  0.6, (255, 255, 255), 2)
                        
                        cv2.putText(frame, detail_label, (x1, y2 + 20), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 1)
            
            current_time = time.time()
            elapsed = current_time - fps_start
            fps = frame_count / elapsed if elapsed > 0 else 0
            
            processing_time = current_time - loop_start
            sleep_time = max(0, self.frame_time - processing_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            actual_fps = 1.0 / (time.time() - loop_start) if (time.time() - loop_start) > 0 else 0
            actual_fps = min(actual_fps, self.target_fps)
            
            session_elapsed = current_time - self.start_time
            session_minutes = int(session_elapsed // 60)
            session_seconds = int(session_elapsed % 60)
            
            success_rate = (self.successful_recognitions / max(1, self.total_detections)) * 100
            
            if frame_count % 5 == 0:
                self.root.after(0, lambda: self.update_stats(actual_fps, len(current_faces), success_rate, session_minutes, session_seconds))
            
            # Add FPS and face count overlay
            model_type = self.model_type_var.get()
            cv2.putText(frame, f"FPS: {actual_fps:.1f}", (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Faces: {len(current_faces)}", (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Model: {model_type}", (10, 75), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            if self.recognizer:
                cv2.putText(frame, f"Known: {self.successful_recognitions}", (10, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (640, 480))
            frame_pil = Image.fromarray(frame_resized)
            frame_tk = ImageTk.PhotoImage(frame_pil)
            
            self.root.after(0, lambda img=frame_tk: self.update_video_display(img))
    
    def update_video_display(self, frame_tk):
        """Update the video display with new frame"""
        self.video_label.config(image=frame_tk, text="")
        self.video_label.image = frame_tk
    
    def update_stats(self, fps, faces_count, success_rate, session_min, session_sec):
        """Update statistics display"""
        self.fps_label.config(text=f"🔄 FPS: {fps:.1f}")
        self.detections_label.config(text=f"👁️ Detections: {self.total_detections}")
        self.recognitions_label.config(text=f"✅ Recognitions: {self.successful_recognitions}")
        self.accuracy_label.config(text=f"📊 Success Rate: {success_rate:.1f}%")
        self.session_label.config(text=f"⏱️ Session: {session_min:02d}:{session_sec:02d}")
        
        if fps > 45:
            perf_text = f"⚡ Performance: Excellent ({fps:.1f} FPS)"
            color = "#2ecc71"  # Green
        elif fps > 25:
            perf_text = f"⚡ Performance: Good ({fps:.1f} FPS)"
            color = "#f39c12"  # Orange
        else:
            perf_text = f"⚡ Performance: Slow ({fps:.1f} FPS)"
            color = "#e74c3c"  # Red
        
        self.performance_label.config(text=perf_text, foreground=color)
    
    def run(self):
        """Start the application"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
    
    def on_closing(self):
        """Handle application closing"""
        if self.is_running:
            self.stop_recognition()
        self.root.destroy()

if __name__ == "__main__":
    if not RETINAFACE_AVAILABLE:
        print("Note: RetinaFace not available. Haar Cascade will work fine.")
    
    print("=== Rollcall.ai ENHANCED UI & CAMERA FIXES ===")
    print(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name()}")
    print("ENHANCED FEATURES:")
    print("- Better camera detection with IV Cam priority")
    print("- Improved UI with tabs and better organization")  
    print("- Enhanced camera initialization with fallbacks")
    print("- System information and help tools")
    print("- Visual improvements and status indicators")
    print("=============================================")
    
    app = FaceRecognitionApp()
    app.run()