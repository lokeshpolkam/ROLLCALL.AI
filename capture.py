
import cv2
import os
import time
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
from datetime import datetime
from PIL import Image, ImageTk
import threading
import json
import mss
import torch
import torch.backends.cudnn as cudnn
import re  # Added for filename sanitization

# NEW: MTCNN imports for multi-angle face detection
try:
    from facenet_pytorch import MTCNN
    MTCNN_AVAILABLE = True
    print("MTCNN available - Multi-angle detection enabled!")
except ImportError:
    print("MTCNN not available. Install with: pip install facenet-pytorch")
    MTCNN_AVAILABLE = False

# === NATURAL QUALITY Constants ===
DEFAULT_SAVE_DIR = "C:/hitler.ai/captured_faces"
FRAME_WIDTH = 2560
FRAME_HEIGHT = 1440
DELAY_BETWEEN_CAPTURES = 0.4
CONFIDENCE_THRESHOLD = 0.88
MIN_FACE_SIZE = 200
MAX_FACE_SIZE = 1500

# Camera options
CAMERA_OPTIONS = [
    ("PC Web Camera", "0"),
    ("External USB Camera", "1"),
    ("IV Cam System", "2"),
    ("Secondary Camera", "3"),
    ("Tertiary Camera", "4")
]

# === Globals ===
student_info = {"name": "", "prn": "", "sem": "", "sec": ""}
current_variation_index = 0
is_capturing = False
camera_index = 0
input_source = "camera"
video_file_path = ""
save_directory = DEFAULT_SAVE_DIR
detector = None
mtcnn_detector = None
app_running = True
cap = None
sct = None

# ESSENTIAL VARIATIONS
variations = [
    {"name": "Neutral Front Face", "images": 35, "captured": 0, "completed": False},
    {"name": "Left Profile (45 degrees)", "images": 30, "captured": 0, "completed": False},
    {"name": "Right Profile (45 degrees)", "images": 30, "captured": 0, "completed": False},
    {"name": "Left Semi-Profile (25 degrees)", "images": 25, "captured": 0, "completed": False},
    {"name": "Right Semi-Profile (25 degrees)", "images": 25, "captured": 0, "completed": False},
    {"name": "Natural Smile", "images": 30, "captured": 0, "completed": False},
    {"name": "Tilted Head Left", "images": 20, "captured": 0, "completed": False},
    {"name": "Tilted Head Right", "images": 20, "captured": 0, "completed": False},
    {"name": "Looking Up (30 degrees)", "images": 20, "captured": 0, "completed": False},
    {"name": "Looking Down (30 degrees)", "images": 20, "captured": 0, "completed": False},
    {"name": "Eyes Closed", "images": 15, "captured": 0, "completed": False},
    {"name": "With Glasses", "images": 25, "captured": 0, "completed": False},
    {"name": "Different Lighting", "images": 25, "captured": 0, "completed": False}
]

# NEW: Filename sanitization function
def sanitize_filename(name):
    """Remove or replace invalid characters for folder names"""
    if not name:
        return "Unknown"
    # Replace invalid characters with underscore
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', name)
    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip().strip('.')
    # Limit length and ensure not empty
    if not sanitized:
        sanitized = "Unknown"
    return sanitized[:50]  # Limit length

# PROFESSIONAL MULTI-ANGLE FACE DETECTOR using MTCNN + OpenCV fallback
def load_face_detector():
    global detector, mtcnn_detector
    
    # Initialize MTCNN for multi-angle detection
    if MTCNN_AVAILABLE:
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            mtcnn_detector = MTCNN(
                image_size=512,
                margin=40,
                min_face_size=80,
                thresholds=[0.6, 0.7, 0.8],
                factor=0.709,
                post_process=True,
                device=device,
                keep_all=True,
                selection_method='probability'
            )
            print(f"MTCNN Multi-Angle Face Detector loaded successfully on {device}")
            return True
        except Exception as e:
            print(f"Error loading MTCNN: {e}")
            mtcnn_detector = None
    
    # Fallback to OpenCV detectors
    try:
        detector = {
            'frontal': cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'),
            'profile': cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml'),
            'alt': cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml'),
            'alt2': cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
        }
        print("Multiple OpenCV Cascade Classifiers loaded for different angles")
        return True
    except Exception as e:
        print(f"Error loading face detectors: {e}")
        messagebox.showerror("Error", f"Failed to load face detectors: {e}")
        return False

# Screen capture function
def capture_screen():
    global sct
    try:
        if sct is None:
            sct = mss.mss()
        
        monitor = sct.monitors[1]
        sct_img = sct.grab(monitor)
        img = np.array(sct_img)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return True, img
    except Exception as e:
        print(f"Screen capture error: {e}")
        return False, None

# PROFESSIONAL MULTI-ANGLE FACE DETECTION
def detect_face_multi_angle(frame):
    global mtcnn_detector, detector
    
    # Method 1: Try MTCNN first (best for multi-angle)
    if mtcnn_detector is not None:
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            boxes, probs, landmarks = mtcnn_detector.detect(pil_image, landmarks=True)
            
            if boxes is not None and len(boxes) > 0:
                best_idx = np.argmax(probs)
                if probs[best_idx] > CONFIDENCE_THRESHOLD:
                    box = boxes[best_idx]
                    confidence = probs[best_idx]
                    
                    x1, y1, x2, y2 = [int(coord) for coord in box]
                    
                    h, w = frame.shape[:2]
                    x1 = max(0, min(x1, w-1))
                    y1 = max(0, min(y1, h-1))
                    x2 = max(0, min(x2, w-1))
                    y2 = max(0, min(y2, h-1))
                    
                    face_width = x2 - x1
                    face_height = y2 - y1
                    
                    if face_width >= MIN_FACE_SIZE and face_height >= MIN_FACE_SIZE:
                        print(f"MTCNN detected face: {face_width}x{face_height}, confidence: {confidence:.3f}")
                        return (x1, y1, x2, y2), confidence, landmarks[best_idx] if landmarks is not None else None
            
        except Exception as e:
            print(f"MTCNN detection error: {e}")
    
    # Method 2: Fallback to multiple OpenCV cascades
    if detector is not None and isinstance(detector, dict):
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            for cascade_name, cascade in detector.items():
                if cascade.empty():
                    continue
                
                if cascade_name == 'profile':
                    scale_factor = 1.1
                    min_neighbors = 3
                    min_size = (MIN_FACE_SIZE//2, MIN_FACE_SIZE//2)
                else:
                    scale_factor = 1.05
                    min_neighbors = 5
                    min_size = (MIN_FACE_SIZE, MIN_FACE_SIZE)
                
                faces = cascade.detectMultiScale(
                    gray,
                    scaleFactor=scale_factor,
                    minNeighbors=min_neighbors,
                    minSize=min_size,
                    maxSize=(MAX_FACE_SIZE, MAX_FACE_SIZE),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                
                if len(faces) > 0:
                    largest_face = max(faces, key=lambda f: f[2] * f[3])
                    x, y, w, h = largest_face
                    
                    confidence = 0.8 if cascade_name == 'frontal' else 0.7
                    print(f"OpenCV {cascade_name} detected face: {w}x{h}")
                    return (x, y, x + w, y + h), confidence, None
                    
        except Exception as e:
            print(f"OpenCV cascade detection error: {e}")
    
    return None, 0, None

# Student Info Form
class StudentInfoForm:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("NATURAL QUALITY Face Capture System")
        self.root.geometry("800x650")
        self.root.resizable(False, False)
        self.root.configure(bg="#008080")
        self.setup_ui()
    
    def setup_ui(self):
        style = ttk.Style()
        style.configure("TFrame", background="#008080")
        style.configure("TLabel", background="#008080", foreground="white")
        style.configure("TButton", background="#20B2AA", foreground="white")
        style.configure("Accent.TButton", font=("Arial", 12, "bold"), background="#20B2AA", foreground="white")
        style.map("Accent.TButton", background=[("active", "#48D1CC")])
        
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        header_frame = ttk.Frame(main_frame)
        header_frame.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        title_label = ttk.Label(header_frame, text="NATURAL QUALITY Dataset Capture",
                               font=("Arial", 18, "bold"), foreground="white")
        title_label.pack(pady=(10, 5))
        
        subtitle_label = ttk.Label(header_frame, text="Original Colors • No Artificial Enhancements • Zero Artifacts",
                                  font=("Arial", 10), foreground="#E0FFFF")
        subtitle_label.pack()
        
        field_frame = ttk.Frame(main_frame)
        field_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=10)
        
        ttk.Label(field_frame, text="Student Name:", font=("Arial", 11, "bold")).grid(row=0, column=0, sticky=tk.W, pady=8)
        self.name_entry = ttk.Entry(field_frame, width=30, font=("Arial", 10))
        self.name_entry.grid(row=0, column=1, pady=8, padx=10, sticky="ew")
        self.name_entry.focus()
        
        ttk.Label(field_frame, text="PRN:", font=("Arial", 11, "bold")).grid(row=1, column=0, sticky=tk.W, pady=8)
        self.prn_entry = ttk.Entry(field_frame, width=30, font=("Arial", 10))
        self.prn_entry.grid(row=1, column=1, pady=8, padx=10, sticky="ew")
        
        ttk.Label(field_frame, text="Semester:", font=("Arial", 11, "bold")).grid(row=2, column=0, sticky=tk.W, pady=8)
        self.sem_entry = ttk.Entry(field_frame, width=30, font=("Arial", 10))
        self.sem_entry.grid(row=2, column=1, pady=8, padx=10, sticky="ew")
        
        ttk.Label(field_frame, text="Section:", font=("Arial", 11, "bold")).grid(row=3, column=0, sticky=tk.W, pady=8)
        self.sec_entry = ttk.Entry(field_frame, width=30, font=("Arial", 10))
        self.sec_entry.grid(row=3, column=1, pady=8, padx=10, sticky="ew")
        
        ttk.Label(field_frame, text="Save Directory:", font=("Arial", 11, "bold")).grid(row=4, column=0, sticky=tk.W, pady=8)
        dir_frame = ttk.Frame(field_frame)
        dir_frame.grid(row=4, column=1, sticky="ew", pady=8, padx=10)
        
        global save_directory
        self.dir_var = tk.StringVar(value=save_directory)
        self.dir_entry = ttk.Entry(dir_frame, textvariable=self.dir_var, width=25, font=("Arial", 10))
        self.dir_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(dir_frame, text="Browse", command=self.browse_directory, width=8).pack(side=tk.RIGHT, padx=(5, 0))
        
        ttk.Label(field_frame, text="Input Source:", font=("Arial", 11, "bold")).grid(row=5, column=0, sticky=tk.W, pady=8)
        source_frame = ttk.Frame(field_frame)
        source_frame.grid(row=5, column=1, sticky=tk.W, pady=8, padx=10)
        
        self.source_var = tk.StringVar(value="camera")
        ttk.Radiobutton(source_frame, text="Camera", variable=self.source_var, value="camera").pack(side=tk.LEFT)
        ttk.Radiobutton(source_frame, text="Screen", variable=self.source_var, value="screen").pack(side=tk.LEFT, padx=(20, 0))
        ttk.Radiobutton(source_frame, text="Video File", variable=self.source_var, value="video").pack(side=tk.LEFT, padx=(20, 0))
        
        self.video_frame = ttk.Frame(field_frame)
        self.video_frame.grid(row=6, column=1, sticky="ew", pady=8, padx=10)
        self.video_frame.grid_remove()
        
        self.video_path_var = tk.StringVar()
        ttk.Entry(self.video_frame, textvariable=self.video_path_var, width=25).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(self.video_frame, text="Browse", command=self.browse_video, width=8).pack(side=tk.RIGHT, padx=(5, 0))
        
        ttk.Label(field_frame, text="Camera Selection:", font=("Arial", 11, "bold")).grid(row=7, column=0, sticky=tk.W, pady=8)
        camera_frame = ttk.Frame(field_frame)
        camera_frame.grid(row=7, column=1, sticky=tk.W, pady=8, padx=10)
        
        self.camera_var = tk.StringVar(value="0")
        camera_dropdown = ttk.Combobox(camera_frame, textvariable=self.camera_var,
                                      values=[name for name, value in CAMERA_OPTIONS],
                                      state="readonly", width=20)
        camera_dropdown.pack(side=tk.LEFT)
        camera_dropdown.set("IV Cam System")
        
        self.source_var.trace('w', self.on_source_change)
        
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=20)
        
        submit_btn = ttk.Button(button_frame, text="Start NATURAL QUALITY Capture",
                               command=self.on_submit, style="Accent.TButton")
        submit_btn.pack(pady=10)
        
        self.root.bind('<Return>', lambda event: self.on_submit())
        
        field_frame.columnconfigure(1, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
    
    def browse_directory(self):
        global save_directory
        directory = filedialog.askdirectory(initialdir=save_directory)
        if directory:
            self.dir_var.set(directory)
    
    def browse_video(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("All files", "*.*")]
        )
        if file_path:
            self.video_path_var.set(file_path)
    
    def on_source_change(self, *args):
        if self.source_var.get() == "video":
            self.video_frame.grid()
        else:
            self.video_frame.grid_remove()
    
    def on_submit(self):
        global student_info, camera_index, save_directory, input_source, video_file_path
        
        student_info["name"] = self.name_entry.get().strip()
        student_info["prn"] = self.prn_entry.get().strip()
        student_info["sem"] = self.sem_entry.get().strip()
        student_info["sec"] = self.sec_entry.get().strip()
        save_directory = self.dir_var.get().strip()
        input_source = self.source_var.get()
        video_file_path = self.video_path_var.get().strip()
        
        if not student_info["name"] or not student_info["prn"]:
            messagebox.showwarning("Missing Info", "Please enter both Name and PRN")
            return
        
        if not save_directory:
            messagebox.showwarning("Missing Info", "Please select a save directory")
            return
        
        if input_source == "video" and not video_file_path:
            messagebox.showwarning("Missing Info", "Please select a video file")
            return
        
        camera_name = self.camera_var.get()
        for name, index in CAMERA_OPTIONS:
            if name == camera_name:
                camera_index = int(index)
                break
        else:
            camera_index = 2
        
        os.makedirs(save_directory, exist_ok=True)
        self.root.destroy()
    
    def run(self):
        self.root.mainloop()

# Initialize capture source
def init_capture_source():
    global cap, input_source, video_file_path, camera_index, sct
    
    print(f"Attempting to initialize camera with index: {camera_index}")
    print(f"Input source: {input_source}")
    
    if input_source == "screen":
        try:
            import mss
            global sct
            if sct is None:
                sct = mss.mss()
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize screen capture: {e}")
            return False
    
    elif input_source == "video":
        if not os.path.exists(video_file_path):
            messagebox.showerror("Error", f"Video file not found: {video_file_path}")
            return False
        
        cap = cv2.VideoCapture(video_file_path)
        if not cap.isOpened():
            messagebox.showerror("Error", f"Failed to open video file: {video_file_path}")
            return False
    
    else:
        camera_indices_to_try = []
        
        if camera_index == 2: # IV Cam System
            camera_indices_to_try = [2, 1, 0, 3, 4]
        elif camera_index == 0: # PC Web Camera
            camera_indices_to_try = [0, 1, 2]
        elif camera_index == 1: # External USB Camera
            camera_indices_to_try = [1, 0, 2]
        else:
            camera_indices_to_try = [camera_index, 0, 1, 2]
        
        cap = None
        successful_index = None
        
        for idx in camera_indices_to_try:
            print(f"Trying camera index: {idx}")
            
            backends_to_try = [
                cv2.CAP_DSHOW,
                cv2.CAP_MSMF,
                cv2.CAP_ANY
            ]
            
            for backend in backends_to_try:
                try:
                    temp_cap = cv2.VideoCapture(idx, backend)
                    
                    if temp_cap.isOpened():
                        ret, frame = temp_cap.read()
                        if ret and frame is not None:
                            cap = temp_cap
                            successful_index = idx
                            print(f"Successfully opened camera {idx} with backend {backend}")
                            break
                        else:
                            temp_cap.release()
                    else:
                        temp_cap.release()
                except Exception as e:
                    print(f"Failed to open camera {idx} with backend {backend}: {e}")
                    continue
                
                if cap is not None:
                    break
            
            if cap is not None:
                break
        
        if cap is None or not cap.isOpened():
            error_msg = """ERROR: Could not open any camera!

Troubleshooting steps:
1. Make sure iVCam app is running on your phone
2. Check if any other application is using the camera
3. Try different camera indices (0, 1, 2, 3, 4)
4. Restart both iVCam app and this script
5. Make sure iVCam is connected via WiFi/USB"""
            
            messagebox.showerror("Camera Error", error_msg)
            print(error_msg)
            return False
        
        print(f"Camera opened successfully at index: {successful_index}")
        
        # NATURAL QUALITY camera settings - NO aggressive enhancements
        try:
            if camera_index == 2: # iVCam
                print("Applying natural camera quality settings...")
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
                cap.set(cv2.CAP_PROP_FPS, 60)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                # Neutral settings - no artificial enhancements
                cap.set(cv2.CAP_PROP_BRIGHTNESS, 128)
                cap.set(cv2.CAP_PROP_CONTRAST, 128)
                cap.set(cv2.CAP_PROP_SATURATION, 128)
                cap.set(cv2.CAP_PROP_SHARPNESS, 128)
                
            else: # Regular webcam
                print("Applying natural webcam settings...")
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                cap.set(cv2.CAP_PROP_FPS, 30)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                cap.set(cv2.CAP_PROP_BRIGHTNESS, 128)
                cap.set(cv2.CAP_PROP_CONTRAST, 128)
                cap.set(cv2.CAP_PROP_SATURATION, 128)
                cap.set(cv2.CAP_PROP_SHARPNESS, 128)
            
        except Exception as e:
            print(f"Warning: Could not set some camera properties: {e}")
        
        # Test frame capture
        print("Testing frame capture...")
        for i in range(5):
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"Frame test {i+1}/5: SUCCESS - {frame.shape}")
                break
            else:
                print(f"Frame test {i+1}/5: FAILED")
                time.sleep(0.1)
        else:
            error_msg = "ERROR: Cannot capture frames from camera. Try restarting iVCam app."
            messagebox.showerror("Frame Capture Error", error_msg)
            cap.release()
            return False
    
    return True

# Get frame function
def get_frame():
    global input_source, cap, sct
    
    if input_source == "screen":
        return capture_screen()
    else:
        try:
            ret, frame = False, None
            
            for attempt in range(3):
                ret, frame = cap.read()
                if ret and frame is not None:
                    break
                time.sleep(0.01)
            
            if not ret or frame is None:
                print("Failed to capture frame")
                return False, None
            
            return ret, frame
            
        except Exception as e:
            print(f"Frame capture error: {e}")
            return False, None

# NATURAL QUALITY face capture function
def capture_variation(variation, folder_name):
    global is_capturing, app_running, input_source
    
    # Sanitize variation name for folder creation
    safe_variation_name = sanitize_filename(variation["name"])
    var_path = os.path.join(folder_name, safe_variation_name)
    os.makedirs(var_path, exist_ok=True)
    
    if not init_capture_source():
        return False
    
    last_capture_time = 0
    is_capturing = False
    last_key_time = 0
    key_delay = 0.3
    
    cv2.namedWindow("NATURAL QUALITY Dataset Capture", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("NATURAL QUALITY Dataset Capture", 1280, 720)
    
    print(f"\nSTARTING NATURAL QUALITY CAPTURE FOR: {variation['name']}")
    print(f"Target: {variation['images']} images")
    print(f"Quality: Natural - No Artificial Enhancements")
    
    while variation["captured"] < variation["images"] and app_running:
        success, frame = get_frame()
        if not success:
            if input_source == "video":
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                print("Frame capture failed")
                break
        
        current_time = time.time()
        fps = int(1.0 / (current_time - last_capture_time + 1e-5)) if last_capture_time > 0 else 0
        
        # ORIGINAL frame for capture (no overlays)
        original_frame = frame.copy()
        
        # Use multi-angle detection
        face_rect, confidence, landmarks = detect_face_multi_angle(frame)
        
        # Get camera name for display
        camera_name = "Unknown"
        for name, index in CAMERA_OPTIONS:
            if int(index) == camera_index:
                camera_name = name
                break
        
        source_name = "Screen" if input_source == "screen" else f"Video: {os.path.basename(video_file_path)}" if input_source == "video" else f"{camera_name}"
        
        detector_info = "MTCNN Multi-Angle" if MTCNN_AVAILABLE and mtcnn_detector is not None else "OpenCV Multi-Cascade"
        
        # Clean info overlay ONLY ON DISPLAY FRAME
        info_text = [
            f"NATURAL QUALITY: {variation['name']}",
            f"Progress: {variation['captured']}/{variation['images']} images",
            f"Confidence: {confidence*100:.1f}%" if confidence > 0 else "[WARNING] No face detected",
            f"Detection: {detector_info}",
            f"Source: {source_name} @ {FRAME_WIDTH}x{FRAME_HEIGHT}",
            f"Status: {'[REC] CAPTURING' if is_capturing else '[STOP] PRESS SPACE TO START'}",
            f"FPS: {fps}",
            "",
            "CONTROLS: SPACE=Start/Stop | R=Reset | Q=Next variation"
        ]
        
        # Add text overlay ONLY to display frame
        y_offset = 30
        for text in info_text:
            if text:
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(frame, (5, y_offset - 25), (text_size[0] + 15, y_offset + 5), (0, 0, 0), -1)
                cv2.putText(frame, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_offset += 35
        
        if face_rect:
            startX, startY, endX, endY = face_rect
            
            # Enhanced bounding box ONLY ON DISPLAY
            box_color = (0, 255, 0) if confidence > CONFIDENCE_THRESHOLD else (0, 165, 255)
            thickness = 3
            
            # Main rectangle
            cv2.rectangle(frame, (startX, startY), (endX, endY), box_color, thickness)
            
            # Professional-style corners
            corner_length = 30
            corner_thickness = 5
            
            # Draw corners
            cv2.line(frame, (startX, startY), (startX + corner_length, startY), box_color, corner_thickness)
            cv2.line(frame, (startX, startY), (startX, startY + corner_length), box_color, corner_thickness)
            cv2.line(frame, (endX, startY), (endX - corner_length, startY), box_color, corner_thickness)
            cv2.line(frame, (endX, startY), (endX, startY + corner_length), box_color, corner_thickness)
            cv2.line(frame, (startX, endY), (startX + corner_length, endY), box_color, corner_thickness)
            cv2.line(frame, (startX, endY), (startX, endY - corner_length), box_color, corner_thickness)
            cv2.line(frame, (endX, endY), (endX - corner_length, endY), box_color, corner_thickness)
            cv2.line(frame, (endX, endY), (endX, endY - corner_length), box_color, corner_thickness)
            
            # Quality and detection info
            quality_text = f"CONFIDENCE: {confidence*100:.1f}%"
            detection_text = f"METHOD: {detector_info}"
            
            text_x = startX
            text_y = startY - 10 if startY - 10 > 40 else endY + 30
            
            # Background for text
            text_size = cv2.getTextSize(quality_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (text_x - 2, text_y - 40), (text_x + text_size[0] + 4, text_y + 5), (0, 0, 0), -1)
            cv2.putText(frame, quality_text, (text_x, text_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
            cv2.putText(frame, detection_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw landmarks if available (MTCNN)
            if landmarks is not None:
                for point in landmarks:
                    cv2.circle(frame, (int(point[0]), int(point[1])), 2, (255, 0, 0), -1)
            
            face_width = endX - startX
            face_height = endY - startY
            
            # NATURAL QUALITY capture logic
            if (is_capturing and 
                variation["captured"] < variation["images"] and
                current_time - last_capture_time > DELAY_BETWEEN_CAPTURES and
                confidence > CONFIDENCE_THRESHOLD and
                face_width >= MIN_FACE_SIZE):
                
                # Extract face from ORIGINAL frame (clean, no overlays)
                margin = int(min(face_width, face_height) * 0.15) # Natural margin
                crop_startX = max(0, startX - margin)
                crop_startY = max(0, startY - margin)
                crop_endX = min(original_frame.shape[1], endX + margin)
                crop_endY = min(original_frame.shape[0], endY + margin)
                
                face_img = original_frame[crop_startY:crop_endY, crop_startX:crop_endX]
                
                if face_img.size > 0:
                    print(f"Processing NATURAL QUALITY image...")
                    
                    # NATURAL QUALITY - Use original face image directly
                    natural_face = face_img
                    
                    # Ensure good resolution but don't force upscaling
                    h, w = natural_face.shape[:2]
                    min_dimension = min(h, w)
                    if min_dimension < 800: # Reasonable minimum
                        scale_factor = 800 / min_dimension
                        new_w = int(w * scale_factor)
                        new_h = int(h * scale_factor)
                        natural_face = cv2.resize(natural_face, (new_w, new_h),
                                                 interpolation=cv2.INTER_LANCZOS4)
                        print(f"Natural quality maintained at: {new_w}x{new_h}")
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    img_path = os.path.join(var_path, f"NATURAL_QUALITY_{timestamp}.png")
                    
                    # Save with natural quality - lossless PNG
                    save_params = [
                        cv2.IMWRITE_PNG_COMPRESSION, 1, # Light compression
                        cv2.IMWRITE_PNG_STRATEGY, cv2.IMWRITE_PNG_STRATEGY_DEFAULT
                    ]
                    
                    success = cv2.imwrite(img_path, natural_face, save_params)
                    
                    if success:
                        file_size = os.path.getsize(img_path) / (1024 * 1024) # MB
                        variation["captured"] += 1
                        last_capture_time = current_time
                        print(f"NATURAL QUALITY: Saved {variation['captured']}/{variation['images']} | Size: {file_size:.2f}MB | {img_path}")
                    else:
                        print(f"ERROR: Failed to save: {img_path}")
        
        # Display frame
        display_frame = cv2.resize(frame, (1280, 720))
        cv2.imshow("NATURAL QUALITY Dataset Capture", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        current_key_time = time.time()
        
        if current_key_time - last_key_time > key_delay:
            if key == ord(' '):
                is_capturing = not is_capturing
                status = "STARTED" if is_capturing else "STOPPED"
                print(f"NATURAL CAPTURE {status}")
                last_key_time = current_key_time
            
            elif key == ord('r'):
                print(f"Resetting variation: {variation['name']}")
                for file in os.listdir(var_path):
                    if file.startswith("NATURAL_QUALITY_"):
                        os.remove(os.path.join(var_path, file))
                variation["captured"] = 0
                is_capturing = False
                last_key_time = current_key_time
            
            elif key == ord('q'):
                break
    
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    
    variation["completed"] = variation["captured"] >= variation["images"]
    print(f"Natural quality variation completed: {variation['name']} | Captured: {variation['captured']}/{variation['images']}")
    
    return variation["completed"]

# Variation Selector
class VariationSelector:
    def __init__(self, folder_name):
        self.folder_name = folder_name
        self.root = tk.Tk()
        self.root.title("NATURAL QUALITY Dataset Capture - Variation Selection")
        self.root.geometry("1000x700")
        self.root.configure(bg="#008080")
        self.setup_ui()
    
    def setup_ui(self):
        style = ttk.Style()
        style.configure("TFrame", background="#008080")
        style.configure("TLabel", background="#008080", foreground="white")
        style.configure("TButton", background="#20B2AA", foreground="white")
        
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        title_label = ttk.Label(header_frame, text="NATURAL QUALITY Dataset Capture",
                               font=("Arial", 18, "bold"), foreground="white")
        title_label.pack(pady=(10, 5))
        
        info_frame = ttk.Frame(main_frame)
        info_frame.pack(fill=tk.X, pady=10)
        
        info_text = f"Student: {student_info['name']} | PRN: {student_info['prn']} | Semester: {student_info['sem']} | Section: {student_info['sec']}"
        info_label = ttk.Label(info_frame, text=info_text, font=("Arial", 12))
        info_label.pack()
        
        # Detection method display
        detection_method = "MTCNN Multi-Angle Detection" if MTCNN_AVAILABLE else "OpenCV Multi-Cascade Detection"
        method_label = ttk.Label(info_frame, text=f"Detection Method: {detection_method}",
                                font=("Arial", 10), foreground="#90EE90" if MTCNN_AVAILABLE else "#FFB6C1")
        method_label.pack()
        
        total_variations = len(variations)
        completed_variations = sum(1 for v in variations if v["completed"])
        progress_percent = (completed_variations / total_variations) * 100 if total_variations > 0 else 0
        
        progress_frame = ttk.Frame(main_frame)
        progress_frame.pack(fill=tk.X, pady=10)
        
        progress_text = f"Progress: {completed_variations}/{total_variations} variations ({progress_percent:.1f}%)"
        progress_label = ttk.Label(progress_frame, text=progress_text, font=("Arial", 11))
        progress_label.pack()
        
        progress_bar = ttk.Progressbar(progress_frame, orient="horizontal", length=400, mode="determinate")
        progress_bar.pack(pady=5)
        progress_bar["value"] = progress_percent
        
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        canvas = tk.Canvas(canvas_frame, bg="#008080", highlightthickness=0)
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        for i, variation in enumerate(variations):
            frame = ttk.Frame(scrollable_frame, relief="solid", borderwidth=1)
            frame.pack(fill=tk.X, pady=2, padx=5)
            
            var_progress = (variation["captured"] / variation["images"]) * 100 if variation["images"] > 0 else 0
            status = "Completed" if variation["completed"] else f"{variation['captured']}/{variation['images']} captured"
            
            btn_text = f"{i+1}. {variation['name']}"
            btn = ttk.Button(
                frame,
                text=btn_text,
                command=lambda idx=i: self.select_variation(idx),
                width=60
            )
            btn.pack(side=tk.LEFT, padx=(10, 20), pady=5)
            
            status_label = ttk.Label(frame, text=status,
                                   foreground="#90EE90" if variation["completed"] else "#E0FFFF")
            status_label.pack(side=tk.LEFT, padx=10, pady=5)
            
            var_progress_bar = ttk.Progressbar(frame, orient="horizontal", length=150, mode="determinate")
            var_progress_bar.pack(side=tk.RIGHT, padx=10, pady=5)
            var_progress_bar["value"] = var_progress
            
            if variation["completed"]:
                btn.state(["disabled"])
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=20)
        
        ttk.Button(button_frame, text="Export Progress", command=self.export_progress).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Load Progress", command=self.load_progress).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Exit", command=self.root.destroy).pack(side=tk.LEFT, padx=5)
    
    def select_variation(self, index):
        self.root.destroy()
        success = capture_variation(variations[index], self.folder_name)
        self.save_progress()
        if success and app_running:
            self.__init__(self.folder_name)
            self.run()
    
    def export_progress(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir=save_directory
        )
        
        if file_path:
            progress_data = {
                "student_info": student_info,
                "variations": variations,
                "timestamp": datetime.now().isoformat()
            }
            
            with open(file_path, 'w') as f:
                json.dump(progress_data, f, indent=4)
            
            messagebox.showinfo("Export Success", f"Progress exported to {file_path}")
    
    def load_progress(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir=save_directory
        )
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    progress_data = json.load(f)
                
                global student_info, variations
                student_info.update(progress_data["student_info"])
                variations = progress_data["variations"]
                
                messagebox.showinfo("Load Success", "Progress loaded successfully")
                self.root.destroy()
                self.__init__(self.folder_name)
                self.run()
            
            except Exception as e:
                messagebox.showerror("Load Error", f"Failed to load progress: {e}")
    
    def save_progress(self):
        progress_file = os.path.join(self.folder_name, "progress.json")
        progress_data = {
            "student_info": student_info,
            "variations": variations,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(progress_file, 'w') as f:
            json.dump(progress_data, f, indent=4)
    
    def run(self):
        self.root.mainloop()

def start_capture():
    global detector, save_directory, app_running
    
    if not load_face_detector():
        return
    
    # SANITIZE folder name to prevent invalid characters
    sanitized_prn = sanitize_filename(student_info["prn"])
    sanitized_name = sanitize_filename(student_info["name"])
    folder_name = f"{save_directory}/{sanitized_prn}_{sanitized_name}"
    
    try:
        os.makedirs(folder_name, exist_ok=True)
        print(f"Created folder: {folder_name}")
    except Exception as e:
        print(f"Error creating folder: {e}")
        # Fallback folder name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{save_directory}/Student_{timestamp}"
        os.makedirs(folder_name, exist_ok=True)
        print(f"Created fallback folder: {folder_name}")
    
    progress_file = os.path.join(folder_name, "progress.json")
    if os.path.exists(progress_file):
        if messagebox.askyesno("Load Progress", "Previous progress found. Load it?"):
            try:
                with open(progress_file, 'r') as f:
                    progress_data = json.load(f)
                global variations
                variations = progress_data["variations"]
            except:
                messagebox.showerror("Error", "Failed to load progress file")
    
    # Save student info
    with open(os.path.join(folder_name, "info.txt"), "w") as f:
        f.write(f"Name: {student_info['name']}\n")
        f.write(f"PRN: {student_info['prn']}\n")
        f.write(f"Semester: {student_info['sem']}\n")
        f.write(f"Section: {student_info['sec']}\n")
        f.write(f"Capture Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Variations: {len(variations)}\n")
        f.write(f"Quality: NATURAL QUALITY (No Artificial Enhancements)\n")
        f.write(f"Detection: {'MTCNN Multi-Angle' if MTCNN_AVAILABLE else 'OpenCV Multi-Cascade'}\n")
        f.write(f"Enhancement: None - Original Image Quality Preserved\n")
        f.write(f"Format: Lossless PNG\n")
        f.write(f"Features: Original Colors, Zero Artifacts, No Cloudiness\n")
    
    selector = VariationSelector(folder_name)
    selector.run()

# Main execution
if __name__ == "__main__":
    os.makedirs(DEFAULT_SAVE_DIR, exist_ok=True)
    
    def load_model_async():
        load_face_detector()
    
    threading.Thread(target=load_model_async, daemon=True).start()
    
    while app_running:
        form = StudentInfoForm()
        form.run()
        start_capture()
        
        if not messagebox.askyesno("Continue", "Capture another student?"):
            app_running = False
