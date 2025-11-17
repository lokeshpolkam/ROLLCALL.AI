ROLLCALL.AI – Smart Attendance System (MTCNN + ArcFace + OpenCV)

ROLLCALL.AI is a real-time, contactless smart attendance system that uses MTCNN for face detection, ArcFace for high-accuracy face recognition, and OpenCV for live image processing. The system captures datasets, trains models, and performs recognition—fully automated and optimized for classroom and organizational attendance management.

🚀 Features

Real-time facial detection using MTCNN / Haar Cascade

High-precision recognition using ArcFace embeddings

Automatic dataset creation with pose and expression variations

Student credential input system for dataset organization

Training UI with auto-stop epoch logic

Recognition UI displaying predicted name & confidence

Attendance logging and automatic identification

Clean and scalable code structure

📸 User Interface Screens
1. Student Credentials Input UI

File: capture u1.png
This interface is used before dataset recording. Users enter student details such as Name, Roll Number, and other credentials. The application uses this information to organize and label dataset images correctly.

2. Dataset Image Capture UI

File: capture u2.png
This screen captures multiple images of the student with real-time face detection. The system automatically collects images with variations (angle, tilt, expression) for robust training. No manual clicking is needed—everything is automated.

3. Training Interface (Haar Cascade + ArcFace)

File: trainer ui.png
This UI trains the recognition model. Haar Cascade is used for face detection during training, while ArcFace embeddings ensure high-accuracy classification. Training progress, loss values, and auto-stop epoch logic are visibly displayed.

4. Real-Time Recognition UI

File: recog ui.png
This is the recognition interface. It uses ArcFace embeddings + Haar Cascade detection to identify faces in real time. The UI shows the camera stream, bounding boxes, predicted name, confidence score, and logs attendance instantly.

📂 Project Structure
ROLLCALL.AI/
│
├── src/
│   ├── capture.py        # Dataset capture module with UI
│   ├── trainer.py        # Training interface (Haar + ArcFace)
│   ├── recog.py          # Real-time recognition interface
│   ├── utils/            # Helper functions
│
├── ui images/            # All UI screenshots
├── models/               # ArcFace / Haar Cascade models
├── data/                 # Stored datasets and embeddings
├── requirements.txt
└── README.md

🛠️ How It Works
1. Detection (MTCNN / Haar Cascade)

Detects faces and extracts key points for alignment.

2. Embedding Generation (ArcFace)

Creates a 512-D facial embedding using a deep neural model.

3. Matching

Uses cosine similarity to match embeddings with the stored database.

4. Attendance Logging

Recognized students are instantly recorded in the attendance logs.

📥 Installation
git clone https://github.com/lokeshpolkam/ROLLCALL.AI
cd ROLLCALL.AI
pip install -r requirements.txt

▶️ Usage
1. Capture Dataset
python src/capture.py

2. Train Model
python src/trainer.py

3. Start Recognition
python src/recog.py

📘 Technologies Used

Python

OpenCV

MTCNN

ArcFace (InsightFace)

NumPy / SciPy

Tkinter (for UI)

📄 Purpose

ROLLCALL.AI aims to replace manual or biometric attendance with an AI-powered solution that is faster, contactless, more accurate, and easy to deploy.
