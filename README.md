# ROLLCALL.AI

ROLLCALL.AI is a smart, real-time attendance system built using **MTCNN / Haar Cascade** for face detection, **ArcFace** for high-accuracy facial recognition, and **OpenCV** for live video processing. It captures datasets, trains models, and performs recognition with a complete UI-based workflow.

---

## 🚀 Features
- Student credential entry system
- Automated dataset image capture with variations
- Training UI (Haar Cascade + ArcFace) with auto-stop
- Real-time recognition UI with name + confidence
- Attendance logging (CSV / SQLite)
- High-accuracy embeddings (ArcFace 512D)
- Fast detection + clean GUI for each step

---

# 📸 UI Screens With Explanations

Below are all the UI screens included in the project.

---

## **1. Student Credentials Input UI**
This screen appears before dataset capture.  
You enter the student's **Name, Roll Number, and Course/Details**, and the system automatically creates a labeled folder for storing images.

![Student Credentials Input UI](https://github.com/lokeshpolkam/ROLLCALL.AI/blob/main/ui%20images/capture%20u1.png?raw=true)

---

## **2. Dataset Image Capture UI**
This UI automatically captures **100–200 images** with variations like tilt, angles, lighting changes, and expressions.  
It uses real-time face detection and shows capture progress.

![Dataset Image Capture UI](https://github.com/lokeshpolkam/ROLLCALL.AI/blob/main/ui%20images/capture%20u2.png?raw=true)

---

## **3. Training UI (Haar Cascade + ArcFace)**
This is the model training dashboard.  
It shows epoch progress, logs, dataset size, and includes **AUTO-STOP** feature when training stabilizes.

![Training UI](https://github.com/lokeshpolkam/ROLLCALL.AI/blob/main/ui%20images/trainer%20ui.png?raw=true)

---

## **4. Real-Time Recognition UI**
This UI performs live recognition using ArcFace embeddings + Haar Cascade.  
Shows bounding boxes, predicted name, confidence score, and logs attendance instantly.

![Recognition UI](https://github.com/lokeshpolkam/ROLLCALL.AI/blob/main/ui%20images/recog%20ui.png?raw=true)

---

# 📂 Project Structure

ROLLCALL.AI/
├── src/
│ ├── capture.py
│ ├── trainer.py
│ ├── recog.py
│ ├── utils.py
│ └── db.py
├── ui images/
├── models/
├── data/
│ ├── enrolled/
│ └── embeddings.pkl
├── requirements.txt
└── README.md


---

# ⚙️ How the System Works

### **1. Enrollment**
User enters student details → system opens camera → captures image dataset automatically.

### **2. Training**
Faces are aligned → ArcFace generates embeddings → embeddings stored → classifier/lookup database created.

### **3. Recognition**
Real-time face detection → embedding extraction → similarity matching → attendance logged.

### **4. Logging**
Each recognized student is logged with:
- Name  
- Roll  
- Timestamp  
- Confidence score  

---

# 🖥️ Quick Start

### **Install Requirements**
pip install -r requirements.txt
Capture Dataset
python src/capture.py

Train Model
python src/trainer.py

Start Recognition
python src/recog.py

📘 Technologies Used

Python

OpenCV

MTCNN / Haar Cascade

ArcFace (InsightFace)

Tkinter GUI

NumPy / Pandas

📄 License

MIT License

🙌 Credits

Developed by Lokesh Polkam
ROLLCALL.AI – 2025
