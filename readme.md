# 🚀 Vision AI Annotator  
### Real-Time Computer Vision System (YOLOv8 + Web Interface)

---

## 📌 Overview

Vision AI Annotator is a real-time computer vision system based on YOLOv8, designed for object detection, video stream analysis, and automated annotation.

The project provides both local and browser-based processing, enabling flexible deployment for developers, researchers, and real-world monitoring scenarios.

---

## 🧠 Key Features

- ⚡ Real-time object detection (YOLOv8)  
- 🌐 Web-based interface (Flask + Bootstrap)  
- 📊 Live statistics and analytics  
- 💾 Automatic annotation saving (JSON)  
- 📸 Snapshot capture with detections  
- ⏸ Pause / resume processing  
- 🔄 Multi-camera support  
- 📱 Mobile device compatibility (WebRTC mode)  

---

## 🔬 AI Capabilities

- Object detection using custom or pre-trained YOLOv8 models  
- Real-time video stream processing  
- Confidence-based filtering  
- Frame-by-frame annotation tracking  
- Statistical aggregation (objects, FPS, activity)  

---

## 🧩 Architecture

Client (Browser / Camera)         ↓    Flask Server         ↓  YOLOv8 Detection Engine         ↓   JSON Storage + API

---

## 🧠 Real-World Use Cases

This system can be used for:

- 🏥 Patient monitoring and behavior analysis  
- 📹 Video surveillance and smart cameras  
- 📊 Dataset collection for machine learning  
- 🧪 Computer vision research and prototyping  
- 🤖 AI demonstration systems  

---

## ⚙️ Installation

### 1. Clone repository
bash git clone https://github.com/mikle125/system_monitoring.git cd system_monitoring 

### 2. Create virtual environment
bash python -m venv venv source venv/bin/activate   # Linux / macOS venv\\Scripts\\activate     # Windows 

### 3. Install dependencies
bash pip install -r requirements.txt 

---

## 🤖 Model Setup

- Place your trained model (best.pt) in the root directory  
- OR use default YOLOv8 model (auto-download)

---

## ▶️ Running the Project

### 🔹 Script 1 — Local Camera
bash python script1.py 

Controls:
- Space — pause / resume  
- S — save annotations  
- C — switch camera  
- Q / ESC — exit  

---

### 🔹 Script 2 — Browser Camera (WebRTC)
bash python script2.py 

Open in browser:
http://localhost:3000

---

## 📊 Features Comparison

| Feature | Script 1 | Script 2 |
|--------|--------|--------|
| Local camera | ✅ | ❌ |
| WebRTC | ❌ | ✅ |
| Multi-user | ❌ | ✅ |
| Mobile support | ⚠️ | ✅ |
| Latency | Low | Medium |

---

## 📁 Project Structure

system_monitoring/ ├── script1.py ├── script2.py ├── requirements.txt ├── README.md ├── best.pt ├── annotations.json └── screenshots/

---

## 🔌 API Endpoints

General:
- GET /api/stats
- POST /api/toggle_pause
- POST /api/save_session
- GET /api/download_annotations
- POST /api/take_snapshot

---

## 📈 Performance

Recommended:
- CPU: 4+ cores  
- RAM: 8+ GB  
- GPU: optional (CUDA supported)  
- Camera: 720p+  

---

## 🐞 Common Issues

- Camera not detected → check device index  
- Low FPS → reduce resolution  
- Model not loading → verify .pt file  
- UI not доступен → check port  

---

## 📜 License

MIT License — free to use and modify  

---

## 👤 Author

Mikhail  

GitHub: https://github.com/mikle125  

---

⭐ If you like this project — give it a star
