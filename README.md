# AI-Based Illegal Garbage Dumping Detection System

Real-time detection of illegal garbage dumping using AI-powered computer vision. Uses YOLOv8 for object detection and pose estimation with a 7-stage analysis pipeline for high accuracy.

## Features

- **Real-time detection** from webcam, mobile phone (IP Webcam), or CCTV/RTSP cameras
- **Video upload analysis** — upload recorded footage for batch processing
- **7-stage detection pipeline** — object detection, pose estimation, tracking, temporal analysis, proximity check, confidence accumulation, cooldown
- **Separate camera window** — live feed runs in a native OpenCV window, not in the browser
- **Streamlit dashboard** — incident management, stats, camera management, video upload
- **GPU accelerated** — CUDA support for NVIDIA RTX GPUs (auto-falls back to CPU)

## Tech Stack

- **Python 3.10+**
- **YOLOv8** (ultralytics) — object detection + pose estimation
- **OpenCV** — camera capture + frame processing
- **Streamlit** — web dashboard
- **SQLite** — incident database
- **PyTorch** — CUDA GPU inference

## Quick Start

### 1. Clone & Setup Virtual Environment

```bash
cd ai-garbage-detect
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 2. Install Dependencies

**Non-RTX laptop (CPU only):**
```bash
pip install -r requirements.txt
```

**RTX laptop (GPU accelerated):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 3. Run the Dashboard

```bash
streamlit run app.py
```

### 4. Add a Camera

1. Go to **Camera Management** in the sidebar
2. Add your camera (webcam, IP stream, or RTSP)
3. Go to **Live Monitor** and click **Start Live Monitor**
4. A separate OpenCV window will open with real-time detection

## Mobile Camera Setup

1. Install **IP Webcam** app on your Android phone
2. Open the app → Start Server
3. Note the URL (e.g., `http://192.168.1.5:8080`)
4. In the dashboard → Camera Management → Add Camera → type: `ip_stream`, URL: `http://YOUR_IP:8080/video`
5. Ensure phone and laptop are on the same WiFi network

## Detection Pipeline

```
Frame → [1] YOLOv8s object detection (person + waste) on CUDA
      → [2] YOLOv8s-pose estimation (throwing/dropping body pose)
      → [3] Object tracking (ByteTrack — persistent IDs)
      → [4] Temporal analysis (newly appeared waste vs pre-existing)
      → [5] Proximity + zone check
      → [6] Confidence accumulator + cooldown
      → [7] Confirmed incident → log + snapshot
```

## Project Structure

```
ai-garbage-detect/
├── app.py                 # Streamlit dashboard
├── live_monitor.py        # OpenCV live camera window
├── detector.py            # YOLOv8 detection + pose wrapper
├── tracker.py             # Object tracking + temporal analysis
├── dump_analyzer.py       # Dump event analysis engine
├── camera_manager.py      # Camera source management
├── video_processor.py     # Video file analysis
├── database.py            # SQLite database
├── config.py              # Configuration & thresholds
├── requirements.txt       # Python dependencies
└── data/                  # Database, uploads, evidence snapshots
```

## Configuration

Edit `config.py` to adjust:
- Detection confidence thresholds
- Proximity distance for person-waste association
- Frame accumulation count for confirming events
- Cooldown duration between incidents
- Pose analysis thresholds
- Video frame sampling rate
