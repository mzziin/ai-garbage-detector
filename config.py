import os
import torch
# NOTE: Pose estimation has been removed — detection relies on
# person + waste proximity and temporal analysis only.

# ============================================================
# Paths
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
EVIDENCE_DIR = os.path.join(DATA_DIR, "evidence")
DB_PATH = os.path.join(DATA_DIR, "garbage_detect.db")

# Create directories if they don't exist
for d in [DATA_DIR, UPLOADS_DIR, EVIDENCE_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# Device & Model Selection
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Use the small model on GPU for better accuracy, nano on CPU for speed
# NOTE: yolov8s.pt is not bundled in the repo;
# ultralytics will auto-download it on first GPU run.
DETECTION_MODEL = "yolov8s.pt" if DEVICE == "cuda" else "yolov8n.pt"

# ============================================================
# Detection Settings
# ============================================================
# Minimum confidence for YOLOv8 object detection
DETECTION_CONFIDENCE = 0.4

# COCO class IDs considered as waste-candidate objects
# person=0, bottle=39, cup=41, handbag=26, backpack=24, umbrella=25, suitcase=28
PERSON_CLASS_ID = 0
WASTE_CLASS_IDS = [39, 41, 26, 24, 25, 28]

# ============================================================
# Dump Analyzer Settings
# ============================================================
# Maximum pixel distance between person and waste to consider "proximity"
PROXIMITY_THRESHOLD = 200

# Number of consecutive frames required to confirm a dump event
FRAME_ACCUMULATION_THRESHOLD = 5

# Cooldown in seconds after logging an incident (prevents duplicates)
COOLDOWN_SECONDS = 30

# ============================================================
# Confidence Score Weights (must sum to 1.0)
# ============================================================
WEIGHT_DETECTION = 0.40
WEIGHT_TEMPORAL = 0.35
WEIGHT_ACCUMULATION = 0.25
# ============================================================
# Temporal Analysis
# ============================================================
# Number of frames to keep in the background model history
BACKGROUND_HISTORY_FRAMES = 60

# Minimum frames a waste object must persist after person leaves
# to be considered "dumped" (vs. temporarily placed)
WASTE_PERSISTENCE_FRAMES = 15

# Maximum pixel movement allowed for waste to be considered "stationary"
STATIONARY_THRESHOLD = 20

# Number of frames to wait after person leaves before confirming a dump
PERSON_LEFT_FRAMES = 10

# ============================================================
# Video Processing
# ============================================================
# Process every Nth frame of uploaded videos
VIDEO_FRAME_SAMPLE_RATE = 5

# ============================================================
# Safe Zones (can be configured per camera via Streamlit)
# Format: list of dicts with "x1", "y1", "x2", "y2" (pixel coords)
# Events inside these zones are ignored (e.g., near dustbins)
# ============================================================
DEFAULT_SAFE_ZONES = []
