"""
config.py — Central Configuration for the Face Recognition Identification System
=================================================================================
Every tunable parameter lives here. No other module hard-codes these values.
To change a setting, edit THIS file only.
"""

from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
#  Project Root & Directory Paths
# ─────────────────────────────────────────────────────────────────────────────
ROOT_DIR       = Path(__file__).resolve().parent.parent   # face-recognition-system/
DATA_DIR       = ROOT_DIR / "data"
ENROLLED_DIR   = DATA_DIR / "enrolled"       # data/enrolled/<PersonName>/<images>
TEST_DIR       = DATA_DIR / "test"           # data/test/<PersonName|Unknown>/<images>
EMBEDDINGS_DIR = ROOT_DIR / "embeddings"
RESULTS_DIR    = ROOT_DIR / "results"
PLOTS_DIR      = RESULTS_DIR / "plots"

# ─────────────────────────────────────────────────────────────────────────────
#  Persisted Database & Output Files
# ─────────────────────────────────────────────────────────────────────────────
DATABASE_PATH  = EMBEDDINGS_DIR / "database.pkl"   # serialised embedding database
EVALUATION_CSV = RESULTS_DIR / "evaluation.csv"    # per-image evaluation rows

# ─────────────────────────────────────────────────────────────────────────────
#  InsightFace / ArcFace Model
# ─────────────────────────────────────────────────────────────────────────────
# "buffalo_l" bundles two ONNX models:
#   • det_10g.onnx       — RetinaFace-based face detector (5-point landmarks)
#   • w600k_r50.onnx     — ArcFace-R50 face recogniser  (512-d embeddings)
# Models are downloaded automatically on first run to ~/.insightface/models/
MODEL_NAME       = "buffalo_l"
MODEL_PROVIDERS  = ["CPUExecutionProvider"]   # change to "CUDAExecutionProvider" for GPU
DETECTION_SIZE   = (640, 640)                 # internal resize for the face detector
DET_SCORE_THRESH = 0.50                       # discard detections below this confidence

# ─────────────────────────────────────────────────────────────────────────────
#  Similarity / Matching Threshold
# ─────────────────────────────────────────────────────────────────────────────
# ArcFace embeddings are L2-normalised → cosine_similarity == dot-product.
#
# Typical ranges for buffalo_l on clean frontal photos:
#   Genuine pairs (same person, diff images) : 0.30 – 0.80
#   Impostor pairs (different persons)       : -0.20 – 0.30
#
# Default 0.35 is a conservative starting point derived from Equal Error Rate
# analysis across the LFW dataset (see results/plots/far_frr_curve.png after
# running `python main.py calibrate`).
# Run calibration on YOUR dataset to find the ideal value for your conditions.
SIMILARITY_THRESHOLD = 0.35

# ─────────────────────────────────────────────────────────────────────────────
#  Labels & Aggregation
# ─────────────────────────────────────────────────────────────────────────────
UNKNOWN_LABEL        = "Unknown"   # label returned when score < threshold
UNKNOWN_TEST_LABEL   = "Unknown"   # sub-folder name for impostor test images
AGGREGATION_STRATEGY = "mean"      # "mean" | "median" — how to combine per-image embeddings
MAX_FACES            = 10          # maximum faces to process per image

# ─────────────────────────────────────────────────────────────────────────────
#  Webcam
# ─────────────────────────────────────────────────────────────────────────────
WEBCAM_DEVICE_ID = 0     # 0 = default camera; increment for additional cameras
FRAME_WIDTH      = 1280
FRAME_HEIGHT     = 720

# ─────────────────────────────────────────────────────────────────────────────
#  Visualisation
# ─────────────────────────────────────────────────────────────────────────────
BBOX_COLOR_KNOWN   = (0, 200, 0)     # BGR green  — identified person
BBOX_COLOR_UNKNOWN = (0, 0, 220)     # BGR red    — unknown face
TEXT_COLOR         = (255, 255, 255) # white text
FONT_SCALE         = 0.70
FONT_THICKNESS     = 2
BOX_THICKNESS      = 2
