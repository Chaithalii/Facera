"""
face_utils.py — Core Face Detection, Alignment, and Embedding Utilities
=========================================================================
Wraps InsightFace's FaceAnalysis pipeline which internally runs two ONNX models:

  1. RetinaFace (det_10g.onnx)
     • Detects all faces in the image.
     • Localises five 2D facial landmarks (left eye, right eye, nose,
       left mouth corner, right mouth corner) for each detected face.

  2. ArcFace-R50 (w600k_r50.onnx)
     • Receives the affine-aligned 112×112 face crop produced from landmarks.
     • Outputs a 512-dimensional, L2-normalised face embedding.

Both models run on CPU via ONNX Runtime (no GPU required).

Key design decisions
---------------------
• Cosine similarity is used as the distance metric because ArcFace embeddings
  are L2-normalised unit vectors.  For unit vectors, cosine similarity equals
  the dot product — fast and numerically stable.
• We re-normalise before every similarity computation to guard against any
  floating-point drift that may occur after aggregation.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from . import config as cfg

log = logging.getLogger(__name__)

# Type alias: (bounding_box [x1,y1,x2,y2], 512-d embedding, detection_score)
FaceResult = Tuple[np.ndarray, np.ndarray, float]


# ─────────────────────────────────────────────────────────────────────────────
#  Model Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model():
    """
    Initialise and return the InsightFace FaceAnalysis model.

    On first call the model weights (~300 MB) are automatically downloaded
    from the InsightFace S3 bucket to ~/.insightface/models/buffalo_l/.
    Subsequent calls use the cached weights.
    """
    try:
        from insightface.app import FaceAnalysis
    except ImportError as exc:
        raise ImportError(
            "InsightFace is not installed.\n"
            "Install it with:  pip install insightface onnxruntime"
        ) from exc

    log.info("Loading InsightFace model '%s' …", cfg.MODEL_NAME)
    model = FaceAnalysis(
        name=cfg.MODEL_NAME,
        providers=cfg.MODEL_PROVIDERS,
    )
    # ctx_id=0 selects the first device (CPU when using CPUExecutionProvider)
    model.prepare(ctx_id=0, det_size=cfg.DETECTION_SIZE)
    log.info("InsightFace model loaded successfully.")
    return model


# ─────────────────────────────────────────────────────────────────────────────
#  Image I/O Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_image(path) -> Optional[np.ndarray]:
    """
    Load an image from disk in BGR format (OpenCV convention).
    Returns None if the file is missing, unreadable, or corrupted — never raises.
    """
    path = Path(path)
    if not path.exists():
        log.warning("Image file not found: %s", path)
        return None
    img = cv2.imread(str(path))
    if img is None:
        log.warning("Failed to decode image (corrupted or unsupported format): %s", path)
    return img


def load_image_rgb(path) -> Optional[np.ndarray]:
    """Load image as RGB numpy array (for matplotlib / Streamlit display)."""
    img_bgr = load_image(path)
    if img_bgr is None:
        return None
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR (OpenCV) to RGB (display/Streamlit)."""
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


# ─────────────────────────────────────────────────────────────────────────────
#  Face Detection + Embedding Extraction
# ─────────────────────────────────────────────────────────────────────────────

def detect_and_embed(
    model,
    img_bgr: np.ndarray,
    max_faces: int = cfg.MAX_FACES,
) -> List[FaceResult]:
    """
    Run the full InsightFace pipeline on a BGR image.

    Pipeline (handled internally by InsightFace):
      Step 1 — RetinaFace detects all faces and their 5-point landmarks.
      Step 2 — Each face is affine-warped to a canonical 112×112 aligned crop
               using the landmarks (brings eyes to fixed pixel positions).
      Step 3 — ArcFace-R50 encodes each aligned crop to a 512-d embedding.
      Step 4 — The embedding is L2-normalised to unit length.

    Parameters
    ----------
    model     : FaceAnalysis instance returned by load_model()
    img_bgr   : BGR image as a uint8 numpy array
    max_faces : hard cap on the number of faces returned (sorted by det_score)

    Returns
    -------
    List of (bbox, embedding, det_score) tuples, sorted by det_score descending.
    Empty list when no face is detected or the image is empty/None.
    """
    if img_bgr is None or img_bgr.size == 0:
        log.debug("detect_and_embed: received empty image, skipping.")
        return []

    try:
        faces = model.get(img_bgr)
    except Exception as exc:
        log.error("InsightFace inference error: %s", exc)
        return []

    results: List[FaceResult] = []
    for face in faces:
        score = float(face.det_score)
        if score < cfg.DET_SCORE_THRESH:
            continue                          # discard low-confidence detections
        bbox = face.bbox.astype(int)          # [x1, y1, x2, y2] pixel coordinates
        emb  = face.embedding.astype(np.float32)  # 512-d, already L2-normalised
        results.append((bbox, emb, score))

    # Sort descending by detection confidence; apply cap
    results.sort(key=lambda r: r[2], reverse=True)
    return results[:max_faces]


# ─────────────────────────────────────────────────────────────────────────────
#  Cosine Similarity
# ─────────────────────────────────────────────────────────────────────────────

def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Compute cosine similarity between two face embeddings.

    Because InsightFace already L2-normalises embeddings, the cosine
    similarity equals the dot product.  We re-normalise here to be safe
    against any floating-point drift that can occur after aggregation
    (e.g. taking the mean of several embeddings).

    Returns a float in [-1.0, 1.0].  Higher = more similar.
    """
    n1 = np.linalg.norm(emb1)
    n2 = np.linalg.norm(emb2)
    if n1 < 1e-10 or n2 < 1e-10:
        return 0.0
    dot = np.dot(emb1 / n1, emb2 / n2)
    return float(np.clip(dot, -1.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
#  Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def draw_result(
    img: np.ndarray,
    bbox: np.ndarray,
    label: str,
    score: float,
    is_known: bool,
) -> np.ndarray:
    """
    Draw a labelled bounding box on the image and return the annotated copy.

    Green box  →  identified (known) person
    Red box    →  unknown face

    The label text shows the predicted name and the cosine similarity score.
    """
    img = img.copy()
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    color = cfg.BBOX_COLOR_KNOWN if is_known else cfg.BBOX_COLOR_UNKNOWN

    # ── Bounding rectangle ───────────────────────────────────────────────────
    cv2.rectangle(img, (x1, y1), (x2, y2), color, cfg.BOX_THICKNESS)

    # ── Label pill (filled rectangle behind the text) ────────────────────────
    text = f"{label}  {score:.3f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, cfg.FONT_SCALE, cfg.FONT_THICKNESS)
    pad   = 4
    bg_y1 = max(y1 - th - baseline - 2 * pad, 0)
    bg_y2 = y1
    cv2.rectangle(img, (x1, bg_y1), (x1 + tw + 2 * pad, bg_y2), color, -1)

    # ── Text ─────────────────────────────────────────────────────────────────
    cv2.putText(
        img, text,
        (x1 + pad, y1 - baseline - pad),
        font, cfg.FONT_SCALE, cfg.TEXT_COLOR, cfg.FONT_THICKNESS, cv2.LINE_AA,
    )
    return img
