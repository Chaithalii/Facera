"""
api.py — Facera REST API Backend
=================================
Flask server that wraps the InsightFace face recognition system
and exposes HTTP endpoints consumed by the Facera website.

Run from the face-recognition-system directory:
    python api.py

Endpoints:
    GET  /api/health           — backend status + enrolled count
    GET  /api/database         — list all enrolled persons
    POST /api/enroll           — enroll a new person
    POST /api/identify         — identify faces in an image
    DELETE /api/delete/<name>  — remove a person from the database

The Facera website at facera/index.html calls these endpoints
via fetch() from the browser.
"""

import base64
import logging
import os
import pickle
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# Make src importable
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import config as cfg
from src.face_utils import cosine_similarity, detect_and_embed, load_model
from src.recognize import identify_face

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

FACERA_DIR = Path(__file__).resolve().parent / "facera"
app = Flask(__name__, static_folder=str(FACERA_DIR), static_url_path="")
CORS(app, origins="*")   # Allow browser requests from file:// and localhost

# ──────────────────────────────────────────────────────────────────────────────
#  Globals (loaded once at startup)
# ──────────────────────────────────────────────────────────────────────────────
_model    = None
_database = {}
_db_lock  = threading.Lock()   # serialises all read-modify-write operations


def get_model():
    global _model
    if _model is None:
        log.info("Loading InsightFace model (buffalo_l)…")
        _model = load_model()
        log.info("Model ready.")
    return _model


def _load_db_from_disk() -> dict:
    """
    Read the pickle file and return its contents (does NOT touch _database).
    Call this INSIDE _db_lock only.
    """
    path = cfg.DATABASE_PATH
    if path.exists():
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as exc:
            log.error("Could not load database: %s", exc)
    return {}


def _save_db_to_disk(db: dict) -> None:
    """
    Write db to the pickle file.
    Call this INSIDE _db_lock only.
    """
    path = cfg.DATABASE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(db, f)
    log.info("Database saved -> %s  (%d persons)", path, len(db))


def refresh_database() -> dict:
    """
    Reload from disk and update the global _database.
    Safe to call at startup or from write endpoints (inside _db_lock).
    """
    global _database
    _database = _load_db_from_disk()
    if not _database:
        seed_path = cfg.ROOT_DIR / "embeddings" / "demo_database.pkl"
        if seed_path.exists():
            try:
                with open(seed_path, "rb") as f:
                    _database = pickle.load(f)
                _save_db_to_disk(_database)
                log.info("Initialized database from seed demo database (%d persons).", len(_database))
            except Exception as exc:
                log.warning("Could not load seed database: %s", exc)
    log.info("Loaded %d persons from database.", len(_database))
    return _database


def persist_database() -> None:
    """Save the in-memory _database to disk (call inside _db_lock)."""
    _save_db_to_disk(_database)



# ──────────────────────────────────────────────────────────────────────────────
#  Image Utilities
# ──────────────────────────────────────────────────────────────────────────────

def decode_b64_image(b64: str) -> np.ndarray | None:
    """Decode a base64 image string (with or without data-URL prefix) → BGR."""
    try:
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        raw = base64.b64decode(b64)
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception as exc:
        log.warning("Could not decode image: %s", exc)
        return None


def encode_bgr_to_b64(img_bgr: np.ndarray, max_size: int = 900) -> str:
    """Resize if needed and encode BGR image → base64 JPEG."""
    h, w = img_bgr.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def draw_results(img_bgr: np.ndarray, results: list) -> np.ndarray:
    """Draw bounding boxes and labels onto a copy of img_bgr."""
    out = img_bgr.copy()
    # Terracotta (176,58,36) in BGR: (36,58,176); Forest green (45,96,64) BGR: (64,96,45)
    for r in results:
        x1, y1, x2, y2 = r["bbox"]
        is_known = r["is_known"]
        color_bgr = (64, 96, 45) if is_known else (36, 58, 176)  # BGR
        cv2.rectangle(out, (x1, y1), (x2, y2), color_bgr, 2)

        label = f"{r['identity']}  {r['similarity']:.3f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        # Label background
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 8, y1), color_bgr, -1)
        cv2.putText(out, label, (x1 + 4, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# ──────────────────────────────────────────────────────────────────────────────
#  Frontend Static Routes
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    return send_from_directory(str(FACERA_DIR), "index.html")

@app.route("/<path:path>", methods=["GET"])
def static_proxy(path):
    target = FACERA_DIR / path
    if target.exists() and target.is_file():
        return send_from_directory(str(FACERA_DIR), path)
    return jsonify({"error": "Not found"}), 404


# ──────────────────────────────────────────────────────────────────────────────
#  Endpoints
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    # Read-only: use in-memory state, no disk reload (avoids race with enroll)
    return jsonify({
        "status":   "ok",
        "model":    cfg.MODEL_NAME,
        "enrolled": len(_database),
        "persons":  sorted(_database.keys()),
    })


@app.route("/api/database", methods=["GET"])
def list_database():
    # Read-only: use in-memory state, no disk reload
    persons = []
    for name, emb in sorted(_database.items()):
        persons.append({
            "name": name,
            "embedding_dim": int(emb.shape[0]),
            "norm": round(float(np.linalg.norm(emb)), 5),
        })
    return jsonify({"persons": persons, "count": len(persons)})


@app.route("/api/enroll", methods=["POST"])
def enroll():
    global _database          # declared here so Python allows use anywhere in this function
    data  = request.get_json(silent=True) or {}
    name  = data.get("name", "").strip().replace(" ", "_")
    imgs  = data.get("images", [])          # list of base64 strings

    if not name:
        return jsonify({"error": "Person name is required."}), 400
    if not imgs:
        return jsonify({"error": "At least one image is required."}), 400

    model = get_model()

    # ── Compute embeddings OUTSIDE the lock (this is the slow part) ───────────
    embeddings: list[np.ndarray] = []
    failed = 0

    for b64 in imgs:
        img_bgr = decode_b64_image(b64)
        if img_bgr is None:
            failed += 1
            continue
        faces = detect_and_embed(model, img_bgr)
        if faces:
            _, emb, _ = faces[0]
            embeddings.append(emb)
        else:
            failed += 1

    if not embeddings:
        return jsonify({
            "error": "No faces detected in any of the uploaded images. "
                     "Ensure the face is clearly visible and well-lit."
        }), 422

    # Aggregate + re-normalise
    agg = np.mean(np.stack(embeddings), axis=0)
    agg = agg / (np.linalg.norm(agg) + 1e-10)
    agg = agg.astype(np.float32)

    # ── Duplicate check: always read the freshest DB from DISK ────────────────
    # We intentionally reload from disk here (same as the write does) so the
    # check is never fooled by a stale in-memory snapshot.
    # Use a lower threshold (0.30) than identification so same-person photos
    # are always caught even if different angles/lighting.
    DUP_THRESHOLD = 0.30
    with _db_lock:
        check_db = _load_db_from_disk()     # freshest on-disk state
        if not check_db:                    # if disk is empty, fall back to memory
            check_db = dict(_database)

    log.info("Duplicate check: comparing '%s' against %d existing persons (threshold=%.2f)",
             name, len(check_db), DUP_THRESHOLD)

    best_match_name  = None
    best_match_score = -1.0
    for existing_name, existing_emb in check_db.items():
        if existing_name == name:              # same name → update allowed, not duplicate
            continue
        sim = float(np.dot(agg, existing_emb))
        log.info("  similarity('%s', '%s') = %.5f", name, existing_name, sim)
        if sim > best_match_score:
            best_match_score = sim
            best_match_name  = existing_name

    log.info("Best match: '%s' @ %.5f (threshold=%.2f)",
             best_match_name, best_match_score, DUP_THRESHOLD)

    if best_match_name and best_match_score >= DUP_THRESHOLD:
        log.info("DUPLICATE BLOCKED: '%s' matches '%s' (%.5f >= %.2f)",
                 name, best_match_name, best_match_score, DUP_THRESHOLD)
        return jsonify({
            "duplicate":    True,
            "matched_name": best_match_name,
            "similarity":   round(best_match_score, 5),
            "error": (
                f"This face is already enrolled as '{best_match_name}' "
                f"(similarity {best_match_score:.3f} \u2265 {DUP_THRESHOLD}). "
                f"Enrolling the same face under a different name is not allowed."
            ),
        }), 409

    # ── Atomic read-modify-write under lock ───────────────────────────────────
    with _db_lock:
        _database = _load_db_from_disk()      # get the freshest state
        _database[name] = agg
        _save_db_to_disk(_database)           # persist immediately

    # ─────────────────────────────────────────────────────────────────────────

    # Save source images to disk (outside lock — non-critical)
    person_dir = cfg.ENROLLED_DIR / name
    person_dir.mkdir(parents=True, exist_ok=True)
    for i, b64 in enumerate(imgs):
        img = decode_b64_image(b64)
        if img is not None:
            cv2.imwrite(str(person_dir / f"enrolled_{i+1:03d}.jpg"), img)

    log.info("Enrolled '%s' -- %d/%d images used.", name, len(embeddings), len(imgs))
    return jsonify({
        "success":        True,
        "name":           name,
        "faces_used":     len(embeddings),
        "total_images":   len(imgs),
        "failed":         failed,
        "embedding_dim":  int(agg.shape[0]),
        "total_enrolled": len(_database),
    })


@app.route("/api/identify", methods=["POST"])
def identify():
    data      = request.get_json(silent=True) or {}
    b64_image = data.get("image", "")
    threshold = float(data.get("threshold", cfg.SIMILARITY_THRESHOLD))

    if not b64_image:
        return jsonify({"error": "Image is required."}), 400

    # Snapshot the current db (no disk reload — use live in-memory state)
    db = dict(_database)
    if not db:
        return jsonify({
            "error": "No persons enrolled yet. Enroll someone first.",
            "face_count": 0,
            "faces": [],
        }), 422

    img_bgr = decode_b64_image(b64_image)
    if img_bgr is None:
        return jsonify({"error": "Could not decode image."}), 400

    model = get_model()
    faces = detect_and_embed(model, img_bgr)

    if not faces:
        blank = encode_bgr_to_b64(img_bgr)
        return jsonify({
            "face_count": 0,
            "faces":      [],
            "annotated_image": blank,
            "message": "No faces detected in this image.",
        })

    results = []
    for bbox, emb, det_score in faces:
        name, sim, all_scores = identify_face(emb, db, threshold)
        results.append({
            "bbox":            [int(x) for x in bbox],
            "identity":        name,
            "similarity":      round(float(sim), 5),
            "detection_score": round(float(det_score), 4),
            "is_known":        name != cfg.UNKNOWN_LABEL,
            "all_scores": {
                k: round(float(v), 5)
                for k, v in sorted(all_scores.items(), key=lambda x: -x[1])
            },
        })

    annotated = draw_results(img_bgr, results)
    return jsonify({
        "face_count":      len(results),
        "faces":           results,
        "annotated_image": encode_bgr_to_b64(annotated),
    })


@app.route("/api/delete/<path:name>", methods=["DELETE"])
def delete_person(name: str):
    name = name.strip()
    with _db_lock:
        global _database
        _database = _load_db_from_disk()      # get the freshest state
        if name not in _database:
            return jsonify({"error": f"'{name}' not found in database."}), 404
        del _database[name]
        _save_db_to_disk(_database)
    log.info("Deleted '%s'.", name)
    return jsonify({"success": True, "deleted": name, "remaining": len(_database)})


# ──────────────────────────────────────────────────────────────────────────────
#  Module-level Startup  (runs under BOTH `python api.py` AND gunicorn api:app)
# ──────────────────────────────────────────────────────────────────────────────
# Gunicorn imports this module and never runs __main__, so startup must happen
# here.  We pre-load the model in a background thread so Gunicorn's worker
# doesn't block (and time out) while InsightFace downloads ~300 MB on first run.

def _startup():
    log.info("=== Facera startup: loading model and database ===")
    get_model()          # download + load InsightFace buffalo_l (once)
    refresh_database()   # seed from demo_database.pkl if live DB is empty
    log.info("=== Startup complete: %d person(s) in database ===", len(_database))

_startup_thread = threading.Thread(target=_startup, daemon=True, name="facera-startup")
_startup_thread.start()


# ──────────────────────────────────────────────────────────────────────────────
#  Entry Point  (only when running directly: `python api.py`)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    log.info("=" * 60)
    log.info("  Facera Web Application & API Server")
    log.info("  InsightFace · ArcFace-R50 · ONNX Runtime")
    log.info("=" * 60)
    log.info("Loading model on startup (first run downloads ~300 MB)…")
    get_model()
    # refresh_database() already called at module level above
    log.info("Enrolled persons: %d", len(_database))
    log.info("Server starting at http://127.0.0.1:%d", port)
    log.info("Open http://127.0.0.1:%d/ in your browser.", port)
    log.info("=" * 60)
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)

