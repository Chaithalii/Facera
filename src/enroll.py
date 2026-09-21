"""
enroll.py — Face Enrollment Pipeline
======================================
Scans the enrollment directory, extracts face embeddings for each person,
aggregates them into a single representative vector, and persists the
resulting database as a pickle file.

Expected directory layout
--------------------------
data/enrolled/
    Person_A/
        image1.jpg
        image2.jpg
        image3.jpg
    Person_B/
        image1.jpg
        ...

For each person:
  1. Load every image in their sub-folder.
  2. Run face detection — use the highest-confidence detection.
  3. Extract the 512-d ArcFace embedding for that face.
  4. Aggregate all per-image embeddings (mean or median, see config.py).
  5. Re-normalise the aggregate to unit length.
  6. Store {person_name: aggregate_embedding} in the database.

Design decisions
-----------------
• Only the top-scored face in each image is used; extra faces are ignored.
  This keeps enrollment deterministic when enrollment images contain only
  one person (the expected case).
• The aggregate embedding is re-normalised after mean/median so that
  cosine similarity remains equivalent to a dot product.
• Missing or undecodable images are skipped gracefully with a log warning.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from rich.console import Console
from rich.progress import track
from rich.table import Table

from . import config as cfg
from .face_utils import load_image, detect_and_embed, cosine_similarity

log     = logging.getLogger(__name__)
console = Console()

# Image file extensions accepted during enrollment
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


# ─────────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────────

def build_database(
    model,
    enrolled_dir: Path = cfg.ENROLLED_DIR,
) -> Dict[str, np.ndarray]:
    """
    Walk `enrolled_dir` and build a face embedding database.

    Parameters
    ----------
    model        : FaceAnalysis instance from face_utils.load_model()
    enrolled_dir : Directory containing one sub-folder per person.

    Returns
    -------
    dict mapping person_name → aggregated 512-d embedding (float32, unit-norm)
    """
    enrolled_dir = Path(enrolled_dir)
    if not enrolled_dir.exists():
        raise FileNotFoundError(f"Enrollment directory not found: {enrolled_dir}")

    person_dirs = sorted([d for d in enrolled_dir.iterdir() if d.is_dir()])
    if not person_dirs:
        raise ValueError(
            f"No person sub-directories found under: {enrolled_dir}\n"
            "Create a folder per person, e.g. data/enrolled/Person_A/"
        )

    database: Dict[str, np.ndarray] = {}
    summary_rows: List[Tuple] = []

    for person_dir in person_dirs:
        name = person_dir.name
        image_paths = _collect_images(person_dir)

        if not image_paths:
            log.warning("No supported images found for '%s' — skipping.", name)
            summary_rows.append((name, 0, 0, "⚠ No images found"))
            continue

        embeddings = _extract_embeddings_for_person(model, name, image_paths)

        if not embeddings:
            log.warning("No valid face detected in any image for '%s' — skipping.", name)
            summary_rows.append((name, len(image_paths), 0, "❌ No face detected"))
            continue

        agg_embedding = _aggregate_embeddings(embeddings)
        database[name] = agg_embedding
        summary_rows.append((name, len(image_paths), len(embeddings), "✅ Enrolled"))
        log.info(
            "Enrolled '%s': %d/%d images yielded a valid embedding.",
            name, len(embeddings), len(image_paths),
        )

    _print_enrollment_summary(summary_rows)
    return database


def save_database(
    database: Dict[str, np.ndarray],
    path: Path = cfg.DATABASE_PATH,
) -> None:
    """
    Persist the embedding database to disk as a pickle file.
    Creates parent directories if needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(database, fh, protocol=pickle.HIGHEST_PROTOCOL)
    console.print(f"\n[bold green]✅ Database saved → {path}[/bold green]")
    console.print(f"   Enrolled persons : [cyan]{len(database)}[/cyan]")


def load_database(
    path: Path = cfg.DATABASE_PATH,
) -> Dict[str, np.ndarray]:
    """
    Load a persisted embedding database from disk.
    Raises FileNotFoundError with a helpful message if the DB does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Embedding database not found: {path}\n"
            "Run enrollment first:  python main.py enroll"
        )
    with open(path, "rb") as fh:
        db = pickle.load(fh)
    log.info("Loaded database with %d enrolled persons from %s", len(db), path)
    return db


def list_enrolled(database: Dict[str, np.ndarray]) -> None:
    """Render a pretty table of enrolled persons to the console."""
    table = Table(
        title="📋 Enrolled Persons", show_header=True, header_style="bold cyan"
    )
    table.add_column("#",              style="dim", width=4)
    table.add_column("Name",           style="bold")
    table.add_column("Embedding Dim",  justify="right")
    table.add_column("Norm",           justify="right")

    for i, (name, emb) in enumerate(sorted(database.items()), start=1):
        table.add_row(
            str(i),
            name,
            str(emb.shape[0]),
            f"{np.linalg.norm(emb):.4f}",
        )
    console.print(table)
    console.print(f"Total enrolled: [bold cyan]{len(database)}[/bold cyan]")


# ─────────────────────────────────────────────────────────────────────────────
#  Internal Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _collect_images(person_dir: Path) -> List[Path]:
    """Return sorted list of supported image paths inside `person_dir`."""
    return sorted(
        p for p in person_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _extract_embeddings_for_person(
    model,
    name: str,
    image_paths: List[Path],
) -> List[np.ndarray]:
    """
    For each image, detect a face and extract its embedding.
    Images where no face is detected are silently skipped.
    Returns list of raw (not aggregated) 512-d embeddings.
    """
    embeddings: List[np.ndarray] = []

    for img_path in track(image_paths, description=f"  Enrolling [cyan]{name}[/cyan]"):
        img = load_image(img_path)
        if img is None:
            # load_image already logs a warning
            continue

        faces = detect_and_embed(model, img)
        if not faces:
            log.debug("No face detected in '%s' for person '%s'.", img_path.name, name)
            continue

        # Take the highest-confidence detection (first after sorting)
        _, emb, _ = faces[0]
        embeddings.append(emb)

    return embeddings


def _aggregate_embeddings(embeddings: List[np.ndarray]) -> np.ndarray:
    """
    Combine multiple embeddings into one representative vector.

    Strategy is controlled by config.AGGREGATION_STRATEGY:
      "mean"   → arithmetic mean (fast, standard)
      "median" → component-wise median (more robust to outliers)

    The result is re-normalised to unit length so it can be used directly
    with cosine similarity (≡ dot product for unit vectors).
    """
    stack = np.stack(embeddings, axis=0)           # shape: (N, 512)

    if cfg.AGGREGATION_STRATEGY == "median":
        agg = np.median(stack, axis=0)
    else:
        agg = np.mean(stack, axis=0)               # default: mean

    # Re-normalise to restore unit-norm property
    norm = np.linalg.norm(agg)
    if norm > 1e-10:
        agg = agg / norm

    return agg.astype(np.float32)


def _print_enrollment_summary(rows: List[Tuple]) -> None:
    """Print a Rich table summarising the enrollment process."""
    table = Table(title="📝 Enrollment Summary", header_style="bold magenta")
    table.add_column("Person",          style="bold")
    table.add_column("Images Found",    justify="right")
    table.add_column("Faces Extracted", justify="right")
    table.add_column("Status")

    for name, n_imgs, n_faces, status in rows:
        table.add_row(name, str(n_imgs), str(n_faces), status)

    console.print(table)
