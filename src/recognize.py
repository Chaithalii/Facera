"""
recognize.py — Face Recognition and Identification
====================================================
Provides the identification logic:
  1. Detect all faces in an input image.
  2. For each detected face, extract its 512-d ArcFace embedding.
  3. Compute cosine similarity against every enrolled embedding.
  4. Select the best match.
  5. Apply the threshold:
       best_score >= threshold  →  return the enrolled person's name
       best_score <  threshold  →  return "Unknown"

Example
--------
Enrolled: Person_A (0.89), Person_B (0.43), Person_C (0.37)
Threshold: 0.65
→ Identified as Person_A (similarity = 0.89)

Enrolled: Person_A (0.42), Person_B (0.51), Person_C (0.38)
Threshold: 0.65
→ Unknown (best similarity = 0.51, below threshold)
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
from rich.console import Console
from rich.table import Table

from . import config as cfg
from .face_utils import cosine_similarity, detect_and_embed, draw_result, load_image

log     = logging.getLogger(__name__)
console = Console()

# Each entry: (bounding_box, predicted_name, best_similarity, detection_score)
IdentifyResult = Tuple[np.ndarray, str, float, float]


# ─────────────────────────────────────────────────────────────────────────────
#  Core Identification Logic
# ─────────────────────────────────────────────────────────────────────────────

def identify_face(
    embedding: np.ndarray,
    database: Dict[str, np.ndarray],
    threshold: float = cfg.SIMILARITY_THRESHOLD,
) -> Tuple[str, float, Dict[str, float]]:
    """
    Compare `embedding` against every enrolled embedding in `database`.

    Algorithm
    ----------
    1. Compute cosine similarity between the query embedding and each enrolled
       representative embedding.
    2. Find the enrolled person with the highest similarity score.
    3. If best_score >= threshold  →  return that person's name.
       If best_score <  threshold  →  return UNKNOWN_LABEL.

    Parameters
    ----------
    embedding : 512-d face embedding from the query image (unit-norm float32)
    database  : dict mapping enrolled person name → representative embedding
    threshold : minimum cosine similarity to accept a match

    Returns
    -------
    (predicted_name, best_similarity, all_scores_dict)
    all_scores_dict lets callers inspect every enrolled person's similarity.
    """
    if not database:
        log.warning("Identification attempted on an empty database.")
        return cfg.UNKNOWN_LABEL, 0.0, {}

    # Compute similarity against every enrolled person
    all_scores: Dict[str, float] = {
        name: cosine_similarity(embedding, db_emb)
        for name, db_emb in database.items()
    }

    # Find the best match
    best_name  = max(all_scores, key=all_scores.get)
    best_score = all_scores[best_name]

    # Threshold decision
    if best_score >= threshold:
        return best_name, best_score, all_scores
    return cfg.UNKNOWN_LABEL, best_score, all_scores


def recognize_image(
    model,
    img_bgr: np.ndarray,
    database: Dict[str, np.ndarray],
    threshold: float = cfg.SIMILARITY_THRESHOLD,
) -> Tuple[List[IdentifyResult], np.ndarray]:
    """
    Detect ALL faces in `img_bgr`, identify each one, and return results
    together with an annotated copy of the image.

    Parameters
    ----------
    model     : FaceAnalysis instance from face_utils.load_model()
    img_bgr   : Input image in BGR format
    database  : Enrolled embedding database
    threshold : Similarity threshold for identification vs rejection

    Returns
    -------
    (results, annotated_image)
    results : list of (bbox, name, similarity, det_score) — one per face
    annotated_image : copy of img_bgr with coloured bounding boxes and labels
    """
    annotated = img_bgr.copy()
    results: List[IdentifyResult] = []

    # ── Step 1: Detect faces ────────────────────────────────────────────────
    faces = detect_and_embed(model, img_bgr)

    if not faces:
        log.info("No face detected in the provided image.")
        return results, annotated

    # ── Step 2: Identify each face ──────────────────────────────────────────
    for bbox, emb, det_score in faces:
        pred_name, sim_score, _ = identify_face(emb, database, threshold)
        is_known = pred_name != cfg.UNKNOWN_LABEL
        results.append((bbox, pred_name, sim_score, det_score))

        # ── Step 3: Annotate ─────────────────────────────────────────────────
        annotated = draw_result(annotated, bbox, pred_name, sim_score, is_known)

    return results, annotated


# ─────────────────────────────────────────────────────────────────────────────
#  Pretty Printing
# ─────────────────────────────────────────────────────────────────────────────

def print_results(
    results: List[IdentifyResult],
    threshold: float = cfg.SIMILARITY_THRESHOLD,
) -> None:
    """Render a Rich table of recognition results to the console."""
    if not results:
        console.print("[yellow]⚠  No faces were detected in this image.[/yellow]")
        return

    table = Table(title="🔍 Face Recognition Results", header_style="bold cyan")
    table.add_column("Face #",      style="dim", width=7)
    table.add_column("Prediction",  style="bold")
    table.add_column("Similarity",  justify="right")
    table.add_column("Threshold",   justify="right")
    table.add_column("Det. Score",  justify="right")
    table.add_column("Verdict")

    for i, (bbox, name, sim, det) in enumerate(results, start=1):
        verdict = (
            "[bold green]✅ Identified[/bold green]"
            if name != cfg.UNKNOWN_LABEL
            else "[bold red]❌ Unknown[/bold red]"
        )
        table.add_row(
            str(i),
            name,
            f"{sim:.4f}",
            f"{threshold}",
            f"{det:.3f}",
            verdict,
        )

    console.print(table)
    console.print(
        f"\n  Threshold used: [yellow]{threshold}[/yellow]  "
        f"(configurable in [cyan]src/config.py[/cyan] or via [cyan]--threshold[/cyan] flag)"
    )


def print_all_scores(all_scores: Dict[str, float], threshold: float) -> None:
    """Print the full score table for a single face query."""
    if not all_scores:
        return
    table = Table(title="📊 Similarity Scores vs All Enrolled Persons", header_style="bold")
    table.add_column("Person",     style="bold")
    table.add_column("Similarity", justify="right")
    table.add_column("Match?")

    for name, score in sorted(all_scores.items(), key=lambda x: x[1], reverse=True):
        match = "[green]✅ Yes[/green]" if score >= threshold else "[red]No[/red]"
        table.add_row(name, f"{score:.4f}", match)
    console.print(table)
