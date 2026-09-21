"""
evaluate.py — System Evaluation Pipeline
==========================================
Runs the face recognition system over a labelled test set and reports a full
suite of recognition metrics.  All metric values are computed from actual
test-data results — no numbers are fabricated.

Metrics reported
-----------------
• Accuracy                           (correct / total)
• Macro Precision / Recall / F1      (per-class then averaged)
• False Acceptance Rate (FAR)        FP / (FP + TN)
• False Rejection Rate (FRR)         FN / (FN + TP)
• True/False Positives/Negatives     raw counts
• Correctly / incorrectly identified counts
• Correctly rejected unknown count

Plots produced
--------------
• confusion_matrix.png       — heatmap of true vs predicted labels
• similarity_distribution.png — genuine vs impostor score histograms + threshold
• far_frr_curve.png           — FAR and FRR as functions of threshold;
                                marks the Equal Error Rate (EER) point

CSV output
----------
• evaluation.csv  — one row per test image with columns:
  image_path, true_label, predicted_label, best_similarity, det_score,
  face_detected, correct

Test directory layout expected
-------------------------------
data/test/
    Person_A/   ← must match an enrolled name; should be identified as Person_A
    Person_B/
    Unknown/    ← impostor images; should be rejected (= cfg.UNKNOWN_TEST_LABEL)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")   # use non-interactive backend — safe for servers/CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rich.console import Console
from rich.table import Table
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from tqdm import tqdm

from . import config as cfg
from .face_utils import detect_and_embed, load_image
from .recognize import identify_face

log     = logging.getLogger(__name__)
console = Console()

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


# ─────────────────────────────────────────────────────────────────────────────
#  Main Evaluation Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(
    model,
    database: Dict[str, np.ndarray],
    test_dir: Path = cfg.TEST_DIR,
    threshold: float = cfg.SIMILARITY_THRESHOLD,
) -> pd.DataFrame:
    """
    Iterate over every test image, run recognition, and collect results.

    For each image:
      • If no face is detected → predicted_label = Unknown (face_detected=False)
      • If a face is detected  → run identify_face() with the given threshold

    Returns a DataFrame with one row per image.
    Columns: image_path, true_label, predicted_label, best_similarity,
             det_score, face_detected, correct
    """
    test_dir = Path(test_dir)
    if not test_dir.exists():
        raise FileNotFoundError(
            f"Test directory not found: {test_dir}\n"
            "Populate it with sub-folders (one per person/Unknown) before evaluating."
        )

    person_dirs = sorted([d for d in test_dir.iterdir() if d.is_dir()])
    if not person_dirs:
        raise ValueError(f"No sub-folders found in test directory: {test_dir}")

    records = []

    for person_dir in person_dirs:
        true_label  = person_dir.name
        image_paths = [
            p for p in sorted(person_dir.iterdir())
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

        for img_path in tqdm(image_paths, desc=f"  Testing [{true_label}]", leave=False):
            record = _evaluate_single_image(
                model, img_path, true_label, database, threshold
            )
            records.append(record)

    if not records:
        console.print("[yellow]⚠  No test images found — results DataFrame is empty.[/yellow]")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    console.print(
        f"\n  Test images processed: [cyan]{len(df)}[/cyan] "
        f"| Faces detected: [cyan]{df['face_detected'].sum()}[/cyan]"
    )
    return df


def _evaluate_single_image(
    model,
    img_path: Path,
    true_label: str,
    database: Dict[str, np.ndarray],
    threshold: float,
) -> dict:
    """Run recognition on one test image and return a result record."""
    img = load_image(img_path)

    if img is None:
        # Corrupted or unreadable image → treat as no detection
        return {
            "image_path":      str(img_path),
            "true_label":      true_label,
            "predicted_label": cfg.UNKNOWN_LABEL,
            "best_similarity": 0.0,
            "det_score":       0.0,
            "face_detected":   False,
            "correct":         true_label == cfg.UNKNOWN_TEST_LABEL,
        }

    faces = detect_and_embed(model, img)

    if not faces:
        # No face found → system treats it as unknown
        is_correct = (true_label == cfg.UNKNOWN_TEST_LABEL)
        return {
            "image_path":      str(img_path),
            "true_label":      true_label,
            "predicted_label": cfg.UNKNOWN_LABEL,
            "best_similarity": 0.0,
            "det_score":       0.0,
            "face_detected":   False,
            "correct":         is_correct,
        }

    # Use the highest-confidence detected face
    bbox, emb, det_score = faces[0]
    pred_label, best_sim, _ = identify_face(emb, database, threshold)
    is_correct = (pred_label == true_label)

    return {
        "image_path":      str(img_path),
        "true_label":      true_label,
        "predicted_label": pred_label,
        "best_similarity": float(best_sim),
        "det_score":       float(det_score),
        "face_detected":   True,
        "correct":         is_correct,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Metric Computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(df: pd.DataFrame, threshold: float) -> Dict:
    """
    Compute the full set of recognition metrics from the evaluation DataFrame.

    Definitions
    -----------
    TP : genuine user correctly identified   (known true, correct prediction)
    FP : impostor falsely accepted           (unknown true, predicted as known)
    FN : genuine user falsely rejected       (known true, predicted as unknown)
    TN : impostor correctly rejected         (unknown true, predicted as unknown)

    FAR = FP / (FP + TN)   — proportion of impostors incorrectly accepted
    FRR = FN / (FN + TP)   — proportion of genuine users incorrectly rejected
    """
    y_true = df["true_label"].tolist()
    y_pred = df["predicted_label"].tolist()

    known_mask   = df["true_label"] != cfg.UNKNOWN_TEST_LABEL
    unknown_mask = df["true_label"] == cfg.UNKNOWN_TEST_LABEL

    n_known   = int(known_mask.sum())
    n_unknown = int(unknown_mask.sum())
    total     = len(df)

    # ── Counts ────────────────────────────────────────────────────────────────
    tp = int((known_mask   & df["correct"]).sum())                          # correct ID
    fn = int((known_mask   & ~df["correct"]).sum())                         # missed known
    fp = int((unknown_mask & (df["predicted_label"] != cfg.UNKNOWN_LABEL)).sum())  # false accept
    tn = int((unknown_mask & (df["predicted_label"] == cfg.UNKNOWN_LABEL)).sum())  # correct reject

    correctly_identified   = tp
    incorrectly_identified = fn + fp   # failed known + falsely accepted impostor
    correctly_rejected     = tn

    # ── Error rates ───────────────────────────────────────────────────────────
    far = fp / n_unknown if n_unknown > 0 else 0.0
    frr = fn / n_known   if n_known   > 0 else 0.0

    # ── Accuracy ──────────────────────────────────────────────────────────────
    accuracy = accuracy_score(y_true, y_pred)

    # ── Precision / Recall / F1  (from sklearn) ───────────────────────────────
    report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    macro_prec   = report.get("macro avg",    {}).get("precision",  0.0)
    macro_rec    = report.get("macro avg",    {}).get("recall",     0.0)
    macro_f1     = report.get("macro avg",    {}).get("f1-score",   0.0)
    weighted_f1  = report.get("weighted avg", {}).get("f1-score",   0.0)

    return {
        "threshold":                  threshold,
        "total_images":               total,
        "faces_detected":             int(df["face_detected"].sum()),
        "n_known_test":               n_known,
        "n_unknown_test":             n_unknown,
        "correctly_identified":       correctly_identified,
        "incorrectly_identified":     incorrectly_identified,
        "correctly_rejected_unknown": correctly_rejected,
        "TP":                         tp,
        "FP":                         fp,
        "FN":                         fn,
        "TN":                         tn,
        "accuracy":                   round(accuracy,    4),
        "FAR":                        round(far,         4),
        "FRR":                        round(frr,         4),
        "macro_precision":            round(macro_prec,  4),
        "macro_recall":               round(macro_rec,   4),
        "macro_f1":                   round(macro_f1,    4),
        "weighted_f1":                round(weighted_f1, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_similarity_distribution(
    df: pd.DataFrame,
    threshold: float,
    save_path: Path,
) -> None:
    """
    Histogram of similarity scores split into genuine and impostor populations.

    Genuine  = correct identifications of known persons
    Impostor = incorrect identifications + unknown test images

    This plot is essential for understanding threshold selection:
    the ideal threshold sits in the valley between the two distributions.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Genuine: known person correctly identified
    genuine  = df[
        df["correct"] & (df["true_label"] != cfg.UNKNOWN_TEST_LABEL)
    ]["best_similarity"]

    # Impostor: either unknown test images or known persons misidentified
    impostor = df[
        ~df["correct"] | (df["true_label"] == cfg.UNKNOWN_TEST_LABEL)
    ]["best_similarity"]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(
        genuine,  bins=30, alpha=0.7, color="#2ecc71",
        label=f"Genuine pairs (n={len(genuine)})", density=True,
    )
    ax.hist(
        impostor, bins=30, alpha=0.7, color="#e74c3c",
        label=f"Impostor pairs (n={len(impostor)})", density=True,
    )
    ax.axvline(
        threshold, color="#f39c12", linestyle="--", linewidth=2,
        label=f"Decision threshold = {threshold}",
    )
    ax.set_xlabel("Cosine Similarity Score", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        "Similarity Score Distribution — Genuine vs Impostor Pairs",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    console.print(f"  📊 Similarity distribution → [cyan]{save_path}[/cyan]")


def plot_confusion_matrix(df: pd.DataFrame, save_path: Path) -> None:
    """
    Plot and save the confusion matrix as a colour-coded heatmap.
    Rows = true labels; Columns = predicted labels.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    y_true = df["true_label"].tolist()
    y_pred = df["predicted_label"].tolist()
    all_labels = sorted(set(y_true + y_pred))

    cm   = confusion_matrix(y_true, y_pred, labels=all_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=all_labels)

    n = len(all_labels)
    fig, ax = plt.subplots(figsize=(max(6, n * 1.4), max(5, n * 1.2)))
    disp.plot(ax=ax, colorbar=True, cmap="Blues", xticks_rotation="vertical")
    ax.set_title("Confusion Matrix — Face Recognition System", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    console.print(f"  📊 Confusion matrix        → [cyan]{save_path}[/cyan]")


def plot_far_frr_curve(
    df: pd.DataFrame,
    save_path: Path,
) -> Optional[float]:
    """
    Sweep thresholds from 0 → 1 and plot FAR and FRR as functions of the threshold.
    The crossing point is the Equal Error Rate (EER) — a data-driven threshold choice.

    Returns the EER threshold value, or None if either population is missing.

    Note: this function re-uses pre-computed best_similarity scores from df,
    which were recorded at a specific threshold.  The sweep approximates what
    the system would do at every threshold point — it gives a good directional
    signal for threshold selection.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    known_df   = df[df["true_label"] != cfg.UNKNOWN_TEST_LABEL].copy()
    unknown_df = df[df["true_label"] == cfg.UNKNOWN_TEST_LABEL].copy()
    n_known    = len(known_df)
    n_unknown  = len(unknown_df)

    if n_known == 0 or n_unknown == 0:
        log.warning(
            "FAR/FRR curve requires both known and unknown test samples. "
            "Add an 'Unknown' sub-folder to data/test/ with impostor images."
        )
        return None

    thresholds = np.linspace(0.0, 1.0, 201)
    fars, frrs = [], []

    for t in thresholds:
        # At threshold t, how many genuine users would be rejected?
        frr = (known_df["best_similarity"] < t).sum() / n_known
        # At threshold t, how many impostors would be accepted?
        far = (unknown_df["best_similarity"] >= t).sum() / n_unknown
        frrs.append(frr)
        fars.append(far)

    fars_arr  = np.array(fars)
    frrs_arr  = np.array(frrs)
    diffs     = np.abs(fars_arr - frrs_arr)
    eer_idx   = int(np.argmin(diffs))
    eer_thresh = float(thresholds[eer_idx])
    eer_rate   = float((fars_arr[eer_idx] + frrs_arr[eer_idx]) / 2)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds, fars_arr,  label="FAR (False Acceptance Rate)", color="#e74c3c", linewidth=2)
    ax.plot(thresholds, frrs_arr,  label="FRR (False Rejection Rate)",  color="#3498db", linewidth=2)
    ax.axvline(
        eer_thresh, color="#2ecc71", linestyle="--", linewidth=2,
        label=f"EER threshold = {eer_thresh:.3f}  (EER = {eer_rate:.3f})",
    )
    ax.set_xlabel("Threshold", fontsize=12)
    ax.set_ylabel("Error Rate", fontsize=12)
    ax.set_title("FAR / FRR Curve — Equal Error Rate Analysis", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

    console.print(f"  📊 FAR/FRR curve           → [cyan]{save_path}[/cyan]")
    console.print(
        f"  [bold yellow]⭐ Suggested EER threshold: {eer_thresh:.3f}  "
        f"(EER = {eer_rate:.3f})[/bold yellow]"
    )
    console.print(
        f"  Set [cyan]SIMILARITY_THRESHOLD = {eer_thresh:.3f}[/cyan] in "
        f"[cyan]src/config.py[/cyan] to use the EER-optimal value for this dataset."
    )
    return eer_thresh


# ─────────────────────────────────────────────────────────────────────────────
#  Console Report
# ─────────────────────────────────────────────────────────────────────────────

def print_metrics(metrics: Dict) -> None:
    """Render a Rich table of all evaluation metrics."""
    table = Table(title="📊 Evaluation Metrics Report", header_style="bold magenta")
    table.add_column("Metric",  style="bold", min_width=38)
    table.add_column("Value",   justify="right")

    def row(label, key, fmt=None):
        v = metrics.get(key, "N/A")
        table.add_row(label, f"{v:{fmt}}" if fmt and isinstance(v, (int, float)) else str(v))

    def sep(title=""):
        table.add_row(f"[dim]── {title} ──[/dim]", "")

    row("Threshold",                      "threshold")
    sep("Dataset")
    row("Total Test Images",              "total_images")
    row("Faces Detected",                 "faces_detected")
    row("Known Test Images",              "n_known_test")
    row("Unknown (Impostor) Test Images", "n_unknown_test")
    sep("Identification Counts")
    row("Correctly Identified  (TP)",     "correctly_identified")
    row("Incorrectly Identified (FP+FN)", "incorrectly_identified")
    row("Correctly Rejected Unknown (TN)","correctly_rejected_unknown")
    sep("Confusion Matrix Entries")
    row("True Positives  (TP)",           "TP")
    row("False Positives (FP)",           "FP")
    row("False Negatives (FN)",           "FN")
    row("True Negatives  (TN)",           "TN")
    sep("Performance Rates")
    row("Accuracy",                       "accuracy",         ".4f")
    row("Macro Precision",                "macro_precision",  ".4f")
    row("Macro Recall",                   "macro_recall",     ".4f")
    row("Macro F1-Score",                 "macro_f1",         ".4f")
    row("Weighted F1-Score",              "weighted_f1",      ".4f")
    row("False Acceptance Rate (FAR)",    "FAR",              ".4f")
    row("False Rejection Rate  (FRR)",    "FRR",              ".4f")

    console.print(table)
