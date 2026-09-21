#!/usr/bin/env python3
"""
main.py — CLI Entry Point for the Face Recognition Identification System
=========================================================================
Provides six sub-commands:

  enroll      Build or rebuild the face embedding database from enrolled images.
  identify    Run recognition on a single image file.
  evaluate    Evaluate the system on the labelled test set and generate reports.
  webcam      Real-time face recognition via webcam.
  info        List all currently enrolled persons.
  calibrate   Find the Equal Error Rate (EER) threshold for your dataset.

Usage examples
--------------
  python main.py enroll
  python main.py identify --image path/to/photo.jpg --show
  python main.py identify --image photo.jpg --output result.jpg
  python main.py evaluate
  python main.py evaluate --threshold 0.40
  python main.py webcam
  python main.py info
  python main.py calibrate
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2
from rich.console import Console
from rich.panel import Panel

# ── Make src importable when running from the project root ───────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config as cfg
from src.enroll    import build_database, save_database, load_database, list_enrolled
from src.face_utils import load_image, load_model
from src.recognize  import recognize_image, print_results, print_all_scores, identify_face
from src.evaluate   import (
    run_evaluation, compute_metrics, print_metrics,
    plot_confusion_matrix, plot_similarity_distribution, plot_far_frr_curve,
)

console = Console()

# Show warnings and above; set DEBUG for verbose InsightFace output
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


# ─────────────────────────────────────────────────────────────────────────────
#  Sub-command: enroll
# ─────────────────────────────────────────────────────────────────────────────

def cmd_enroll(args) -> None:
    """Build the face database from images in the enrollment directory."""
    console.print(Panel(
        "🧑  [bold cyan]Face Enrollment[/bold cyan]\n"
        f"Source : [yellow]{args.enrolled_dir}[/yellow]\n"
        f"Output : [yellow]{args.db_path}[/yellow]",
        expand=False,
    ))

    model = load_model()
    db    = build_database(model, enrolled_dir=Path(args.enrolled_dir))

    if not db:
        console.print("[red]❌ No persons could be enrolled. Check the enrollment directory.[/red]")
        sys.exit(1)

    save_database(db, path=Path(args.db_path))


# ─────────────────────────────────────────────────────────────────────────────
#  Sub-command: identify
# ─────────────────────────────────────────────────────────────────────────────

def cmd_identify(args) -> None:
    """Identify all faces in a single image."""
    console.print(Panel(
        f"🔍  [bold cyan]Face Identification[/bold cyan]\n"
        f"Image     : [yellow]{args.image}[/yellow]\n"
        f"Threshold : [yellow]{args.threshold}[/yellow]",
        expand=False,
    ))

    # Load DB and model
    db    = load_database(Path(args.db_path))
    model = load_model()

    # Load image
    img = load_image(args.image)
    if img is None:
        console.print(f"[red]❌ Could not load image: {args.image}[/red]")
        sys.exit(1)

    # Run recognition
    results, annotated = recognize_image(model, img, db, threshold=args.threshold)
    print_results(results, threshold=args.threshold)

    # Optionally print per-person similarity breakdown
    if args.verbose and results:
        from src.face_utils import detect_and_embed, cosine_similarity
        faces = detect_and_embed(model, img)
        if faces:
            _, emb, _ = faces[0]
            _, _, all_scores = identify_face(emb, db, args.threshold)
            print_all_scores(all_scores, args.threshold)

    # Save or display annotated image
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), annotated)
        console.print(f"\n[bold green]💾 Annotated image saved → {out}[/bold green]")

    if args.show:
        cv2.imshow("Face Recognition  |  Press any key to close", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
#  Sub-command: evaluate
# ─────────────────────────────────────────────────────────────────────────────

def cmd_evaluate(args) -> None:
    """Run full evaluation on the labelled test set."""
    console.print(Panel(
        f"📊  [bold cyan]System Evaluation[/bold cyan]\n"
        f"Test dir  : [yellow]{args.test_dir}[/yellow]\n"
        f"Threshold : [yellow]{args.threshold}[/yellow]",
        expand=False,
    ))

    db    = load_database(Path(args.db_path))
    model = load_model()

    # Create output directories
    cfg.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Run evaluation
    df = run_evaluation(model, db, test_dir=Path(args.test_dir), threshold=args.threshold)

    if df.empty:
        console.print("[red]No results — check your test directory structure.[/red]")
        sys.exit(1)

    # Compute and display metrics
    metrics = compute_metrics(df, threshold=args.threshold)
    print_metrics(metrics)

    # Save per-image CSV
    df.to_csv(cfg.EVALUATION_CSV, index=False)
    console.print(f"\n  📄 Detailed CSV  → [cyan]{cfg.EVALUATION_CSV}[/cyan]")

    # Generate plots
    console.print("\n  Generating plots…")
    plot_confusion_matrix(df, cfg.PLOTS_DIR / "confusion_matrix.png")
    plot_similarity_distribution(df, args.threshold, cfg.PLOTS_DIR / "similarity_distribution.png")
    eer = plot_far_frr_curve(df, cfg.PLOTS_DIR / "far_frr_curve.png")

    console.print(f"\n[bold green]✅ Evaluation complete. Results in {cfg.RESULTS_DIR}[/bold green]")


# ─────────────────────────────────────────────────────────────────────────────
#  Sub-command: webcam
# ─────────────────────────────────────────────────────────────────────────────

def cmd_webcam(args) -> None:
    """Run real-time face recognition on the live webcam feed."""
    console.print(Panel(
        f"📷  [bold cyan]Live Webcam Recognition[/bold cyan]\n"
        f"Threshold : [yellow]{args.threshold}[/yellow]\n"
        "Press [bold]Q[/bold] to quit.",
        expand=False,
    ))

    db    = load_database(Path(args.db_path))
    model = load_model()

    cap = cv2.VideoCapture(cfg.WEBCAM_DEVICE_ID)
    if not cap.isOpened():
        console.print(
            f"[red]❌ Cannot open webcam (device ID = {cfg.WEBCAM_DEVICE_ID}).\n"
            "Check that a camera is connected and change WEBCAM_DEVICE_ID in config.py.[/red]"
        )
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cfg.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.FRAME_HEIGHT)
    console.print("[green]✅ Webcam opened — press Q to quit.[/green]")

    while True:
        ret, frame = cap.read()
        if not ret:
            console.print("[yellow]⚠  Frame capture failed — exiting.[/yellow]")
            break

        results, annotated = recognize_image(model, frame, db, threshold=args.threshold)
        cv2.imshow("Face Recognition  |  Press Q to quit", annotated)

        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    console.print("[green]Webcam session ended.[/green]")


# ─────────────────────────────────────────────────────────────────────────────
#  Sub-command: info
# ─────────────────────────────────────────────────────────────────────────────

def cmd_info(args) -> None:
    """Print the list of enrolled persons and database path."""
    console.print(Panel(
        f"📋  [bold cyan]Database Info[/bold cyan]\n"
        f"Path: [yellow]{args.db_path}[/yellow]",
        expand=False,
    ))
    db = load_database(Path(args.db_path))
    list_enrolled(db)
    console.print(
        f"\nThreshold (config): [yellow]{cfg.SIMILARITY_THRESHOLD}[/yellow]  "
        f"(use [cyan]--threshold[/cyan] to override at runtime)"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Sub-command: calibrate
# ─────────────────────────────────────────────────────────────────────────────

def cmd_calibrate(args) -> None:
    """
    Run evaluation with threshold=0 (captures all raw scores) and plot the
    FAR/FRR curve to find the EER-optimal threshold for this specific dataset.
    """
    console.print(Panel(
        f"⚙️  [bold cyan]Threshold Calibration[/bold cyan]\n"
        f"Test dir : [yellow]{args.test_dir}[/yellow]\n"
        "Sweeping thresholds 0 → 1 to find the EER point…",
        expand=False,
    ))

    db    = load_database(Path(args.db_path))
    model = load_model()
    cfg.PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Run with threshold=0 so ALL scores are captured (no early rejection)
    df = run_evaluation(model, db, test_dir=Path(args.test_dir), threshold=0.0)

    if df.empty:
        console.print("[red]No test data — cannot calibrate.[/red]")
        sys.exit(1)

    eer_thresh = plot_far_frr_curve(df, cfg.PLOTS_DIR / "far_frr_curve.png")

    if eer_thresh is not None:
        console.print(
            f"\n[bold]Recommendation:[/bold] Set "
            f"[cyan]SIMILARITY_THRESHOLD = {eer_thresh:.3f}[/cyan] "
            f"in [cyan]src/config.py[/cyan]."
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Argument Parser
# ─────────────────────────────────────────────────────────────────────────────

def _shared_parser() -> argparse.ArgumentParser:
    """Parent parser with arguments shared across all sub-commands."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--db-path", default=str(cfg.DATABASE_PATH),
        metavar="PATH",
        help=f"Embedding database file (default: {cfg.DATABASE_PATH})",
    )
    p.add_argument(
        "--threshold", type=float, default=cfg.SIMILARITY_THRESHOLD,
        metavar="FLOAT",
        help=f"Cosine similarity threshold (default: {cfg.SIMILARITY_THRESHOLD})",
    )
    return p


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description=(
            "Face Recognition Identification System\n"
            "Uses InsightFace (ArcFace-R50) + ONNX Runtime on CPU."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    shared = _shared_parser()
    sub    = parser.add_subparsers(dest="command", required=True)

    # ── enroll ──────────────────────────────────────────────────────────────
    p = sub.add_parser("enroll", parents=[shared], help="Build face database from enrollment images")
    p.add_argument(
        "--enrolled-dir", default=str(cfg.ENROLLED_DIR), metavar="DIR",
        help=f"Directory with per-person sub-folders (default: {cfg.ENROLLED_DIR})",
    )
    p.set_defaults(func=cmd_enroll)

    # ── identify ─────────────────────────────────────────────────────────────
    p = sub.add_parser("identify", parents=[shared], help="Identify faces in an image")
    p.add_argument("-i", "--image", required=True, metavar="FILE", help="Input image path")
    p.add_argument("-o", "--output", default=None, metavar="FILE", help="Save annotated image")
    p.add_argument("--show",    action="store_true", help="Display annotated image in a window")
    p.add_argument("--verbose", action="store_true", help="Print per-person similarity scores")
    p.set_defaults(func=cmd_identify)

    # ── evaluate ─────────────────────────────────────────────────────────────
    p = sub.add_parser("evaluate", parents=[shared], help="Run evaluation on the test set")
    p.add_argument(
        "--test-dir", default=str(cfg.TEST_DIR), metavar="DIR",
        help=f"Test directory with labelled sub-folders (default: {cfg.TEST_DIR})",
    )
    p.set_defaults(func=cmd_evaluate)

    # ── webcam ───────────────────────────────────────────────────────────────
    p = sub.add_parser("webcam", parents=[shared], help="Live webcam face recognition")
    p.set_defaults(func=cmd_webcam)

    # ── info ─────────────────────────────────────────────────────────────────
    p = sub.add_parser("info", parents=[shared], help="List enrolled persons")
    p.set_defaults(func=cmd_info)

    # ── calibrate ────────────────────────────────────────────────────────────
    p = sub.add_parser("calibrate", parents=[shared], help="Find optimal EER threshold")
    p.add_argument(
        "--test-dir", default=str(cfg.TEST_DIR), metavar="DIR",
        help=f"Test directory (default: {cfg.TEST_DIR})",
    )
    p.set_defaults(func=cmd_calibrate)

    return parser


# ─────────────────────────────────────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
