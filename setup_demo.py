"""
setup_demo.py — Download and Prepare Demo Data for the Face Recognition System
================================================================================
This script sets up a working demonstration of the system without requiring
you to provide your own face images.

It downloads the Labeled Faces in the Wild (LFW) dataset via scikit-learn,
selects a subset of people with sufficient images, and organises them into
the enrollment and test directory structure expected by the system.

What this script does
----------------------
1. Download LFW (≈250 MB, cached after first run by scikit-learn in ~/scikit_learn_data/)
2. Select persons with enough images:
     • 5 persons for enrollment + testing (enrolled persons)
     • 2 additional persons for "Unknown" impostor testing
3. For each enrolled person:
     • Place the first ENROLL_PER_PERSON images  → data/enrolled/<Name>/
     • Place the next  TEST_PER_PERSON images    → data/test/<Name>/
4. For "Unknown":
     • Place TEST_PER_PERSON images from each impostor → data/test/Unknown/
5. Run enrollment and print a summary.

Usage
------
  python setup_demo.py
  python setup_demo.py --skip-enroll     # only download data, do not run enrollment
  python setup_demo.py --n-persons 8     # enroll 8 persons instead of 5

After running this script, use:
  python main.py evaluate                # full evaluation
  python main.py identify --image data/test/PersonName/image.jpg --show
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.progress import track

console = Console()
log = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
ENROLL_PER_PERSON = 5    # images per person used for enrollment
TEST_PER_PERSON   = 3    # images per person used for testing
MIN_IMAGES        = ENROLL_PER_PERSON + TEST_PER_PERSON   # minimum to qualify

# ── Resolve project root (this script lives at project root) ──────────────────
ROOT_DIR     = Path(__file__).resolve().parent
ENROLLED_DIR = ROOT_DIR / "data" / "enrolled"
TEST_DIR     = ROOT_DIR / "data" / "test"


# ─────────────────────────────────────────────────────────────────────────────
#  Download & Organise LFW
# ─────────────────────────────────────────────────────────────────────────────

def download_lfw(min_faces_per_person: int = MIN_IMAGES):
    """Download LFW via scikit-learn (cached after first run)."""
    console.print("\n[bold cyan]⬇  Downloading LFW dataset via scikit-learn…[/bold cyan]")
    console.print("   (First run downloads ~250 MB to ~/scikit_learn_data/ — please wait)")

    try:
        from sklearn.datasets import fetch_lfw_people
    except ImportError:
        console.print("[red]❌ scikit-learn not installed. Run: pip install scikit-learn[/red]")
        sys.exit(1)

    # slice_=(0,250),(0,250) → full 250×250 images, color=True → RGB float32
    lfw = fetch_lfw_people(
        min_faces_per_person=min_faces_per_person,
        resize=1.0,
        color=True,
        slice_=(slice(0, 250), slice(0, 250)),
    )
    console.print(
        f"   ✅ LFW loaded: [cyan]{lfw.images.shape[0]}[/cyan] images, "
        f"[cyan]{len(lfw.target_names)}[/cyan] persons"
    )
    return lfw


def organise_data(lfw, n_enrolled: int = 5, n_impostors: int = 2) -> None:
    """
    Split the LFW data into enrolled and test directories.

    Parameters
    ----------
    lfw          : object returned by fetch_lfw_people
    n_enrolled   : number of persons to enroll
    n_impostors  : number of additional persons for Unknown test images
    """
    images       = lfw.images          # shape: (N, 250, 250, 3) — RGB float32 in [0,1]
    targets      = lfw.target          # integer label per image
    target_names = lfw.target_names    # string name per integer label

    # ── Group image indices by person ─────────────────────────────────────────
    from collections import defaultdict
    person_to_indices = defaultdict(list)
    for idx, label in enumerate(targets):
        person_to_indices[label].append(idx)

    # Sort persons by number of images (descending) for a balanced dataset
    sorted_persons = sorted(
        person_to_indices.items(),
        key=lambda kv: len(kv[1]),
        reverse=True,
    )

    enrolled_persons  = sorted_persons[:n_enrolled]
    impostor_persons  = sorted_persons[n_enrolled : n_enrolled + n_impostors]

    console.print(
        f"\n  Enrolled persons  : {[target_names[p] for p, _ in enrolled_persons]}"
    )
    console.print(
        f"  Impostor persons  : {[target_names[p] for p, _ in impostor_persons]}"
    )

    # ── Save enrolled images ──────────────────────────────────────────────────
    for label, indices in track(enrolled_persons, description="  Saving enrolled images"):
        name     = _sanitise_name(target_names[label])
        out_enroll = ENROLLED_DIR / name
        out_test   = TEST_DIR / name
        out_enroll.mkdir(parents=True, exist_ok=True)
        out_test.mkdir(parents=True,   exist_ok=True)

        for i, img_idx in enumerate(indices[:ENROLL_PER_PERSON + TEST_PER_PERSON]):
            img_rgb   = images[img_idx]                        # float32 [0,1]
            img_uint8 = (img_rgb * 255).clip(0, 255).astype(np.uint8)
            img_bgr   = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)

            if i < ENROLL_PER_PERSON:
                out_path = out_enroll / f"img_{i+1:03d}.jpg"
            else:
                out_path = out_test   / f"img_{i+1:03d}.jpg"

            cv2.imwrite(str(out_path), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # ── Save impostor images → data/test/Unknown/ ─────────────────────────────
    unknown_dir = TEST_DIR / "Unknown"
    unknown_dir.mkdir(parents=True, exist_ok=True)
    counter = 0

    for label, indices in track(impostor_persons, description="  Saving Unknown test images"):
        name = target_names[label]
        for img_idx in indices[:TEST_PER_PERSON]:
            img_rgb   = images[img_idx]
            img_uint8 = (img_rgb * 255).clip(0, 255).astype(np.uint8)
            img_bgr   = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
            out_path  = unknown_dir / f"impostor_{counter:03d}.jpg"
            cv2.imwrite(str(out_path), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            counter += 1

    console.print(
        f"\n  [green]✅ Data organised:[/green]\n"
        f"     Enrolled : {ENROLLED_DIR}\n"
        f"     Test     : {TEST_DIR}"
    )


def _sanitise_name(name: str) -> str:
    """Convert LFW name (e.g. 'George_W_Bush') to a clean folder name."""
    return name.replace(" ", "_")


# ─────────────────────────────────────────────────────────────────────────────
#  Optional: Run Enrollment
# ─────────────────────────────────────────────────────────────────────────────

def run_enrollment() -> None:
    """Run the enrollment pipeline after data is set up."""
    console.print("\n[bold cyan]🧑  Running Enrollment…[/bold cyan]")
    from src.face_utils import load_model
    from src.enroll     import build_database, save_database
    from src             import config as cfg

    model = load_model()
    db    = build_database(model)
    if db:
        save_database(db)
    else:
        console.print("[red]❌ Enrollment failed — no valid faces found.[/red]")


# ─────────────────────────────────────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Download LFW and set up demo data.")
    parser.add_argument(
        "--n-persons", type=int, default=5,
        help="Number of persons to enroll (default: 5)",
    )
    parser.add_argument(
        "--n-impostors", type=int, default=2,
        help="Number of impostor persons for Unknown testing (default: 2)",
    )
    parser.add_argument(
        "--skip-enroll", action="store_true",
        help="Only download and organise data; skip enrollment step",
    )
    args = parser.parse_args()

    console.print(Panel(
        "🚀  [bold cyan]Face Recognition System — Demo Setup[/bold cyan]\n"
        f"Downloading LFW dataset and preparing {args.n_persons} enrolled persons + "
        f"{args.n_impostors} impostors for testing.",
        expand=False,
    ))

    # Step 1: Download LFW
    min_imgs = ENROLL_PER_PERSON + TEST_PER_PERSON
    lfw = download_lfw(min_faces_per_person=min_imgs)

    # Step 2: Organise into directory structure
    organise_data(lfw, n_enrolled=args.n_persons, n_impostors=args.n_impostors)

    # Step 3: Run enrollment (unless skipped)
    if not args.skip_enroll:
        run_enrollment()

    console.print(Panel(
        "[bold green]✅ Setup complete![/bold green]\n\n"
        "Next steps:\n"
        "  [cyan]python main.py info[/cyan]              — list enrolled persons\n"
        "  [cyan]python main.py evaluate[/cyan]          — run full evaluation\n"
        "  [cyan]python main.py calibrate[/cyan]         — find optimal threshold\n"
        "  [cyan]python main.py identify -i <img>[/cyan] — identify a specific image\n"
        "  [cyan]streamlit run app.py[/cyan]              — open the web dashboard",
        expand=False,
    ))


if __name__ == "__main__":
    main()
