"""
app.py — Streamlit Web Dashboard for the Face Recognition Identification System
================================================================================
Five interactive pages accessible via the sidebar:

  📋 Dashboard    — Database overview and enrolled-person gallery
  ➕ Enroll       — Upload images to add a new person to the database
  🔍 Identify     — Upload a photo and see the identification result
  📊 Evaluate     — Run evaluation and visualise metrics
  🗑️ Manage       — Delete enrolled persons from the database

Run with:
  streamlit run app.py
"""

import io
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Make src importable
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import config as cfg
from src.enroll    import build_database, save_database, load_database
from src.face_utils import load_model, detect_and_embed, cosine_similarity, bgr_to_rgb, draw_result
from src.recognize  import identify_face, recognize_image
from src.evaluate   import (
    run_evaluation, compute_metrics,
    plot_confusion_matrix, plot_similarity_distribution, plot_far_frr_curve,
)

logging.basicConfig(level=logging.WARNING)

# ─────────────────────────────────────────────────────────────────────────────
#  Page Configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Face Recognition System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
#  Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* App background */
    .stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(255,255,255,0.04);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    /* Cards */
    .metric-card {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    .metric-card h2 { color: #a78bfa; font-size: 2.2rem; margin: 0; }
    .metric-card p  { color: #cbd5e1; margin: 4px 0 0; font-size: 0.9rem; }

    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #7c3aed, #2563eb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    /* Result boxes */
    .result-known {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid #10b981;
        border-radius: 10px;
        padding: 16px;
    }
    .result-unknown {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid #ef4444;
        border-radius: 10px;
        padding: 16px;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #2563eb);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }

    /* Hide streamlit branding */
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Cached Resources (loaded once across reruns)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading InsightFace model (first run downloads ~300 MB)…")
def get_model():
    """Load InsightFace FaceAnalysis model (cached across all sessions)."""
    return load_model()


def get_database() -> Dict[str, np.ndarray]:
    """Load the embedding database from disk (no caching — re-read on each action)."""
    if not cfg.DATABASE_PATH.exists():
        return {}
    try:
        return load_database(cfg.DATABASE_PATH)
    except Exception:
        return {}


def save_db(db: Dict[str, np.ndarray]) -> None:
    """Persist database and clear any stale cache."""
    save_database(db, cfg.DATABASE_PATH)


# ─────────────────────────────────────────────────────────────────────────────
#  Helper: Convert uploaded file → BGR numpy array
# ─────────────────────────────────────────────────────────────────────────────

def uploaded_to_bgr(file) -> Optional[np.ndarray]:
    """Convert a Streamlit UploadedFile to a BGR numpy array."""
    try:
        pil_img = Image.open(file).convert("RGB")
        img_rgb = np.array(pil_img, dtype=np.uint8)
        return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    except Exception as exc:
        st.error(f"Could not decode image: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  Sidebar Navigation
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🧠 Face Recognition")
    st.markdown("ArcFace · InsightFace · ONNX Runtime")
    st.divider()

    page = st.radio(
        "Navigate to",
        ["📋 Dashboard", "➕ Enroll", "🔍 Identify", "📊 Evaluate", "🗑️ Manage"],
        label_visibility="collapsed",
    )
    st.divider()

    db_status = cfg.DATABASE_PATH.exists()
    if db_status:
        st.success(f"✅ Database exists  \n`{cfg.DATABASE_PATH.name}`")
    else:
        st.warning("⚠️ No database yet  \nGo to **Enroll** or run `setup_demo.py`")

    st.caption(f"Threshold: **{cfg.SIMILARITY_THRESHOLD}** (edit `src/config.py`)")


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 1: Dashboard
# ─────────────────────────────────────────────────────────────────────────────

if page == "📋 Dashboard":
    st.markdown('<p class="section-header">📋 System Dashboard</p>', unsafe_allow_html=True)
    st.caption("Overview of the enrolled face database and system configuration.")

    db = get_database()

    # ── Metric Cards ──────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'<div class="metric-card"><h2>{len(db)}</h2><p>Enrolled Persons</p></div>',
            unsafe_allow_html=True,
        )
    with c2:
        emb_dim = next(iter(db.values())).shape[0] if db else 512
        st.markdown(
            f'<div class="metric-card"><h2>{emb_dim}</h2><p>Embedding Dimensions</p></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="metric-card"><h2>{cfg.SIMILARITY_THRESHOLD}</h2><p>Similarity Threshold</p></div>',
            unsafe_allow_html=True,
        )
    with c4:
        model_label = cfg.MODEL_NAME
        st.markdown(
            f'<div class="metric-card"><h2 style="font-size:1.2rem">{model_label}</h2><p>Model Pack</p></div>',
            unsafe_allow_html=True,
        )

    st.divider()

    if not db:
        st.info("No persons enrolled yet. Go to **➕ Enroll** to add people.")
    else:
        st.subheader("👥 Enrolled Persons")
        enrolled_df = pd.DataFrame([
            {"Name": name, "Embedding Dim": emb.shape[0], "L2 Norm": round(float(np.linalg.norm(emb)), 4)}
            for name, emb in sorted(db.items())
        ])
        st.dataframe(enrolled_df, use_container_width=True, hide_index=True)

    st.divider()

    # ── Evaluation Results (if CSV exists) ────────────────────────────────────
    if cfg.EVALUATION_CSV.exists():
        st.subheader("📊 Latest Evaluation Results")
        eval_df = pd.read_csv(cfg.EVALUATION_CSV)
        accuracy = eval_df["correct"].mean()
        det_rate = eval_df["face_detected"].mean()

        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy",         f"{accuracy:.2%}")
        col2.metric("Detection Rate",   f"{det_rate:.2%}")
        col3.metric("Test Images",      str(len(eval_df)))

        st.dataframe(eval_df.tail(20), use_container_width=True, hide_index=True)

    # ── Plots (if they exist) ─────────────────────────────────────────────────
    plot_files = list(cfg.PLOTS_DIR.glob("*.png")) if cfg.PLOTS_DIR.exists() else []
    if plot_files:
        st.subheader("📈 Evaluation Plots")
        cols = st.columns(min(len(plot_files), 2))
        for i, plot_path in enumerate(sorted(plot_files)):
            cols[i % 2].image(str(plot_path), caption=plot_path.stem.replace("_", " ").title(), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 2: Enroll
# ─────────────────────────────────────────────────────────────────────────────

elif page == "➕ Enroll":
    st.markdown('<p class="section-header">➕ Enroll a New Person</p>', unsafe_allow_html=True)
    st.caption(
        "Upload **multiple** photos of one person. The system extracts a face embedding from "
        "each image and stores the averaged representation in the database."
    )

    with st.form("enroll_form"):
        person_name = st.text_input("Person Name", placeholder="e.g. Alice_Smith")
        uploaded    = st.file_uploader(
            "Upload face images (JPG/PNG)",
            type=["jpg", "jpeg", "png", "bmp"],
            accept_multiple_files=True,
        )
        submitted = st.form_submit_button("🚀 Enroll Person")

    if submitted:
        # Validate inputs
        person_name = person_name.strip().replace(" ", "_")
        if not person_name:
            st.error("Please enter a valid person name.")
        elif not uploaded:
            st.error("Please upload at least one image.")
        else:
            model = get_model()
            db    = get_database()

            embeddings = []
            cols = st.columns(min(len(uploaded), 5))
            st.info(f"Processing {len(uploaded)} image(s) for **{person_name}**…")

            for i, file in enumerate(uploaded):
                img_bgr = uploaded_to_bgr(file)
                if img_bgr is None:
                    continue

                # Display thumbnail
                cols[i % len(cols)].image(
                    bgr_to_rgb(img_bgr), caption=file.name, use_container_width=True
                )

                faces = detect_and_embed(model, img_bgr)
                if faces:
                    _, emb, score = faces[0]
                    embeddings.append(emb)
                    cols[i % len(cols)].caption(f"✅ det={score:.2f}")
                else:
                    cols[i % len(cols)].caption("❌ No face")

            if not embeddings:
                st.error("No faces were detected in any of the uploaded images. "
                         "Try clearer, frontal photos with good lighting.")
            else:
                # Aggregate and save
                agg = np.mean(np.stack(embeddings), axis=0)
                agg = agg / (np.linalg.norm(agg) + 1e-10)
                db[person_name] = agg.astype(np.float32)
                save_db(db)

                st.success(
                    f"✅ **{person_name}** enrolled successfully!\n\n"
                    f"Faces used: **{len(embeddings)} / {len(uploaded)}**"
                )

                # Save images to disk as well
                person_dir = cfg.ENROLLED_DIR / person_name
                person_dir.mkdir(parents=True, exist_ok=True)
                for j, file in enumerate(uploaded):
                    img_bgr = uploaded_to_bgr(file)
                    if img_bgr is not None:
                        cv2.imwrite(str(person_dir / f"uploaded_{j+1:03d}.jpg"), img_bgr)


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 3: Identify
# ─────────────────────────────────────────────────────────────────────────────

elif page == "🔍 Identify":
    st.markdown('<p class="section-header">🔍 Identify a Face</p>', unsafe_allow_html=True)
    st.caption("Upload a photo. The system detects all faces and identifies each one.")

    # Threshold slider
    threshold = st.slider(
        "Similarity Threshold",
        min_value=0.0, max_value=1.0,
        value=cfg.SIMILARITY_THRESHOLD, step=0.01,
        help="Faces with similarity below this value are labelled Unknown.",
    )

    uploaded = st.file_uploader(
        "Upload image (JPG/PNG)", type=["jpg", "jpeg", "png", "bmp"]
    )

    if uploaded:
        db = get_database()
        if not db:
            st.error("Database is empty. Enroll some people first.")
        else:
            model   = get_model()
            img_bgr = uploaded_to_bgr(uploaded)

            if img_bgr is not None:
                results, annotated = recognize_image(model, img_bgr, db, threshold=threshold)

                col1, col2 = st.columns([1.2, 1])
                with col1:
                    st.subheader("Annotated Image")
                    st.image(bgr_to_rgb(annotated), use_container_width=True)

                with col2:
                    st.subheader("Results")
                    if not results:
                        st.warning("⚠️ No faces detected in this image.")
                    else:
                        for i, (bbox, name, sim, det) in enumerate(results, 1):
                            is_known = name != cfg.UNKNOWN_LABEL
                            css_cls  = "result-known" if is_known else "result-unknown"
                            icon     = "✅" if is_known else "❌"
                            st.markdown(
                                f'<div class="{css_cls}">'
                                f"<b>Face #{i}</b><br>"
                                f"{icon} <b>{name}</b><br>"
                                f"Similarity: <b>{sim:.4f}</b> (threshold: {threshold})<br>"
                                f"Detection score: {det:.3f}"
                                f"</div><br>",
                                unsafe_allow_html=True,
                            )

                        # Full score breakdown
                        with st.expander("🔬 Full similarity scores (all enrolled persons)"):
                            _, emb, _ = detect_and_embed(model, img_bgr)[0]
                            _, _, all_scores = identify_face(emb, db, threshold)
                            score_df = pd.DataFrame(
                                [(k, round(v, 4), "✅ Match" if v >= threshold else "❌")
                                 for k, v in sorted(all_scores.items(), key=lambda x: -x[1])],
                                columns=["Person", "Similarity", "Match?"],
                            )
                            st.dataframe(score_df, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 4: Evaluate
# ─────────────────────────────────────────────────────────────────────────────

elif page == "📊 Evaluate":
    st.markdown('<p class="section-header">📊 System Evaluation</p>', unsafe_allow_html=True)
    st.caption(
        f"Runs the recognition system on every image in `{cfg.TEST_DIR}` and "
        "computes accuracy, FAR, FRR, confusion matrix, and more."
    )

    threshold = st.slider(
        "Similarity Threshold for Evaluation",
        min_value=0.0, max_value=1.0,
        value=cfg.SIMILARITY_THRESHOLD, step=0.01,
    )

    if not cfg.TEST_DIR.exists() or not any(cfg.TEST_DIR.iterdir()):
        st.warning(
            "Test directory is empty. Run `python setup_demo.py` to populate it, "
            "or add images manually to `data/test/<PersonName>/`."
        )
    elif st.button("▶️ Run Evaluation"):
        db = get_database()
        if not db:
            st.error("Database is empty — enroll persons first.")
        else:
            model = get_model()
            cfg.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
            cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

            with st.spinner("Running evaluation on test set…"):
                df = run_evaluation(model, db, threshold=threshold)

            if df.empty:
                st.error("No results. Check test directory structure.")
            else:
                metrics = compute_metrics(df, threshold)
                df.to_csv(cfg.EVALUATION_CSV, index=False)

                # Metric grid
                st.subheader("Key Metrics")
                m1, m2, m3, m4, m5, m6 = st.columns(6)
                m1.metric("Accuracy",   f"{metrics['accuracy']:.2%}")
                m2.metric("Macro F1",   f"{metrics['macro_f1']:.3f}")
                m3.metric("FAR",        f"{metrics['FAR']:.3f}")
                m4.metric("FRR",        f"{metrics['FRR']:.3f}")
                m5.metric("TP",         metrics["TP"])
                m6.metric("TN",         metrics["TN"])

                m7, m8, m9 = st.columns(3)
                m7.metric("Macro Precision", f"{metrics['macro_precision']:.3f}")
                m8.metric("Macro Recall",    f"{metrics['macro_recall']:.3f}")
                m9.metric("Weighted F1",     f"{metrics['weighted_f1']:.3f}")

                st.subheader("Detailed Results")
                st.dataframe(df, use_container_width=True, hide_index=True)

                # Generate and display plots
                with st.spinner("Generating plots…"):
                    cm_path   = cfg.PLOTS_DIR / "confusion_matrix.png"
                    dist_path = cfg.PLOTS_DIR / "similarity_distribution.png"
                    eer_path  = cfg.PLOTS_DIR / "far_frr_curve.png"

                    plot_confusion_matrix(df, cm_path)
                    plot_similarity_distribution(df, threshold, dist_path)
                    plot_far_frr_curve(df, eer_path)

                col1, col2 = st.columns(2)
                if cm_path.exists():
                    col1.image(str(cm_path), caption="Confusion Matrix", use_container_width=True)
                if dist_path.exists():
                    col2.image(str(dist_path), caption="Score Distribution", use_container_width=True)
                if eer_path.exists():
                    st.image(str(eer_path), caption="FAR / FRR Curve (EER Analysis)", use_container_width=True)

                st.success(f"✅ Evaluation complete! CSV saved to `{cfg.EVALUATION_CSV}`")

    # Show previous results if they exist
    elif cfg.EVALUATION_CSV.exists():
        st.info("Showing last evaluation results. Click **Run Evaluation** to refresh.")
        df = pd.read_csv(cfg.EVALUATION_CSV)
        st.dataframe(df, use_container_width=True, hide_index=True)

        plot_files = list(cfg.PLOTS_DIR.glob("*.png")) if cfg.PLOTS_DIR.exists() else []
        if plot_files:
            cols = st.columns(2)
            for i, p in enumerate(sorted(plot_files)):
                cols[i % 2].image(str(p), caption=p.stem.replace("_", " ").title(), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 5: Manage
# ─────────────────────────────────────────────────────────────────────────────

elif page == "🗑️ Manage":
    st.markdown('<p class="section-header">🗑️ Manage Enrolled Persons</p>', unsafe_allow_html=True)
    st.caption("Remove a person from the face database.")

    db = get_database()

    if not db:
        st.info("No persons enrolled yet.")
    else:
        st.subheader("Enrolled Persons")
        enrolled_df = pd.DataFrame(
            [{"Name": n, "Embedding Dim": v.shape[0]} for n, v in sorted(db.items())]
        )
        st.dataframe(enrolled_df, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Delete a Person")

        person_to_delete = st.selectbox(
            "Select person to delete", options=sorted(db.keys())
        )

        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🗑️ Delete", type="primary"):
                if person_to_delete in db:
                    del db[person_to_delete]
                    save_db(db)
                    st.success(f"✅ **{person_to_delete}** removed from the database.")
                    st.rerun()

        st.divider()
        st.subheader("⚠️ Clear Entire Database")
        if st.button("🔥 Delete ALL enrolled persons"):
            confirm = st.checkbox("I confirm I want to delete all enrollments.")
            if confirm:
                save_db({})
                st.success("Database cleared.")
                st.rerun()
