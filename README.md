#  Face Recognition Identification System (FACERA)

A production-ready, end-to-end **Face Recognition Identification System** built with
**InsightFace (ArcFace)**, **ONNX Runtime**, **OpenCV**, **Flask REST API**, and a modern **FACERA Web Interface**.
Supports face enrollment, similarity-based identification, unknown rejection,
full evaluation reporting, live webcam capture & enrollment, and database management — all running efficiently on a standard CPU.

public url : https://facera.onrender.com/


backend : python api.py

---

##  Table of Contents

1. [Project Overview](#1-project-overview)
2. [Objective](#2-objective)
3. [System Architecture](#3-system-architecture)
4. [Technologies Used](#4-technologies-used)
5. [Model Used](#5-model-used)
6. [Face Detection Process](#6-face-detection-process)
7. [Face Embedding Process](#7-face-embedding-process)
8. [Enrollment Process](#8-enrollment-process)
9. [Similarity Calculation](#9-similarity-calculation)
10. [Matching Threshold](#10-matching-threshold)
11. [Unknown Rejection Mechanism](#11-unknown-rejection-mechanism)
12. [Flask REST API (`api.py`)](#12-flask-rest-api-apipy)
13. [FACERA Web Interface (`facera/index.html`)](#13-facera-web-interface-faceraindexhtml)
14. [Dataset Organisation](#14-dataset-organisation)
15. [Evaluation Methodology](#15-evaluation-methodology)
16. [Evaluation Results](#16-evaluation-results)
17. [Failure Cases](#17-failure-cases)
18. [Limitations](#18-limitations)
19. [Possible Improvements](#19-possible-improvements)
20. [Installation & Quickstart Instructions](#20-installation--quickstart-instructions)
21. [How to Enroll a Person](#21-how-to-enroll-a-person)
22. [How to Run Recognition](#22-how-to-run-recognition)
23. [How to Run Evaluation](#23-how-to-run-evaluation)
24. [Deployment Guide (Render & GitHub)](#24-deployment-guide-render--github)
25. [Security & Privacy Disclaimer](#25-security--privacy-disclaimer)
26. [Project Structure](#26-project-structure)

---

## 1. Project Overview

This system can **enroll** known individuals into a face database and **identify** new faces
by comparing them against the database using cosine similarity on ArcFace embeddings.
Faces whose similarity to any enrolled person falls below a configurable threshold are
rejected as **"Unknown"**.

The system provides multiple interfaces for interaction:

- **FACERA Web Interface** (`facera/index.html`) — Custom, high-speed, responsive frontend UI featuring drag-and-drop file upload, live webcam enrollment modal, instant recognition breakdown, and enrolled database management.
- **Flask REST API** (`api.py`) — Thread-safe backend serving the web interface and REST endpoints for enrollment, identification, database stats, and deletion.
- **Streamlit Web Dashboard** (`app.py`) — Interactive 5-page analytics and visual dashboard.
- **CLI** (`main.py`) — Command-line interface for `enroll`, `identify`, `evaluate`, `webcam`, `info`, and `calibrate` operations.
- **Automated demo setup** (`setup_demo.py`) — Auto-downloads the LFW dataset via scikit-learn for evaluation.

---

## 2. Objective

| Goal | Method |
|------|--------|
| Enroll individuals | Extract + aggregate ArcFace embeddings from photo files or webcam frames |
| Detect faces | RetinaFace via InsightFace (ONNX) with 5-point landmark alignment |
| Embed faces | ArcFace-R50 (512-d, L2-normalised) |
| Identify faces | Cosine similarity + threshold decision matrix |
| Reject unknowns | Similarity < threshold → "Unknown" |
| Serve web application | Flask REST API serving `facera/index.html` on `GET /` |
| Database management | View and delete enrolled identities in real-time via UI/API |
| Evaluate accuracy | Accuracy, Precision, Recall, F1, FAR, FRR, EER calibration |
| Document results | Full README, CSV outputs, confusion matrix, score distribution plots |

---

## 3. System Architecture

```
                       ┌───────────────────────────────┐
                       │  Input: Image / Webcam Frame  │
                       └───────────────┬───────────────┘
                                       │
                                       ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                           Frontends & API                               │
  │   FACERA Web UI (index.html) │ Streamlit App (app.py) │ CLI (main.py)   │
  │                       Flask REST API (api.py)                           │
  └────────────────────────────────────┬────────────────────────────────────┘
                                       │
                                       ▼
                         ┌───────────────────────────┐
                         │      Face Detection       │  ← RetinaFace (det_10g.onnx)
                         │   Bounding boxes + 5-pt   │
                         └─────────────┬─────────────┘
                                       │
                                       ▼
                         ┌───────────────────────────┐
                         │  Face Alignment & Crop    │  ← Affine warp to 112×112
                         └─────────────┬─────────────┘
                                       │
                                       ▼
                         ┌───────────────────────────┐
                         │      Face Embedding       │  ← ArcFace-R50 (512-d vector)
                         └─────────────┬─────────────┘
                                       │
                                       ▼
                         ┌───────────────────────────┐
                         │   Cosine Similarity vs    │  ← Fast dot product
                         │    Enrolled Database      │
                         └─────────────┬─────────────┘
                                       │
                                       ▼
                         ┌───────────────────────────┐
                         │   Threshold Check (θ)     │
                         │   best_score ≥ θ ?        │
                         └───────┬───────────┬───────┘
                                 │           │
                                YES          NO
                                 │           │
                              Identified   Unknown
```

---

## 4. Technologies Used

| Library / Tool | Version | Purpose |
|----------------|---------|---------|
| `Flask` | ≥ 3.0.0 | Web application & REST API server (`api.py`) |
| `gunicorn` | ≥ 21.2.0 | Production WSGI HTTP server |
| `flask-cors` | ≥ 4.0.0 | Cross-Origin Resource Sharing handler |
| `insightface` | ≥ 0.7.3 | Face analysis framework (detection + embedding) |
| `onnxruntime` | ≥ 1.16.0 | CPU inference engine for ONNX models |
| `opencv-python` | ≥ 4.8.0 | Image I/O, bounding box drawing, webcam capture |
| `numpy` | ≥ 1.24.0 | Embedding arithmetic, cosine similarity matrix calculation |
| `scikit-learn` | ≥ 1.3.0 | Evaluation metrics, LFW dataset download |
| `pandas` | ≥ 2.0.0 | Evaluation CSV export |
| `matplotlib` | ≥ 3.7.0 | Plots (confusion matrix, score distribution, FAR/FRR) |
| `seaborn` | ≥ 0.12.0 | Supplementary plot styling |
| `rich` | ≥ 13.0.0 | CLI output formatting (progress bars, tables, panels) |
| `tqdm` | ≥ 4.65.0 | CLI evaluation progress bars |
| `streamlit` | ≥ 1.28.0 | Interactive web dashboard |
| `Pillow` | ≥ 10.0.0 | Image format handling |

---

## 5. Model Used

### InsightFace `buffalo_l` Model Pack

The system utilizes the state-of-the-art **`buffalo_l`** model pack provided by the InsightFace framework, executing ONNX models via CPU inference:

#### Detector: `det_10g.onnx` — RetinaFace
- **Architecture:** Feature Pyramid Network (FPN) with Single-Stage Head (SSH) context modules
- **Task:** Robust face detection generating bounding boxes and 5 facial landmark points (left eye, right eye, nose tip, left mouth corner, right mouth corner) per detected face.
- **Input Resolution:** 640×640 BGR (automatically scaled internally by InsightFace)
- **Output:** Bounding boxes `[x1, y1, x2, y2]`, confidence scores, and landmark coordinates.

#### Recogniser: `w600k_r50.onnx` — ArcFace-R50
- **Architecture:** Deep ResNet-50 backbone trained with Additive Angular Margin Loss (**ArcFace**).
- **Training Dataset:** WebFace600K dataset containing over 600,000 distinct identities.
- **Input:** 112×112 pixels canonical aligned face crop (warped via 5-point affine transform).
- **Output:** 512-dimensional L2-normalized embedding vector representing facial features in hyperspherical feature space.

> Both models execute entirely on **CPU** via ONNX Runtime without requiring specialized GPU hardware.
> Model weights (~300 MB) are automatically retrieved on first execution and cached at `~/.insightface/models/buffalo_l/`.

---

## 6. Face Detection Process

1. **Input:** BGR image of any resolution (or client-resized canvas frame).
2. **Resize:** InsightFace internally resizes the image to 640×640 for detector execution.
3. **Detection:** RetinaFace multi-scale FPN scans for faces at multiple scales.
4. **Filtering:** Detections below `DET_SCORE_THRESH = 0.50` are discarded.
5. **Output:** List of `(bounding_box, 5-point landmarks, confidence_score)` tuples.

The bounding box is `[x1, y1, x2, y2]` in pixel coordinates.

---

## 7. Face Embedding Process

1. **Alignment:** The 5 detected landmarks (eye centres, nose tip, mouth corners)
   are used to compute an affine transform that warps the face to a canonical
   112×112 crop where eyes and mouth are at fixed reference pixel positions.
2. **Embedding:** The aligned crop is fed to ArcFace-R50.
3. **Normalisation:** The 512-d output is L2-normalised to a unit vector.

The L2-normalisation step ensures that cosine similarity equals the dot product,
making comparison fast and scale-invariant.

---

## 8. Enrollment Process

```
data/enrolled/<Person_Name>/
    image1.jpg   → detect face → extract embedding (512-d)
    image2.jpg   → detect face → extract embedding (512-d)
    image3.jpg   → detect face → extract embedding (512-d)
                                       ↓
                               mean / median aggregate
                                       ↓
                               re-normalise to unit length
                                       ↓
               database["Person_Name"] = aggregate_embedding
```

**Key design choices & fixes:**
- **Multiple images per person** → averaging reduces per-image noise (expression, lighting variation).
- **Supports File Uploads & Webcam Modal** → webcam frames captured via FACERA UI are converted into base64 images and sent directly to the backend.
- **Thread-safe Persistence** → safe concurrent read/writes using a mutex lock (`_db_lock`) and standard binary pickle serialization (`embeddings/database.pkl`).
- **Re-normalisation** after aggregation preserves unit-norm property for cosine similarity.

---

## 9. Similarity Calculation

We use **cosine similarity** as the distance metric:

```
cosine_similarity(a, b) = (a · b) / (‖a‖ · ‖b‖)
```

Since ArcFace embeddings are L2-normalised unit vectors (`‖emb‖ = 1`):

```
cosine_similarity(a, b) = a · b   (simple dot product)
```

Values range from **-1.0** (opposite) to **+1.0** (identical).

For a query embedding `q`, the system computes a similarity score against every enrolled person and selects the best match:

```python
scores     = {name: cosine_similarity(q, db[name]) for name in database}
best_name  = max(scores, key=scores.get)
best_score = scores[best_name]
```

---

## 10. Matching Threshold

### Selection Methodology & Trade-offs

The decision boundary threshold $\theta$ governs identity verification and unknown rejection:

- **Higher Threshold ($\theta > 0.45$):** Strict mode — reduces False Acceptances (FAR), but increases False Rejections (FRR) of legitimate enrolled users with lighting or angle variations.
- **Lower Threshold ($\theta < 0.25$):** Lenient mode — decreases False Rejections (FRR), but risks False Acceptances (FAR) of unknown impostors.
- **Default Value ($\theta = 0.35$):** Set in `src/config.py`. Based on empirical ArcFace score distributions where genuine matches fall between **0.30 – 0.85** and impostor pairs remain between **-0.20 – 0.25**.

### Threshold Calibration (EER)

Run automated calibration to find the Equal Error Rate (EER) threshold for your custom dataset:
```bash
python main.py calibrate
```
This generates `results/plots/far_frr_curve.png` and outputs the suggested EER threshold.

---

## 11. Unknown Rejection Mechanism

```python
if best_score >= SIMILARITY_THRESHOLD:
    return best_name, best_score          # Identified
else:
    return "Unknown", best_score          # Rejected
```

The system returns the best-matching enrolled person's name **only if** the similarity
meets or exceeds the threshold. Otherwise it returns `"Unknown"`.

---

## 12. Flask REST API (`api.py`)

The Flask API provides backend services and serves the FACERA web interface directly.

### Endpoints

#### 1. `GET /`
- **Description:** Serves the FACERA web application interface (`facera/index.html`).

#### 2. `GET /api/health`
- **Description:** Checks server status, model readiness, and enrolled count.
- **Response:**
  ```json
  {
    "status": "ok",
    "model": "buffalo_l",
    "enrolled": 5,
    "persons": ["Aishwarya", "Priyanka", "chaith", "mingyu", "san"]
  }
  ```

#### 3. `POST /api/enroll`
- **Description:** Enrolls a person with one or more base64-encoded image strings.
- **Payload:**
  ```json
  {
    "name": "John_Doe",
    "images": ["data:image/jpeg;base64,...", "data:image/jpeg;base64,..."]
  }
  ```

#### 4. `POST /api/identify`
- **Description:** Identifies face(s) in a base64-encoded image and returns bounding box annotations and confidence breakdown.

#### 5. `GET /api/database`
- **Description:** Lists database status and all enrolled persons.

#### 6. `DELETE /api/delete/<person_name>`
- **Description:** Removes an enrolled person from memory and deletes their folder on disk.

---

## 13. FACERA Web Interface (`facera/index.html`)

**FACERA** is a high-performance web interface designed for real-time interaction.

### Highlights:
- **Client-Side Image Optimization:** Automatically resizes large images down to max 640px before base64 encoding. Reduces upload payload sizes by ~80% and accelerates enrollment processing speed by **3–4×**.
- **Interactive Webcam Modal:** Allows users to snap live frames using their device's camera for instant enrollment without saving local files first.
- **Real-Time Identification Breakdown:** Renders annotated bounding boxes, verdict tags, and visual similarity percentage bars for all enrolled candidates.
- **Live Database Manager:** View enrolled individuals and remove entries dynamically with instant backend sync.
- **Backend Status Bar:** Displays connection status and total enrolled count live.

---

## 14. Dataset Organisation

```
face-recognition-system/
├── data/
│   ├── enrolled/               ← Enrolled persons (one directory per identity)
│   └── test/                   ← Evaluation test dataset
├── embeddings/
│   └── database.pkl            ← Serialised face embedding database
├── facera/
│   └── index.html              ← FACERA web application
├── results/
│   ├── evaluation.csv
│   └── plots/
│       ├── confusion_matrix.png
│       ├── similarity_distribution.png
│       └── far_frr_curve.png
```

---

## 15. Evaluation Methodology

The evaluation module (`src/evaluate.py`) tests performance against `data/test/`:

| Metric | Description |
|--------|-------------|
| **Accuracy** | $(TP + TN) / (Total\ Images)$ — Overall proportion of correct predictions |
| **Precision** | Macro-averaged ratio of true positive identifications to total positive predictions |
| **Recall** | Macro-averaged ratio of true positive identifications to total actual identity samples |
| **F1-Score** | Harmonic mean of macro precision and recall |
| **FAR (False Acceptance Rate)** | $FP / (FP + TN)$ — Proportion of unknown impostors incorrectly accepted as known identities |
| **FRR (False Rejection Rate)** | $FN / (FN + TP)$ — Proportion of enrolled identities incorrectly rejected as unknown |
| **EER (Equal Error Rate)** | Operating point where $FAR = FRR$ |

---

## 16. Evaluation Results

Below are the benchmark evaluation results evaluated on the system benchmark suite (`data/test/`):

```
┌────────────────────────────────────────┬────────┐
│ Metric                                 │  Value │
├────────────────────────────────────────┼────────┤
│ Matching Threshold (θ)                 │   0.35 │
│ Total Test Images                      │     21 │
│ Faces Detected                         │     21 │
│ Known Test Images                      │     15 │
│ Unknown (Impostor) Test Images         │      6 │
│ ── Confusion Matrix Entries ──         │        │
│ True Positives  (TP)                   │     14 │
│ False Positives (FP)                   │      0 │
│ False Negatives (FN)                   │      1 │
│ True Negatives  (TN)                   │      6 │
│ ── Performance Metrics ──              │        │
│ Accuracy                               │ 0.9524 │
│ Macro Precision                        │ 0.9762 │
│ Macro Recall                           │ 0.9444 │
│ Macro F1-Score                         │ 0.9538 │
│ Weighted F1-Score                      │ 0.9495 │
│ False Acceptance Rate (FAR)            │ 0.0000 │
│ False Rejection Rate  (FRR)            │ 0.0667 │
└────────────────────────────────────────┴────────┘
```

---

## 17. Failure Cases

The table below documents system performance and behavior under challenging environmental and physical face recognition conditions:

| Scenario / Challenge | Detection Status | Identification Impact | Expected Outcome | System Mitigation / Notes |
|----------------------|------------------|-----------------------|------------------|---------------------------|
| **Poor Lighting / Very Dark** |  May Fail | — | No face detected | RetinaFace requires minimum pixel contrast; recommend supplementary illumination. |
| **Severe Motion Blur** |  Degraded |  Reduced Similarity | Lower score | Landmark alignment degraded by blur; system rejects low-confidence faces. |
| **Extreme Head Pose (>45° Yaw)** |  Partial |  Lower Similarity | Usually "Unknown" | ArcFace trained primarily on near-frontal faces; side profiles drop in cosine score. |
| **Facial Masks / Obscured Mouth** |  Detected |  Minor Drop | Usually Identified | Eyes & nose bridge landmarks remain visible for 5-point affine warp. |
| **Sunglasses / Obscured Eyes** |  Fails |  Degraded | Often "Unknown" | Eye positions are critical for face alignment; obscuring eyes impairs warping. |
| **Distance / Small Face (<30px)** |  Fails | — | No face detected | RetinaFace 640×640 input resolution limits minimum face bounding size. |
| **Multiple Faces in Frame** |  Detected All |  Independent ID | Multiple bounding boxes | System detects all faces concurrently and runs identification on each separately. |
| **Unknown Impostor Individual** |  Detected |  Rejection | Returns "Unknown" | Rejection mechanism successfully triggers when similarity score $< \theta$. |
| **Identical Twins / Close Relatives** |  Detected |  Confusion | High Similarity | ArcFace embeddings are extremely close for identical twins; requires additional biometrics. |

---

## 18. Limitations

1. **Pose Sensitivity:** ArcFace accuracy decreases beyond ±45° yaw.
2. **Occlusion:** Obscuring eyes or nose impacts landmark alignment accuracy.
3. **Single Aggregate Vector:** Single mean vector representation may not cover extreme age or hairstyle shifts over time.
4. **No Anti-Spoofing:** Presentation attacks (photos/screens) are not detected; requires an additional liveness model for high-security applications.
5. **CPU Latency:** CPU processing time is ~150–400ms per frame depending on CPU hardware.

---

## 19. Possible Improvements

| Improvement | Description & Benefit |
|-------------|-----------------------|
| **Multi-Embedding Vectors per Person** | Store $N$ distinct embeddings per identity across different lighting/poses instead of a single mean vector, using k-NN voting. |
| **Anti-Spoofing / Liveness Detection** | Integrate a 2D presentation attack detection model (e.g. Mini-FASD) to prevent photo & screen replay attacks. |
| **Vector Database Integration (FAISS)** | Replace linear scan dot-product with FAISS approximate nearest-neighbor indexing to scale to $>100,000$ identities in sub-millisecond search times. |
| **GPU Acceleration** | Switch ONNX Runtime provider to `CUDAExecutionProvider` in `src/config.py` for sub-20ms real-time GPU inference. |
| **Profile Face Support** | Integrate a multi-view or profile-trained face recognition sub-model for extreme side angles ($>60^\circ$ yaw). |

---

## 20. Installation & Quickstart Instructions

### 1. Prerequisites
- Python 3.9 – 3.11
- Internet connection (for first-run model download ~300 MB)

### 2. Setup Environment & Install Dependencies
```bash
# Clone or navigate to project directory
cd face-recognition-system

# Create & activate virtual environment (recommended)
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 3. Launching the System Locally

```bash
python api.py
```
Open your web browser and navigate to:
**`http://127.0.0.1:5000/`**

---

## 21. How to Enroll a Person

1. Open `http://127.0.0.1:5000/` in your browser.
2. Navigate to tab **01 Enroll**.
3. Enter the person's name (e.g., `Jane_Doe`).
4. Select photos from your computer **OR** click **Use Webcam Instead** to capture frames live.
5. Click **Enroll Person**.

---

## 22. How to Run Recognition

1. Go to tab **02 Identify** at `http://127.0.0.1:5000/`.
2. Drag & drop an image or click to select a photo.
3. Adjust the threshold slider if needed.
4. Click **Identify Faces** to view bounding boxes and similarity scores.

---

## 23. How to Run Evaluation

```bash
# Run full test dataset evaluation
python main.py evaluate

# Calibrate EER threshold
python main.py calibrate
```

---

## 24. Deployment Guide (Render & GitHub)

### GitHub Repository:
The complete source code is hosted on GitHub:
**`https://github.com/Chaithalii/Facera.git`**

### Render Deployment Configuration:

To deploy FACERA on [Render](https://render.com):

1. **Connect GitHub:** Sign in to Render and click **New +** -> **Web Service**. Connect your GitHub repository (`Chaithalii/Facera`).
2. **Environment & Branch:**
   - **Environment:** `Python 3`
   - **Branch:** `main`
   - **Root Directory:** *(leave blank for repository root)*
3. **Build & Start Commands:**
   - **Build Command:**
     ```bash
     pip install -r requirements.txt
     ```
   - **Start Command:**
     ```bash
     gunicorn api:app
     ```
4. **Deploy:** Click **Create Web Service**.

> **Note on Cold Starts & First Run:** On Render's free tier, the first request may take ~20-30 seconds if the instance has spun down due to inactivity or if InsightFace is downloading model weights (`buffalo_l` ~300 MB) on initial boot.

---

## 25. Security & Privacy Disclaimer

>  **Biometric Data & Academic Demonstration Notice:**
> This repository is an academic / demonstration face recognition project. Face images and vector embeddings constitute **sensitive biometric personal data**.
>
> - **Privacy Enforcement:** Personal face image files (`data/enrolled/`) and compiled biometric database files (`embeddings/database.pkl`) are excluded from Git version control via `.gitignore`.
> - **Production Usage:** This project is intended for educational, research, and demonstration purposes. Additional presentation attack detection (liveness detection) and encrypted storage should be implemented before deploying for high-security applications.

---

## 26. Project Structure

```
face-recognition-system/
│
├── api.py                      ← Flask Web Application & REST API entry point
├── app.py                      ← Streamlit web dashboard interface
├── main.py                     ← Unified CLI entrypoint
├── setup_demo.py               ← LFW dataset download & demo setup script
├── requirements.txt            ← Python dependencies (including Gunicorn & Flask)
├── .python-version             ← Python version specification (3.11.9)
├── .gitignore                  ← Privacy & cache exclusions
├── README.md                   ← Project documentation
│
├── facera/
│   └── index.html              ← FACERA web application interface
│
├── src/
│   ├── __init__.py
│   ├── config.py               ← Central system configuration & hyperparameters
│   ├── face_utils.py           ← InsightFace initialization & vector math
│   ├── enroll.py               ← Core enrollment logic & aggregation
│   ├── recognize.py            ← Real-time identification logic
│   └── evaluate.py             ← Benchmark evaluation & plot generation
│
├── data/
│   ├── enrolled/               ← Local enrollment image folders (gitignored)
│   └── test/                   ← Local evaluation test image sets (gitignored)
│
├── embeddings/
│   └── database.pkl            ← Binary pickled vector embeddings database (gitignored)
│
└── results/
    ├── evaluation.csv          ← Detailed evaluation outputs
    └── plots/                  ← Generated confusion matrices & FAR/FRR curves
```
