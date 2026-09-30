# 🚥 DyMATES — Dynamic Multi-Agent Traffic Enforcement System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)](https://pytorch.org/)
[![YOLOv11](https://img.shields.io/badge/YOLOv11-Nano-00ffff.svg)](https://github.com/ultralytics/ultralytics)
[![ByteTrack](https://img.shields.io/badge/ByteTrack-Motion%20Tracking-green.svg)](https://github.com/ifzhang/ByteTrack)
[![PaddleOCR](https://img.shields.io/badge/PaddleOCR-v2.7-red.svg)](https://github.com/PaddlePaddle/PaddleOCR)
[![mAP@50](https://img.shields.io/badge/mAP%4050-90.4%25-brightgreen.svg)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg)](https://streamlit.io/)

**DyMATES** (Dynamic Multi-Agent Traffic Enforcement System) is a modular, real-time computer vision pipeline engineered to automate traffic violation monitoring from standard CCTV surveillance feeds. Combining fine-tuned **YOLOv11-Nano**, **ByteTrack** multi-object tracking, custom trajectory analytics, **PaddleOCR** Automatic Number Plate Recognition (ANPR), and an **SQLite** evidence backend, DyMATES detects multi-class violations simultaneously from a single camera stream.

---

## 🚀 Key Features & 4-in-1 Enforcement Concept

DyMATES replaces costly multi-sensor roadside setups with a single, intelligent vision pipeline capable of detecting four primary traffic infractions:

- 🪖 **Helmet Compliance Detection**: Identifies helmeted (`with_helmet`) vs. unhelmeted (`without_helmet`) riders on two-wheelers using spatial bounding-box association and temporal voting.
- 👥 **Triple Riding Offenses**: Performs inclusive passenger load spatial clustering to count total occupants on any motorcycle, flagging 3+ riders regardless of helmet usage.
- 🚫 **Wrong-Way Driving**: Analyzes vehicle motion trajectories using centroid vector directional angles to detect vehicles driving against the established traffic flow vector (>140° deviation).
- ⚡ **Rash & Reckless Driving**: Tracks lateral centroid variance and trajectory jitter using Exponential Moving Average (**EMA smoothing \(\alpha = 0.65\)**) to isolate dangerous swerving from normal camera motion.
- 🔤 **Dual-Crop ANPR & OCR Voting**: Extracts license plate text using PaddleOCR enhanced with bicubic 2× upscaling, Contrast Limited Adaptive Histogram Equalization (CLAHE), and a 10-frame rolling majority voting engine.
- 📩 **Evidence Logging & Anti-Spam E-Challan**: Saves snapshot evidence to an SQLite database (`violations.db`) protected by a 5-second anti-spam cooldown guard, and dispatches single consolidated HTML email alerts with HD proof inline.
- 🔒 **Privacy Blurring**: Automatically applies Gaussian blur to detected faces for privacy compliance (GDPR guidelines).

---

## 🏗️ 5-Layer System Architecture

```mermaid
flowchart TD
    A[📹 CCTV Video Stream / Image] --> L1[Layer 1: Preprocessing & Contrast Enhancement]
    L1 --> L2[Layer 2: AI Object Detection - YOLOv11-Nano]
    L2 --> L3[Layer 3: Motion Tracking - ByteTrack]
    L3 --> L4[Layer 4: Violation Logic Engine & Trajectory Analytics]
    L4 --> L5[Layer 5: Evidence & Output - ANPR OCR, SQLite & Email Alert]

    subgraph Layer 1: Image Quality
        L1 --- L1a[CLAHE + Unsharp Mask Edge Sharpening]
    end

    subgraph Layer 2: Detection
        L2 --- L2a[Motorcycle | Helmet | No-Helmet | License Plate]
    end

    subgraph Layer 3: Motion
        L3 --- L3a[Kalman Filter + Re-ID Spatial Association]
    end

    subgraph Layer 4: Rule Engine
        L4 --- L4a[Static: Helmet & Triple Riding]
        L4 --- L4b[Dynamic: Wrong-Way Angle & Rash Jitter]
    end

    subgraph Layer 5: Evidence
        L5 --- L5a[Dual-Crop PaddleOCR ANPR]
        L5 --- L5b[SQLite Database + CSV Export]
        L5 --- L5c[Consolidated SMTP E-Challan Alert]
    end
```

---

## 📊 Empirical Evaluation & Benchmarks

Fine-tuned on a consolidated multi-class traffic surveillance dataset (**6,765 images**, 5,498 train / 633 val / 634 test split), DyMATES achieved significant accuracy gains over baseline pre-trained models.

### 📈 Overall Test Set Results (`models/best.pt`)

| Evaluation Metric | Baseline Model | **DyMATES Fine-Tuned** | Net Improvement |
| :--- | :---: | :---: | :---: |
| **mAP@50 (IoU = 0.50)** | `57.46%` | **90.40%** | **$+32.94\%$** 🚀 |
| **Precision (P)** | `66.48%` | **90.00%** | **$+23.52\%$** 🚀 |
| **Recall (R)** | `51.91%` | **80.90%** | **$+28.99\%$** 🚀 |
| **F1-Score** | `58.29%` | **85.20%** | **$+26.91\%$** 🚀 |

### 🏷️ Per-Class Performance Breakdown

| Class Index | Class Label | Precision (P) | Recall (R) | **mAP@50 (%)** |
| :---: | :--- | :---: | :---: | :---: |
| `0` | `with_helmet` | 92.3% | 78.7% | **90.20%** |
| `1` | `without_helmet` | 87.7% | 83.0% | **90.50%** |
| `2` | `motorcycle` | 100.0% | — | Reinforced by ByteTrack |
| `3` | `license_plate` | 100.0% | — | Reinforced by ANPR OCR |

### ⚡ Pipeline Speed & Latency

| Environment | Inference FPS | Per-Frame Latency | Details |
| :--- | :---: | :---: | :--- |
| **Static Image Inference** | **19.6 FPS** | ~51 ms | CLAHE + YOLOv11-Nano |
| **Full Video Tracking Pipeline** | **13.9 FPS** | **71.8 ms** | Full 5-layer pipeline (1080p @ 30 FPS) |

---

## 🛠️ Technology Stack

- **Core AI & Computer Vision**: Python 3.10+, PyTorch 2.1+, Ultralytics YOLOv11-Nano, Supervision (ByteTrack), OpenCV, NumPy.
- **ANPR & OCR Engine**: PaddleOCR v2.7 / v4, CLAHE preprocessing, RegEx pattern validation.
- **Backend & Database**: SQLite3, Python `smtplib` / `email.mime`, Pandas.
- **Web Interface**: Streamlit Dashboard (`app.py`).

---

## 📁 Repository Directory Structure

```
DyMATES/
├── app.py                         # Streamlit Interactive Web Application Dashboard
├── main.py                        # Master CLI Pipeline (Video / Image Processing)
├── detector.py                    # Layer 1 & 2: Image Preprocessing + YOLOv11 + ByteTrack
├── violation_engine.py            # Layer 4: Helmet Compliance & Triple-Riding Logic Engine
├── dynamic_violations.py          # Layer 4: Wrong-Way Angle & Rash Driving Trajectory Engine
├── ocr_engine.py                  # Layer 5: PaddleOCR ANPR Engine + Multi-Frame Voting
├── database.py                    # Layer 5: SQLite Storage, Anti-Spam Guard & CSV Export
├── email_alert_engine.py          # Layer 5: SMTP E-Challan HTML Alert Dispatcher
├── models/
│   └── best.pt                    # Fine-tuned YOLOv11-Nano weight checkpoint
├── videos/                        # Test surveillance footage input directory
├── snapshots/                     # Captured infraction proof images
├── violations.db                  # Auto-generated SQLite evidence database
├── requirements.txt               # Dependencies list
└── model_evaluation_report.md     # Full empirical benchmark report
```

---

## 🔧 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/DyMATES.git
cd DyMATES
```

### 2. Create a Virtual Environment & Install Dependencies
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt
pip install paddlepaddle paddleocr streamlit
```

### 3. Verify Model Weights
Ensure fine-tuned model weights exist at `models/best.pt`. If missing, the pipeline will automatically fall back to downloading pretrained `yolo11n.pt`.

---

## 💻 Usage Guide

### Option 1: Streamlit Interactive Dashboard (Recommended)
Launch the web interface for single-click image/video analysis, live database views, and automated E-Challan dispatches:

```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

### Option 2: Command Line Interface (CLI)

#### Process Video Feed
```bash
python main.py --video videos/test_video.mp4 --save-output
```

#### Process Image Directory
```bash
python main.py --images dataset/test/images --save-output
```

#### CLI Command Options
- `--video <path>` : Path to input MP4/AVI surveillance video.
- `--images <dir>` : Path to directory of test images.
- `--model <path>` : Path to model weights (default: `models/best.pt`).
- `--save-output`  : Save annotated output video/images to `output/`.
- `--lane-direction <deg>` : Manually override dominant lane direction angle (e.g., `90` for vertical).

---

### Option 3: Run Model Evaluation & Audit Benchmarks

Run empirical evaluation scripts on held-out test datasets:
```bash
python evaluate_gpu.py           # Comprehensive PyTorch test set benchmark
python run_full_video_audit.py   # Run 5-layer pipeline audit on test surveillance video
```

---

## 📜 License & Citation

This project is released under the **MIT License**.

If you use DyMATES in your research or project, please cite:
```bibtex
@misc{dymates2026,
  author = {DyMATES Core Development Team},
  title = {DyMATES: Dynamic Multi-Agent Traffic Enforcement System},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository}
}
```
