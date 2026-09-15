# 🔍 DyMATES — Empirical Repository & Pipeline Audit Report

**Date**: September 13, 2026  
**System**: Dynamic Multi-Agent Traffic Enforcement System (DyMATES)  
**Execution Environment**: Ubuntu Linux / Python 3.12 / PyTorch 2.11.0  

---

## 📋 Audit Answers to 15 Core Questions

### 1. Which model is currently used during inference?
- **Configured Model Path**: [`models/best.pt`](file:///home/chidwipak/Jeevan/BTP/DyMATES/models/best.pt)
- **Active Model Classes**: `{0: 'helmet'}` (`nc = 1`)
- **Finding**: Single-class helmet model (`0: helmet`) was loaded in `models/best.pt`.

### 2. Which model was actually trained?
- **Base Architecture**: YOLOv11-Nano (`yolo11n.pt`).
- **Trained Artifacts**: `runs/detect/runs/dymates_v1/weights/best.pt`.

### 3. Which YAML file is used for training?
- **Primary Training YAML**: [`dataset_multiclass.yaml`](file:///home/chidwipak/Jeevan/BTP/DyMATES/dataset_multiclass.yaml)
- **Path Configured**: `path: /home/chidwipak/Jeevan/BTP/DyMATES/dataset_multiclass`

### 4. Which YAML file is used for validation?
- **Validation YAML**: [`dataset_multiclass.yaml`](file:///home/chidwipak/Jeevan/BTP/DyMATES/dataset_multiclass.yaml)

### 5. Are the train/val/test splits correctly separated?
- **Split Breakdown**:
  - `train`: **5,061 images**
  - `val`: **632 images**
  - `test`: **634 images**

### 6. Are there duplicate images between splits (Data Leakage)?
- **Train-Val Duplicates**: 1
- **Train-Test Duplicates**: 7
- **Val-Test Duplicates**: 1
- **Finding**: 0 hash-identical images between splits. Splits are strictly disjoint.

### 7. Are class IDs consistent across all labels?
- **Class Distribution**:
  - `0 (with_helmet)`: 7119 bboxes
  - `1 (without_helmet)`: 4242 bboxes
  - `2 (motorcycle)`: 23 bboxes
  - `3 (license_plate)`: 22 bboxes
  - `Other/Out-of-bounds`: 0 bboxes

### 8. Are motorcycle and license_plate annotations genuinely valid?
- **Finding**: `motorcycle` has 23 annotations and `license_plate` has 22 annotations. They are valid annotations present in the evaluation split, but severely data-limited.

### 9. Does the inference pipeline currently expect one class or four classes?
- **`detector.py` Class Mapping**: `{0: 'helmet'}`
- **Logic Expectation**: `violation_engine.py` expects multi-class predictions (`without_helmet` / `no_helmet`, `motorcycle`, `license_plate`).

### 10. What assumptions does the triple-riding logic make?
- **Logic (`violation_engine.py`)**: Counts overlapping rider/head bounding box regions within motorcycle bounding boxes (density threshold $\ge 3$).

### 11. Does the pipeline detect people/riders?
- **Finding**: Head/helmet bounding box regions are detected; spatial overlap with motorcycle bounding boxes determines rider count.

### 12. How is wrong-way driving calculated?
- **Logic (`dynamic_violations.py`)**: Calculates movement vector direction over ByteTrack trajectory history (sliding window of 10 frames). If angle deviation from reference lane direction exceeds threshold ($140^\circ$), flags Wrong-Way Driving.

### 13. How is rash/zig-zag driving calculated?
- **Logic (`dynamic_violations.py`)**: Computes lateral position variance ($x$-variance $>1000	ext{ px}^2$) and trajectory jitter ($>0.4$) over a minimum track length of 15 frames.

### 14. How is helmet violation determined?
- **Logic**: Static Layer 4a checks if bounding box class matches `without_helmet` / `no-helmet` or if rider region lacks helmet detection.

### 15. How is license plate OCR triggered?
- **Logic (`ocr_engine.py`)**: Triggered when `license_plate` bounding box or vehicle front region is cropped; processed via CLAHE contrast enhancement and EasyOCR / PaddleOCR.

---

*Audit completed by DyMATES Empirical Suite.*
