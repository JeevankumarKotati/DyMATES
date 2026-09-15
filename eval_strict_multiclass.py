"""
DyMATES - Strict Multi-Class Validation & Unseen Test Evaluation Engine
Runs PyTorch model.val() on both Validation and Unseen Test splits using dataset_multiclass.yaml.
Outputs:
  - output/multiclass_validation_report.json
  - output/multiclass_test_report.json
  - output/multiclass_evaluation_report.md
"""

import os
import json
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(BASE_DIR, "models", "dymates_yolov11n_multiclass_best.pt")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(BASE_DIR, "runs", "detect", "runs", "dymates_v1", "weights", "best.pt")

OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 75)
print("  DyMATES — STRICT MULTI-CLASS VALIDATION & TEST SUITE")
print("=" * 75)
print(f"Loading Model Weights: {MODEL_PATH}")

model = YOLO(MODEL_PATH)
class_names = model.model.names
print(f"Model Architecture Classes: {class_names}")
print("=" * 75 + "\n")


def evaluate_split(split_name):
    print(f"[{split_name.upper()} SPLIT] Running PyTorch model.val()...")
    yaml_path = os.path.join(BASE_DIR, "dataset_multiclass.yaml")
    
    val_res = model.val(data=yaml_path, split=split_name, batch=16, plots=False, verbose=True)
    
    map50 = float(val_res.box.map50) * 100
    map50_95 = float(val_res.box.map) * 100
    mp = float(val_res.box.mp) * 100
    mr = float(val_res.box.mr) * 100
    f1 = (2 * mp * mr) / max(1e-6, (mp + mr))

    per_class = {}
    try:
        ap_indices = list(val_res.box.ap_class_index) if hasattr(val_res.box, "ap_class_index") else []
        ap50_vals = val_res.box.ap50 if hasattr(val_res.box, "ap50") else []
        for idx, cid in enumerate(ap_indices):
            cname = class_names.get(int(cid), f"class_{cid}")
            c_map50 = float(ap50_vals[idx]) * 100 if idx < len(ap50_vals) else 0.0
            per_class[cname] = {
                "class_id": int(cid),
                "ap50_pct": round(c_map50, 2)
            }
    except Exception as e:
        print(f"Warning: Could not parse per-class metrics: {e}")
        for cid, cname in class_names.items():
            per_class[cname] = {"class_id": cid, "ap50_pct": 0.0}

    report = {
        "split_name": split_name,
        "model_file": MODEL_PATH,
        "overall_metrics": {
            "map50_pct": round(map50, 2),
            "map50_95_pct": round(map50_95, 2),
            "precision_pct": round(mp, 2),
            "recall_pct": round(mr, 2),
            "f1_score_pct": round(f1, 2)
        },
        "per_class_metrics": per_class
    }
    return report


def main():
    val_report = evaluate_split("val")
    with open(os.path.join(OUTPUT_DIR, "multiclass_validation_report.json"), "w") as f:
        json.dump(val_report, f, indent=2)

    test_report = evaluate_split("test")
    with open(os.path.join(OUTPUT_DIR, "multiclass_test_report.json"), "w") as f:
        json.dump(test_report, f, indent=2)

    # Generate Markdown Report
    md_content = f"""# 📊 DyMATES — Strict Multi-Class Evaluation Report

**Model Artifact**: `{MODEL_PATH}`  
**Dataset Architecture**: `dataset_multiclass.yaml`  
**Classes Evaluated**: `0: with_helmet`, `1: without_helmet`, `2: motorcycle`, `3: license_plate`  
**Date**: September 13, 2026  

---

## 🎯 1. Held-Out Unseen Test Set Performance

| Metric Parameter | Measured Value (%) | Metric Description |
| :--- | :---: | :--- |
| **mAP@50 (IoU = 0.50)** | **{test_report['overall_metrics']['map50_pct']}%** | Mean Average Precision at 0.50 IoU threshold across all 4 classes |
| **mAP@50-95 (IoU 0.50..0.95)** | **{test_report['overall_metrics']['map50_95_pct']}%** | Mean Average Precision across 0.50 to 0.95 IoU thresholds |
| **Precision (P)** | **{test_report['overall_metrics']['precision_pct']}%** | Percentage of total bounding box predictions that were true positives |
| **Recall (R)** | **{test_report['overall_metrics']['recall_pct']}%** | Percentage of total ground-truth violation instances detected |
| **F1-Score** | **{test_report['overall_metrics']['f1_score_pct']}%** | Harmonic mean of Precision and Recall |

---

## 🏷️ 2. Per-Class Performance Breakdown (Unseen Test Set)

| Class ID | Class Name | Target Infraction | AP@50 Accuracy (%) | Data Limitation Status |
| :---: | :--- | :--- | :---: | :--- |
| `0` | **`with_helmet`** | Compliant Rider | **{test_report['per_class_metrics'].get('with_helmet', {}).get('ap50_pct', 0.0)}%** | Well-represented (7,119 BBoxes) |
| `1` | **`without_helmet`** | Helmet Violation Alert | **{test_report['per_class_metrics'].get('without_helmet', {}).get('ap50_pct', 0.0)}%** | Well-represented (4,242 BBoxes) |
| `2` | **`motorcycle`** | Triple Riding Analysis | **{test_report['per_class_metrics'].get('motorcycle', {}).get('ap50_pct', 0.0)}%** | Data-Limited (23 BBoxes total) |
| `3` | **`license_plate`** | ANPR Region Crop | **{test_report['per_class_metrics'].get('license_plate', {}).get('ap50_pct', 0.0)}%** | Data-Limited (22 BBoxes total) |

---

## 🔍 3. Validation Set Performance vs Test Set

| Metric Parameter | Validation Split (632 Images) | Unseen Test Split (634 Images) | Delta |
| :--- | :---: | :---: | :---: |
| **mAP@50** | **{val_report['overall_metrics']['map50_pct']}%** | **{test_report['overall_metrics']['map50_pct']}%** | `{test_report['overall_metrics']['map50_pct'] - val_report['overall_metrics']['map50_pct']:+.2f}%` |
| **Precision (P)** | **{val_report['overall_metrics']['precision_pct']}%** | **{test_report['overall_metrics']['precision_pct']}%** | `{test_report['overall_metrics']['precision_pct'] - val_report['overall_metrics']['precision_pct']:+.2f}%` |
| **Recall (R)** | **{val_report['overall_metrics']['recall_pct']}%** | **{test_report['overall_metrics']['recall_pct']}%** | `{test_report['overall_metrics']['recall_pct'] - val_report['overall_metrics']['recall_pct']:+.2f}%` |

---

*Report generated by DyMATES Evaluation Suite.*
"""

    md_path = os.path.join(OUTPUT_DIR, "multiclass_evaluation_report.md")
    with open(md_path, "w") as f:
        f.write(md_content)

    print("\n" + "=" * 75)
    print("  EVALUATION COMPLETE!")
    print(f"  • Val JSON:  {os.path.join(OUTPUT_DIR, 'multiclass_validation_report.json')}")
    print(f"  • Test JSON: {os.path.join(OUTPUT_DIR, 'multiclass_test_report.json')}")
    print(f"  • MD Report: {md_path}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
