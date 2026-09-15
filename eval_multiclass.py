"""
DyMATES - Unified 4-Class Model Evaluation Suite
Runs PyTorch model.val() on dataset_multiclass.yaml to evaluate all 4 classes:
  0: with_helmet
  1: without_helmet
  2: motorcycle
  3: license_plate
"""

import os
import json
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")

print("=" * 75)
print("  DyMATES — UNIFIED 4-CLASS EVALUATION BENCHMARK")
print("=" * 75)
print(f"Target Model: {MODEL_PATH}")

if not os.path.exists(MODEL_PATH):
    print("  Notice: models/best.pt not yet present. Using yolo11n.pt as base...")
    MODEL_PATH = "yolo11n.pt"

model = YOLO(MODEL_PATH)
print(f"Model Summary: {model.model.names}")
print("=" * 75 + "\n")

print("[1/2] Running PyTorch model.val() on dataset_multiclass.yaml...")
try:
    val_res = model.val(data=os.path.join(BASE_DIR, "dataset_multiclass.yaml"), split="val", plots=False, verbose=True)
    map50 = float(val_res.box.map50) * 100
    map50_95 = float(val_res.box.map) * 100
    mp = float(val_res.box.mp) * 100
    mr = float(val_res.box.mr) * 100
    f1 = (2 * mp * mr) / max(1e-6, (mp + mr))
except Exception as e:
    print(f"Validation failed: {e}")
    map50, map50_95, mp, mr, f1 = 0.0, 0.0, 0.0, 0.0, 0.0

print("\n" + "=" * 75)
print("  UNIFIED 4-CLASS EMPIRICAL METRICS")
print("=" * 75)
print(f"  • mAP@50 (IoU=0.50):       {map50:.2f}%")
print(f"  • mAP@50-95 (IoU=0.5..0.95): {map50_95:.2f}%")
print(f"  • Mean Precision (P):       {mp:.2f}%")
print(f"  • Mean Recall (R):          {mr:.2f}%")
print(f"  • F1-Score:                 {f1:.2f}%")
print("=" * 75 + "\n")

report = {
    "model_path": MODEL_PATH,
    "dataset_yaml": "dataset_multiclass.yaml",
    "total_matched_pairs": 6327,
    "training_pairs": 5061,
    "validation_pairs": 632,
    "test_pairs": 634,
    "metrics": {
        "map50_pct": round(map50, 2),
        "map50_95_pct": round(map50_95, 2),
        "precision_pct": round(mp, 2),
        "recall_pct": round(mr, 2),
        "f1_score_pct": round(f1, 2),
    }
}

os.makedirs("output", exist_ok=True)
out_json = os.path.join(BASE_DIR, "output", "multiclass_eval.json")
with open(out_json, "w") as f:
    json.dump(report, f, indent=2)

print(f"Results exported to {out_json}")
