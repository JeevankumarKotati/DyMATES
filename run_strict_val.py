"""
DyMATES - Strict Empirical PyTorch Model Evaluation
Runs ultralytics model.val() on dataset_large.yaml and dataset_prepared/data.yaml
to compute exact PyTorch-calculated mAP50, mAP50-95, Precision, Recall, and F1.
"""

import os
import json
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")

print("=" * 75)
print("  DyMATES — STRICT EMPIRICAL MODEL EVALUATION (PyTorch val engine)")
print("=" * 75)
print(f"Loading weights: {MODEL_PATH}")

if not os.path.exists(MODEL_PATH):
    print("  ⚠ Notice: models/best.pt not found. Using yolo11n.pt...")
    MODEL_PATH = "yolo11n.pt"

model = YOLO(MODEL_PATH)
print(f"Model Summary: {model.model.names}")
print("=" * 75 + "\n")

# Run PyTorch val engine on validation set
print("[1/2] Running PyTorch validation engine on dataset_large.yaml...")
try:
    val_results = model.val(data=os.path.join(BASE_DIR, "dataset_large.yaml"), split="val", plots=False, verbose=True)
    map50 = float(val_results.box.map50) * 100
    map50_95 = float(val_results.box.map) * 100
    mp = float(val_results.box.mp) * 100
    mr = float(val_results.box.mr) * 100
    f1 = (2 * mp * mr) / max(1e-6, (mp + mr))
except Exception as e:
    print(f"Validation on dataset_large.yaml failed: {e}")
    # Fallback to dataset_prepared if dataset_large fails
    val_results = model.val(data=os.path.join(BASE_DIR, "dataset_prepared", "data.yaml"), split="val", plots=False, verbose=True)
    map50 = float(val_results.box.map50) * 100
    map50_95 = float(val_results.box.map) * 100
    mp = float(val_results.box.mp) * 100
    mr = float(val_results.box.mr) * 100
    f1 = (2 * mp * mr) / max(1e-6, (mp + mr))

print("\n" + "=" * 75)
print("  EMPIRICAL PYTORCH VAL METRICS")
print("=" * 75)
print(f"  • mAP@50 (IoU=0.50):       {map50:.2f}%")
print(f"  • mAP@50-95 (IoU=0.5..0.95): {map50_95:.2f}%")
print(f"  • Mean Precision (P):       {mp:.2f}%")
print(f"  • Mean Recall (R):          {mr:.2f}%")
print(f"  • F1-Score:                 {f1:.2f}%")
print("=" * 75 + "\n")

# Class breakdown if available
if hasattr(val_results.box, "maps") and val_results.box.maps is not None:
    print("[Class Breakdown]")
    for cid, cname in model.model.names.items():
        if cid < len(val_results.box.maps):
            class_map50 = float(val_results.box.maps[cid]) * 100
            print(f"  Class {cid} ({cname}): mAP@50 = {class_map50:.2f}%")

strict_report = {
    "model_file": MODEL_PATH,
    "empirical_metrics": {
        "map50_pct": round(map50, 2),
        "map50_95_pct": round(map50_95, 2),
        "precision_pct": round(mp, 2),
        "recall_pct": round(mr, 2),
        "f1_score_pct": round(f1, 2),
    }
}

os.makedirs("output", exist_ok=True)
with open("output/strict_empirical_eval.json", "w") as f:
    json.dump(strict_report, f, indent=2)

print(f"\nSaved empirical results to output/strict_empirical_eval.json")
