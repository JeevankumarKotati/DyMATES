"""
DyMATES - Model Evaluation & Metrics Script
============================================

Generates evaluation metrics, plots, and statistics for the presentation:
  - mAP50, mAP50-95, Precision, Recall per class
  - Confusion matrix
  - Per-class performance table
  - FPS benchmarking
  - Detection samples with annotations

Usage:
    python evaluate.py                                   # evaluate best model
    python evaluate.py --model models/best.pt            # specific model
    python evaluate.py --benchmark                       # run FPS benchmark
    python evaluate.py --samples 10                      # save 10 sample images
"""

import argparse
import os
import sys
import time
import json
import glob
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] ultralytics not installed. Run: pip install ultralytics")
    sys.exit(1)


def find_model():
    """Find the best available model."""
    candidates = [
        "models/best.pt",
        "runs/dymates_v1/weights/best.pt",
        "runs/detect/dymates_v1/weights/best.pt",
        "runs/detect/runs/dymates_v1/weights/best.pt",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    
    # Recursive search
    for p in Path("runs").rglob("best.pt"):
        return str(p)
    
    return "yolo11n.pt"


def evaluate_model(args):
    """Run YOLO validation and collect metrics."""
    model_path = args.model if os.path.exists(args.model) else find_model()
    
    print(f"\n{'='*65}")
    print(f"  DyMATES — Model Evaluation")
    print(f"{'='*65}")
    print(f"  Model:    {model_path}")
    print(f"  Dataset:  {args.data}")
    print(f"{'='*65}\n")

    model = YOLO(model_path)
    class_names = model.model.names
    print(f"[Eval] Classes: {class_names}")
    print(f"[Eval] Number of classes: {len(class_names)}\n")

    # ── Run Validation ────────────────────────────────────────────
    print("[Eval] Running validation on test/validation set...\n")
    
    try:
        results = model.val(
            data=args.data,
            imgsz=640,
            batch=args.batch,
            device=args.device,
            plots=True,           # Generate confusion matrix, PR curves
            save_json=True,       # Save COCO-format results
            verbose=True,
        )
        
        # Extract metrics
        metrics = {
            "mAP50": float(results.box.map50) if hasattr(results.box, 'map50') else 0,
            "mAP50_95": float(results.box.map) if hasattr(results.box, 'map') else 0,
            "precision": float(results.box.mp) if hasattr(results.box, 'mp') else 0,
            "recall": float(results.box.mr) if hasattr(results.box, 'mr') else 0,
        }
        
        # Per-class metrics
        per_class = {}
        if hasattr(results.box, 'ap50'):
            for i, cls_name in class_names.items():
                if i < len(results.box.ap50):
                    per_class[cls_name] = {
                        "AP50": float(results.box.ap50[i]),
                        "AP50_95": float(results.box.ap[i]) if i < len(results.box.ap) else 0,
                    }
        
        # Print metrics
        print(f"\n{'='*65}")
        print(f"  DyMATES — Evaluation Results")
        print(f"{'='*65}")
        print(f"  Overall Metrics:")
        print(f"    mAP@50:      {metrics['mAP50']:.4f} ({metrics['mAP50']*100:.1f}%)")
        print(f"    mAP@50-95:   {metrics['mAP50_95']:.4f} ({metrics['mAP50_95']*100:.1f}%)")
        print(f"    Precision:   {metrics['precision']:.4f} ({metrics['precision']*100:.1f}%)")
        print(f"    Recall:      {metrics['recall']:.4f} ({metrics['recall']*100:.1f}%)")
        print(f"\n  Per-Class Performance:")
        print(f"  {'Class':<20} {'AP@50':>10} {'AP@50-95':>10}")
        print(f"  {'-'*42}")
        for cls_name, cls_metrics in per_class.items():
            print(f"  {cls_name:<20} {cls_metrics['AP50']*100:>9.1f}% {cls_metrics['AP50_95']*100:>9.1f}%")
        print(f"{'='*65}\n")
        
        # Save metrics to JSON
        os.makedirs("output", exist_ok=True)
        metrics_path = "output/evaluation_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump({
                "model": model_path,
                "timestamp": datetime.now().isoformat(),
                "classes": class_names,
                "overall": metrics,
                "per_class": per_class,
            }, f, indent=2)
        print(f"[Eval] Metrics saved to: {metrics_path}")
        
    except Exception as e:
        print(f"[Eval] Validation error: {e}")
        print("[Eval] This might be because the dataset split is incorrect.")
        print("[Eval] Continuing with benchmark and samples...\n")
        metrics = {}


def run_fps_benchmark(args):
    """Benchmark inference speed."""
    model_path = args.model if os.path.exists(args.model) else find_model()
    
    print(f"\n{'='*65}")
    print(f"  DyMATES — FPS Benchmark")
    print(f"{'='*65}\n")
    
    model = YOLO(model_path)
    
    # Create dummy frames at different resolutions
    resolutions = [(640, 640), (1280, 720), (1920, 1080)]
    
    for w, h in resolutions:
        dummy = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        
        # Warmup
        for _ in range(5):
            model(dummy, conf=0.3, verbose=False, imgsz=640)
        
        # Benchmark
        times = []
        for _ in range(50):
            t0 = time.time()
            model(dummy, conf=0.3, verbose=False, imgsz=640)
            times.append(time.time() - t0)
        
        avg_time = np.mean(times)
        fps = 1.0 / avg_time
        print(f"  {w}x{h}: {fps:.1f} FPS (avg {avg_time*1000:.1f}ms per frame)")
    
    print(f"\n{'='*65}\n")


def save_sample_detections(args):
    """Save sample detection images for the presentation."""
    model_path = args.model if os.path.exists(args.model) else find_model()
    
    print(f"\n[Eval] Generating {args.samples} sample detections...\n")
    
    model = YOLO(model_path)
    class_names = model.model.names
    
    # Find test images
    image_dirs = [
        "dataset/test/images",
        "dataset/valid/images",
        "dataset/val/images",
        "dataset/train/images",
    ]
    
    image_paths = []
    for d in image_dirs:
        if os.path.exists(d):
            for ext in ["*.jpg", "*.jpeg", "*.png"]:
                image_paths.extend(glob.glob(os.path.join(d, ext)))
            if image_paths:
                print(f"[Eval] Using images from: {d}")
                break
    
    if not image_paths:
        print("[Eval] No images found for sample generation.")
        return
    
    # Select random samples
    np.random.seed(42)
    selected = np.random.choice(
        image_paths, 
        size=min(args.samples, len(image_paths)), 
        replace=False
    )
    
    os.makedirs("output/samples", exist_ok=True)
    
    for i, img_path in enumerate(selected):
        frame = cv2.imread(img_path)
        if frame is None:
            continue
        
        results = model(frame, conf=0.25, imgsz=640, verbose=False)
        
        # Draw detections with nice styling
        if results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = class_names.get(cls_id, f"cls{cls_id}")
                
                # Color coding
                if "without" in cls_name.lower() or "no" in cls_name.lower():
                    color = (0, 0, 255)   # Red for violations
                    label = f"!! {cls_name} {conf:.2f}"
                elif "motor" in cls_name.lower() or "bike" in cls_name.lower():
                    color = (255, 180, 0)  # Orange for motorcycles
                    label = f"{cls_name} {conf:.2f}"
                elif "rider" in cls_name.lower() or "person" in cls_name.lower():
                    color = (255, 100, 0)  # Dark orange for riders
                    label = f"{cls_name} {conf:.2f}"
                else:
                    color = (0, 220, 0)    # Green for safe
                    label = f"{cls_name} {conf:.2f}"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 4),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add DyMATES branding
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 35), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, "DyMATES Detection", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 120), 2)
        
        out_path = f"output/samples/sample_{i+1:02d}.jpg"
        cv2.imwrite(out_path, frame)
    
    print(f"[Eval] {len(selected)} sample images saved to: output/samples/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DyMATES - Model Evaluation")
    parser.add_argument("--model", type=str, default="models/best.pt",
                        help="Path to model weights")
    parser.add_argument("--data", type=str, default="dataset/data.yaml",
                        help="Path to data.yaml")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size for validation")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for evaluation")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run FPS benchmark")
    parser.add_argument("--samples", type=int, default=10,
                        help="Number of sample detection images to save")
    args = parser.parse_args()
    
    if os.path.exists(args.data) or args.benchmark:
        if not args.benchmark:
            evaluate_model(args)
        else:
            run_fps_benchmark(args)
    
    save_sample_detections(args)
    
    print("\n[Eval] Done! Check 'output/' for all results.")
    print("[Eval] Use these in your presentation slides.\n")
