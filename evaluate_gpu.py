"""
DyMATES - Comprehensive Model Accuracy & Performance Evaluation
================================================================
Evaluates detection accuracy (mAP50, Precision, Recall, F1) for Image test dataset
and processes Video test footage for both Static and Dynamic violation detection.
"""

import os
import sys
import time
import json
import glob
import cv2
import numpy as np
import pandas as pd
from datetime import datetime

import torch
from ultralytics import YOLO

# Import local pipeline modules
from detector import DyMATESDetector
from violation_engine import check_violations, check_violations_simple, get_stats, reset_state
from dynamic_violations import check_dynamic_violations, get_dynamic_stats, reset_dynamic_state
from database import init_db, get_all_violations, log_violation

def run_evaluation():
    print(f"{'='*75}")
    print(f"  DyMATES — Comprehensive Model Accuracy & Testing Benchmark")
    print(f"{'='*75}")

    # Determine best device
    device = "cpu"
    device_name = "CPU"
    if torch.cuda.is_available():
        try:
            # Quick test CUDA execution
            t_test = torch.zeros((1, 3, 640, 640), device="cuda:0")
            device = "cuda:0"
            device_name = torch.cuda.get_device_name(0)
        except Exception:
            device = "cpu"
            device_name = "CPU (CUDA Fallback)"

    print(f"  Execution Device: {device} ({device_name})")
    print(f"  PyTorch Version:  {torch.__version__}")
    print(f"  CUDA Available:   {torch.cuda.is_available()}")
    print(f"{'='*75}\n")

    # Resolve model weights
    model_path = "models/best.pt"
    if not os.path.exists(model_path):
        model_path = "runs/detect/runs/dymates_v1/weights/last.pt"
    if not os.path.exists(model_path):
        model_path = "yolo11n.pt"

    print(f"[1/4] Loading model weights: {model_path}")
    model = YOLO(model_path)
    class_names = model.model.names
    print(f"      Model Detection Classes: {class_names}\n")

    report = {
        "timestamp": datetime.now().isoformat(),
        "device": device_name,
        "model_path": model_path,
        "class_names": class_names,
    }

    # -------------------------------------------------------------
    # PART 1: IMAGE ACCURACY & VALIDATION METRICS (mAP, P, R, F1)
    # -------------------------------------------------------------
    print(f"[2/4] --- PART 1: IMAGE ACCURACY & DETECTION PERFORMANCE ---")
    data_yaml = "dataset/data.yaml"

    try:
        val_results = model.val(
            data=data_yaml,
            split="test" if os.path.exists("dataset/test") else "val",
            imgsz=640,
            batch=4,
            device=device,
            plots=True,
            verbose=True,
        )

        map50 = float(val_results.box.map50)
        map50_95 = float(val_results.box.map)
        mp = float(val_results.box.mp)
        mr = float(val_results.box.mr)
        f1 = 2 * (mp * mr) / (mp + mr) if (mp + mr) > 0 else 0.0

        per_class_ap = {}
        if hasattr(val_results.box, 'ap50'):
            for i, cls_name in class_names.items():
                if i < len(val_results.box.ap50):
                    per_class_ap[cls_name] = {
                        "AP50": float(val_results.box.ap50[i]),
                        "AP50_95": float(val_results.box.ap[i]) if i < len(val_results.box.ap) else 0.0,
                    }

        print(f"\n  [Image Accuracy Metrics Summary]")
        print(f"    - Overall mAP@50:      {map50:.4f} ({map50*100:.2f}%)")
        print(f"    - Overall mAP@50-95:   {map50_95:.4f} ({map50_95*100:.2f}%)")
        print(f"    - Precision (P):       {mp:.4f} ({mp*100:.2f}%)")
        print(f"    - Recall (R):          {mr:.4f} ({mr*100:.2f}%)")
        print(f"    - F1-Score:            {f1:.4f} ({f1*100:.2f}%)")
        
        for cls_name, cls_metrics in per_class_ap.items():
            print(f"    - Class '{cls_name}': AP@50 = {cls_metrics['AP50']*100:.2f}%, AP@50-95 = {cls_metrics['AP50_95']*100:.2f}%")

        report["image_metrics"] = {
            "mAP50": map50,
            "mAP50_95": map50_95,
            "precision": mp,
            "recall": mr,
            "f1_score": f1,
            "per_class": per_class_ap,
        }

    except Exception as e:
        print(f"  [Warning] Image validation dataset check error: {e}")
        report["image_metrics"] = {"error": str(e)}

    # -------------------------------------------------------------
    # PART 2: IMAGE INFERENCE SPEED & CONFIDENCE DISTRIBUTION
    # -------------------------------------------------------------
    test_img_paths = glob.glob("dataset/test/images/*.jpg") + glob.glob("dataset/test/images/*.png")
    if test_img_paths:
        print(f"\n[3/4] --- PART 2: INDIVIDUAL IMAGE INFERENCE BENCHMARK ({len(test_img_paths)} images) ---")
        img_times = []
        total_dets = 0
        conf_list = []

        for img_p in test_img_paths:
            img = cv2.imread(img_p)
            if img is None:
                continue
            
            t0 = time.time()
            res = model(img, imgsz=640, device=device, verbose=False)[0]
            dt = time.time() - t0
            img_times.append(dt)

            boxes = res.boxes
            if boxes is not None and len(boxes) > 0:
                total_dets += len(boxes)
                conf_list.extend(boxes.conf.cpu().numpy().tolist())

        avg_img_time = np.mean(img_times) if img_times else 0
        img_fps = 1.0 / avg_img_time if avg_img_time > 0 else 0
        avg_conf = np.mean(conf_list) if conf_list else 0.0

        print(f"  [Image Inference Results]")
        print(f"    - Images Evaluated:     {len(test_img_paths)}")
        print(f"    - Latency per Image:    {avg_img_time*1000:.2f} ms")
        print(f"    - Inference Speed:      {img_fps:.1f} FPS")
        print(f"    - Total Objects Found:  {total_dets}")
        print(f"    - Average Confidence:   {avg_conf:.3f}")

        report["image_inference"] = {
            "images_count": len(test_img_paths),
            "avg_latency_ms": round(avg_img_time * 1000, 2),
            "fps": round(img_fps, 1),
            "total_detections": total_dets,
            "avg_confidence": round(float(avg_conf), 3),
        }

    # -------------------------------------------------------------
    # PART 3: VIDEO PIPELINE & BEHAVIORAL ACCURACY TEST
    # -------------------------------------------------------------
    video_path = "videos/test_video.mp4"
    if os.path.exists(video_path):
        print(f"\n[4/4] --- PART 3: VIDEO PIPELINE & BEHAVIORAL VIOLATION ACCURACY TEST ---")
        init_db()
        reset_state()
        reset_dynamic_state()

        detector = DyMATESDetector(model_path=model_path, confidence=0.25)
        cap = cv2.VideoCapture(video_path)
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps_rate = cap.get(cv2.CAP_PROP_FPS) or 30

        frame_count = 0
        frame_times = []
        static_viols = []
        dynamic_viols = []

        t_vid_start = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            
            t0 = time.time()
            detections = detector.detect_and_track(frame)
            sv = check_violations(detections, detector.class_names)
            dv = check_dynamic_violations(detections, detector.class_names, lane_direction=90)
            dt = time.time() - t0
            frame_times.append(dt)

            if sv: static_viols.extend(sv)
            if dv: dynamic_viols.extend(dv)

        cap.release()
        total_vid_time = time.time() - t_vid_start
        avg_frame_time = np.mean(frame_times) if frame_times else 0
        vid_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0

        static_stats = get_stats()
        dynamic_stats = get_dynamic_stats()

        print(f"  [Video Evaluation Results]")
        print(f"    - Input Video File:      {video_path}")
        print(f"    - Frames Analyzed:       {frame_count}/{total_video_frames}")
        print(f"    - Total Processing Time: {total_vid_time:.2f} seconds")
        print(f"    - Processing Latency:    {avg_frame_time*1000:.2f} ms / frame")
        print(f"    - Pipeline Speed:        {vid_fps:.1f} FPS (Source FPS: {video_fps_rate:.0f})")
        print(f"    - Unique Vehicles Tracked:{static_stats.get('total_tracked', 0)}")
        print(f"    - Helmet Violations:     {static_stats.get('helmet_violations', 0)}")
        print(f"    - Triple Riding:         {static_stats.get('triple_violations', 0)}")
        print(f"    - Wrong-Way Violations:  {dynamic_stats.get('wrong_way_violations', 0)}")
        print(f"    - Rash Driving:          {dynamic_stats.get('rash_driving_violations', 0)}")

        report["video_metrics"] = {
            "video_path": video_path,
            "total_frames": frame_count,
            "avg_frame_latency_ms": round(avg_frame_time * 1000, 2),
            "video_fps": round(vid_fps, 1),
            "objects_tracked": static_stats.get('total_tracked', 0),
            "helmet_violations": static_stats.get('helmet_violations', 0),
            "triple_violations": static_stats.get('triple_violations', 0),
            "wrong_way_violations": dynamic_stats.get('wrong_way_violations', 0),
            "rash_driving_violations": dynamic_stats.get('rash_driving_violations', 0),
        }

    # -------------------------------------------------------------
    # SAVE JSON REPORT
    # -------------------------------------------------------------
    os.makedirs("output", exist_ok=True)
    out_json = "output/test_evaluation_report.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*75}")
    print(f"  DyMATES — Evaluation Finished!")
    print(f"  Saved full accuracy metrics report to: {out_json}")
    print(f"{'='*75}\n")

if __name__ == "__main__":
    run_evaluation()
