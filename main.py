"""
DyMATES — Dynamic Multi-Agent Traffic Enforcement System
=========================================================
Master Pipeline: All 5 Layers + Dynamic Violations

Architecture (Slide 11):
  Layer 1: Preprocessing   → 640×640 resize + Mild CLAHE
  Layer 2: YOLOv11-Nano    → Single-pass unified detection
  Layer 3: ByteTrack       → Persistent tracking (occlusion recovery)
  Layer 4: Violation Logic  → Static (helmet, triple riding) + Dynamic (wrong-way, rash)
  Layer 5: Evidence + DB   → Snapshots + SQLite logging + Face blur

Violations Detected (Slide 2):
  1. No Helmet      — Hierarchical linking + temporal voting (Slide 14)
  2. Triple Riding   — Rider count per motorcycle + temporal voting (Slide 14)
  3. Wrong-Way       — Vector angle analysis on centroid trajectory (Slide 15)
  4. Rash Driving    — Lateral variance monitoring / zig-zag detection (Slide 15)

Usage:
    python main.py                                    # default settings
    python main.py --video videos/test_video.mp4      # specify video
    python main.py --model models/best.pt             # use trained model
    python main.py --images dataset/test/images       # run on images
    python main.py --save-output                      # save annotated output
    python main.py --lane-direction 90                # set lane direction (degrees)
"""

import argparse
import os
import sys
import time
import glob
import cv2
import numpy as np

from detector import DyMATESDetector
from violation_engine import (
    check_violations, check_violations_simple,
    get_stats, reset_state
)
from dynamic_violations import (
    check_dynamic_violations, get_dynamic_stats, reset_dynamic_state
)
from database import init_db, log_violation, print_violations_summary
from ocr_engine import extract_license_plate_text


# ══════════════════════════════════════════════════════════════════
# LAYER 5: EVIDENCE CAPTURE (Privacy + DB + Snapshots)
# ══════════════════════════════════════════════════════════════════
def blur_faces(frame, detections, class_names):
    """
    Layer 5 (Privacy): Apply Gaussian blur to detected head regions.
    Slide 16: GDPR compliance via automated face blurring.
    """
    if detections.class_id is None or len(detections) == 0:
        return frame

    for i in range(len(detections)):
        cls_id = int(detections.class_id[i])
        cls_name = class_names.get(cls_id, "").lower()

        # Blur top 30% of rider/person detection boxes (head region)
        x1, y1, x2, y2 = map(int, detections.xyxy[i])
        head_y2 = y1 + int((y2 - y1) * 0.3)
        
        # Clamp coordinates
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        head_y2 = min(frame.shape[0], head_y2)
        
        head_region = frame[y1:head_y2, x1:x2]
        if head_region.size > 0:
            blurred = cv2.GaussianBlur(head_region, (21, 21), 10)
            frame[y1:head_y2, x1:x2] = blurred

    return frame


def save_evidence(frame, violation, frame_count):
    """
    Layer 5 (Evidence): Save violation snapshot and log to database.
    """
    snap_name = f"violation_track{violation['track_id']}_{violation['type']}_frame{frame_count}.jpg"
    snap_path = os.path.join("snapshots", snap_name)
    cv2.imwrite(snap_path, frame)
    
    # Layer 4.5: License Plate Extraction
    plate_text = "N/A"
    if "box" in violation and violation["box"] is not None and len(violation["box"]) == 4:
        plate_text = extract_license_plate_text(frame, violation["box"])
        
    log_violation(
        track_id=violation["track_id"],
        violation_type=violation["type"],
        frame_number=frame_count,
        snapshot_path=snap_path,
        confidence=violation.get("confidence", 0.0),
        plate_text=plate_text,
    )
    return snap_path


# ══════════════════════════════════════════════════════════════════
# VIDEO PIPELINE
# ══════════════════════════════════════════════════════════════════
def run_video_pipeline(args):
    """
    Full 5-layer video pipeline with ALL violation types:
    Frame → L1(CLAHE) → L2(YOLO) → L3(ByteTrack) → L4(All Violations) → L5(Evidence)
    """
    print(f"\n{'='*70}")
    print(f"  DyMATES — Dynamic Multi-Agent Traffic Enforcement System")
    print(f"  Real-Time Multi-Violation Detection Pipeline (Video Mode)")
    print(f"{'='*70}")
    print(f"  Pipeline: L1(CLAHE) → L2(YOLO) → L3(ByteTrack) → L4(Logic) → L5(DB)")
    print(f"  Violations: No Helmet | Triple Riding | Wrong-Way | Rash Driving")
    print(f"{'='*70}\n")

    # Initialize Layer 5: Database
    init_db()
    reset_state()
    reset_dynamic_state()
    os.makedirs("snapshots", exist_ok=True)

    # Check video
    if not os.path.exists(args.video):
        print(f"[ERROR] Video not found: {args.video}")
        print(f"  Download a test video and place at: {args.video}")
        return

    # Resolve model path
    model_path = _resolve_model_path(args.model)

    # Initialize Layers 1-3 (Detector)
    detector = DyMATESDetector(
        model_path=model_path, confidence=args.confidence
    )

    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {args.video}")
        return

    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[Pipeline] Video: {args.video}")
    print(f"[Pipeline] Resolution: {width}x{height} @ {fps_video:.0f}fps")
    print(f"[Pipeline] Total frames: {total_frames}")
    print(f"[Pipeline] Lane direction: {args.lane_direction}°")
    print(f"[Pipeline] Privacy blur: {'ON' if args.blur_faces else 'OFF'}")
    print(f"[Pipeline] Starting...\n")

    # Output writer
    out_writer = None
    if args.save_output:
        os.makedirs("output", exist_ok=True)
        out_path = "output/dymates_output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(out_path, fourcc, fps_video, (width, height))
        print(f"[Pipeline] Saving output to: {out_path}\n")

    # ── Main Loop ─────────────────────────────────────────────────
    frame_count = 0
    total_violations_count = 0
    prev_time = time.time()
    fps_display = 0.0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # ── Layers 1→2→3: Detect + Track ────────────────────
            detections = detector.detect_and_track(frame)

            # ── Layer 4a: Static Violations (helmet + triple) ───
            static_violations = check_violations(detections, detector.class_names)

            # ── Layer 4b: Dynamic Violations (wrong-way + rash) ─
            dynamic_violations = check_dynamic_violations(
                detections, detector.class_names, args.lane_direction
            )

            # Combine all violations
            all_violations = static_violations + dynamic_violations

            # ── Layer 5: Log Violations + Evidence ───────────────
            for v in all_violations:
                total_violations_count += 1
                save_evidence(frame, v, frame_count)

            # ── Annotate ─────────────────────────────────────────
            annotated = detector.annotate_frame(frame, detections, all_violations)

            if args.blur_faces:
                annotated = blur_faces(annotated, detections, detector.class_names)

            # FPS calculation
            curr_time = time.time()
            fps_display = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time

            # Dashboard overlay
            static_stats = get_stats()
            dynamic_stats = get_dynamic_stats()
            combined_stats = {**static_stats, **dynamic_stats}
            annotated = detector.draw_dashboard(
                annotated, frame_count, fps_display, total_violations_count, combined_stats
            )

            # Display GUI window if --show is passed
            if getattr(args, "show", False):
                try:
                    cv2.imshow("DyMATES - Traffic Enforcement System", annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        print("\n[Pipeline] Stopped by user.")
                        break
                    elif key == ord(" "):
                        print("[Pipeline] Paused. Press any key to continue...")
                        cv2.waitKey(0)
                except Exception as e:
                    print(f"[Warning] GUI Display unavailable: {e}")

            if out_writer:
                out_writer.write(annotated)

    except KeyboardInterrupt:
        print("\n[Pipeline] Interrupted.")
    finally:
        cap.release()
        if out_writer:
            out_writer.release()
        cv2.destroyAllWindows()

    _print_final_report(frame_count, fps_display, total_violations_count, args)


# ══════════════════════════════════════════════════════════════════
# IMAGE PIPELINE
# ══════════════════════════════════════════════════════════════════
def run_image_pipeline(args):
    """
    Run the pipeline on a directory of images.
    Layers 1→2→(skip L3)→L4(simple)→L5
    """
    print(f"\n{'='*65}")
    print(f"  DyMATES — Helmet Violation Detection (Image Mode)")
    print(f"{'='*65}\n")

    init_db()
    os.makedirs("snapshots", exist_ok=True)
    os.makedirs("output/image_results", exist_ok=True)

    model_path = _resolve_model_path(args.model)
    detector = DyMATESDetector(model_path=model_path, confidence=args.confidence)

    # Find images
    image_dir = args.images
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(image_dir, ext)))

    if not image_paths:
        print(f"[ERROR] No images found in: {image_dir}")
        return

    print(f"[Pipeline] Found {len(image_paths)} images")
    print(f"[Pipeline] Processing...\n")

    stats = {"processed": 0, "violations": 0, "safe": 0}
    total_time = 0

    for i, img_path in enumerate(image_paths):
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        t0 = time.time()

        # Layer 1: Preprocess
        preprocessed = detector.preprocess(frame)

        # Layer 2: Detect
        detections = detector.detect(preprocessed)

        dt = time.time() - t0
        total_time += dt
        stats["processed"] += 1

        # Layer 4 (simplified for images): Direct violation check
        violations = check_violations_simple(detections, detector.class_names)

        # Annotate
        annotated = frame.copy()
        if len(detections) > 0:
            labels = []
            for j in range(len(detections)):
                cls_id = int(detections.class_id[j])
                cls_name = detector.class_names.get(cls_id, f"cls{cls_id}")
                conf = float(detections.confidence[j])
                labels.append(f"{cls_name} {conf:.2f}")
            annotated = detector.box_annotator.annotate(scene=annotated, detections=detections)
            annotated = detector.label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

        # Draw violation alerts
        for v in violations:
            stats["violations"] += 1
            plate_text = "N/A"
            if v.get("box") is not None:
                box = v["box"].astype(int)
                plate_text = extract_license_plate_text(frame, box)
                cv2.rectangle(annotated, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 3)
                cv2.putText(annotated, f"!! {v['type']} ({v['confidence']:.2f}) | LP: {plate_text} !!",
                            (box[0], max(20, box[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Save evidence
            snap_name = f"violation_{stats['violations']}_{os.path.basename(img_path)}"
            snap_path = os.path.join("snapshots", snap_name)
            cv2.imwrite(snap_path, annotated)
            log_violation(
                track_id=v.get("track_id", -1),
                violation_type=v["type"],
                frame_number=i,
                snapshot_path=snap_path,
                confidence=v["confidence"],
                plate_text=plate_text,
            )
            print(f"  [!!] VIOLATION in {os.path.basename(img_path)} "
                  f"(type={v['type']}, conf={v['confidence']:.2f}, LP={plate_text})")

        # Save all annotated images
        out_path = os.path.join("output/image_results", f"det_{os.path.basename(img_path)}")
        cv2.imwrite(out_path, annotated)

        if (i + 1) % 20 == 0 or (i + 1) == len(image_paths):
            print(f"  [{i+1}/{len(image_paths)}] processed ({1.0/dt:.1f} FPS)")

    avg_fps = stats["processed"] / total_time if total_time > 0 else 0
    print(f"\n{'='*65}")
    print(f"  DyMATES — Image Processing Complete")
    print(f"{'='*65}")
    print(f"  Images processed:  {stats['processed']}")
    print(f"  Avg FPS:           {avg_fps:.1f}")
    print(f"  Violations found:  {stats['violations']}")
    print(f"  Results saved to:  output/image_results/")
    print(f"  Snapshots:         snapshots/")
    print(f"{'='*65}\n")

    print_violations_summary()


# ══════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════
def _resolve_model_path(model_arg):
    """Find the best available model weights."""
    if os.path.exists(model_arg):
        return model_arg
    if os.path.exists("models/best.pt"):
        print(f"[Info] Using trained model: models/best.pt")
        return "models/best.pt"
    if os.path.exists("runs/detect/runs/dymates_v1/weights/best.pt"):
        print(f"[Info] Using trained model from runs/")
        return "runs/detect/runs/dymates_v1/weights/best.pt"
    if os.path.exists("runs/dymates_v1/weights/best.pt"):
        return "runs/dymates_v1/weights/best.pt"

    print(f"[Info] No custom model found. Using pretrained yolo11n.pt")
    print(f"  → Run 'python train.py' for helmet-specific model.\n")
    return "yolo11n.pt"


def _print_final_report(frame_count, fps, violations, args):
    """Print comprehensive end-of-run summary."""
    static_stats = get_stats()
    dynamic_stats = get_dynamic_stats()

    print(f"\n{'='*70}")
    print(f"  DyMATES — Run Complete")
    print(f"{'='*70}")
    print(f"  Frames processed:        {frame_count}")
    print(f"  Avg FPS:                 {fps:.1f}")
    print(f"  Total violations:        {violations}")
    print(f"  ├── Helmet violations:   {static_stats.get('helmet_violations', 0)}")
    print(f"  ├── Triple riding:       {static_stats.get('triple_violations', 0)}")
    print(f"  ├── Wrong-way:           {dynamic_stats.get('wrong_way_violations', 0)}")
    print(f"  └── Rash driving:        {dynamic_stats.get('rash_driving_violations', 0)}")
    print(f"  Objects tracked:         {static_stats.get('total_tracked', 0)}")
    print(f"  Trajectories analyzed:   {dynamic_stats.get('trajectories_tracked', 0)}")
    print(f"{'='*70}\n")

    print_violations_summary()

    if args.save_output:
        print(f"[Output] Annotated video: output/dymates_output.mp4")
    print(f"[Output] Violation snapshots: snapshots/")
    print(f"[Output] Database: violations.db\n")


# ══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DyMATES - Dynamic Multi-Agent Traffic Enforcement System"
    )
    parser.add_argument("--video", type=str, default="videos/test_video.mp4",
                        help="Path to input video")
    parser.add_argument("--images", type=str, default=None,
                        help="Path to image directory (overrides --video)")
    parser.add_argument("--model", type=str, default="models/best.pt",
                        help="Path to YOLO model weights")
    parser.add_argument("--confidence", type=float, default=0.3,
                        help="Detection confidence threshold")
    parser.add_argument("--blur-faces", action="store_true", default=True,
                        help="Enable face blurring for privacy")
    parser.add_argument("--save-output", action="store_true", default=False,
                        help="Save annotated output video")
    parser.add_argument("--show", action="store_true", default=False,
                        help="Display GUI window during video execution")
    parser.add_argument("--lane-direction", type=float, default=90.0,
                        help="Expected lane direction in degrees (90=downward)")
    args = parser.parse_args()

    if args.images:
        run_image_pipeline(args)
    else:
        run_video_pipeline(args)
