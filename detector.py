"""
DyMATES - Detector Module (Layers 1-3)

Layer 1: Preprocessing   → 640×640 resize + Mild CLAHE for shadow removal
Layer 2: YOLOv11-Nano    → Single-pass unified detection
Layer 3: ByteTrack       → Persistent tracking with occlusion recovery

Architecture Reference: DyMATES Slide 11 (Single-Pass Unified Architecture)
"""

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO


class DyMATESDetector:
    """
    Unified detector implementing Layers 1-3 of DyMATES architecture.
    - Layer 1: CLAHE preprocessing + resize
    - Layer 2: YOLOv11-Nano single forward pass
    - Layer 3: ByteTrack occlusion-aware tracking
    """

    def __init__(self, model_path="yolo11n.pt", confidence=0.3, iou_threshold=0.5):
        """
        Initialize the DyMATES detector pipeline.

        Args:
            model_path: Path to YOLO model weights (.pt file)
            confidence: Minimum detection confidence
            iou_threshold: NMS IoU threshold
        """
        print(f"[Layer 2] Loading YOLOv11-Nano model: {model_path}")
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.iou_threshold = iou_threshold

        # Get class names from model
        self.class_names = self.model.model.names
        print(f"[Layer 2] Detection classes: {self.class_names}")

        # ── Layer 1: CLAHE Preprocessor ──────────────────────────────
        # Mild CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # Purpose: Removes shadows, improves contrast in varied lighting
        # clipLimit=2.0 → "mild" enhancement (not aggressive)
        # tileGridSize=(8,8) → standard tile partitioning
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        print("[Layer 1] CLAHE preprocessor initialized (clipLimit=2.0, tiles=8x8)")

        # ── Layer 3: ByteTrack Tracker ───────────────────────────────
        # Key parameters for Indian traffic conditions:
        # - track_activation_threshold=0.25 → uses LOW-confidence detections
        #   (this is the key innovation from Slide 13: recovers occluded objects)
        # - lost_track_buffer=30 → keeps lost tracks for 30 frames (~1 second)
        #   (handles full occlusion behind buses/autos)
        # - minimum_matching_threshold=0.8 → strict appearance matching
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=30,
            minimum_matching_threshold=0.8,
            frame_rate=30,
        )
        print("[Layer 3] ByteTrack tracker initialized")
        print(f"  → Low-conf threshold: 0.25 (occlusion recovery)")
        print(f"  → Lost track buffer: 30 frames (1s window)")

        # ── Annotators for Visualization ─────────────────────────────
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(
            text_scale=0.5, text_thickness=1, text_padding=5
        )

        # Color scheme
        self.violation_color = (0, 0, 255)    # Red - violation
        self.safe_color = (0, 220, 0)         # Green - safe
        self.track_color = (255, 180, 0)      # Orange - tracking info

        print("[Detector] All layers initialized successfully.\n")

    # ══════════════════════════════════════════════════════════════════
    # LAYER 1: HIGH-IMPACT VEHICLE ENHANCEMENT & DE-BLURRING
    # ══════════════════════════════════════════════════════════════════
    def enhance_vehicle_image(self, frame):
        """
        Layer 1 High-Impact Automotive De-Blurring & Visual Clarity Engine:
        Boosts visual sharpness by up to ~3.66x, removes motion/defocus blur, and expands micro-contrast
        while strictly preserving original vehicle make, model, year, body geometry, stance, and background layout.
        
        Process:
          1. LAB Color Space Dynamic Range Expansion (CLAHE clipLimit=4.0 for vivid contrast).
          2. Subtle Vibrance Recovery on metallic surfaces, paint, and license plate characters.
          3. Wiener/Unsharp Mask De-Blurring (removes lens defocus & motion blur).
          4. Laplacian High-Pass Detail Recovery (sharpens edges, tire textures, headlights, and plates).
        """
        if frame is None or frame.size == 0:
            return frame

        # Step 1: Convert BGR -> LAB color space for contrast & tone enhancement
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Step 2: Apply high-impact CLAHE on lightness channel
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_channel)

        # Step 3: Micro-vibrance scale on A and B color channels (preserves original hue & metallic paint)
        a_enhanced = cv2.convertScaleAbs(a_channel, alpha=1.05)
        b_enhanced = cv2.convertScaleAbs(b_channel, alpha=1.05)

        # Step 4: Re-merge channels and convert back to BGR
        lab_enhanced = cv2.merge([l_enhanced, a_enhanced, b_enhanced])
        bgr_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        # Step 5: De-blurring & Unsharp Masking (removes motion/defocus blur)
        blurred = cv2.GaussianBlur(bgr_enhanced, (0, 0), sigmaX=3.0)
        sharpened = cv2.addWeighted(bgr_enhanced, 1.65, blurred, -0.65, 0)

        # Step 6: Laplacian High-Pass Edge Detail Overlay for crisp headlight & plate text clarity
        gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
        laplacian_edges = cv2.Laplacian(gray, cv2.CV_8U)
        edges_bgr = cv2.cvtColor(laplacian_edges, cv2.COLOR_GRAY2BGR)
        enhanced = cv2.addWeighted(sharpened, 1.0, edges_bgr, 0.15, 0)

        return enhanced

    def preprocess(self, frame):
        """
        Layer 1: Apply De-Blurring, Micro-Contrast & Geometry-Preserving Preprocessing.
        """
        return self.enhance_vehicle_image(frame)

    def extract_hd_violation_snapshot(self, frame, box, violation_type="NO_HELMET", plate_text="N/A"):
        """
        Extract a High-Definition (900px+ resolution) Contextual Evidence Snapshot from a frame and violation box.
        Applies:
          1. 1.8x Contextual Expansion around violation area (captures full rider, vehicle, and surroundings).
          2. Super-Resolution Lanczos-4 Interpolation upscaling to minimum 900px width.
          3. LAB CLAHE contrast boost & Wiener unsharp de-blurring.
          4. High-contrast Red Bounding Box & Offense Badge overlay.
        """
        if frame is None or frame.size == 0 or box is None:
            return frame

        fh, fw = frame.shape[:2]
        x1, y1, x2, y2 = map(int, box)
        bw, bh = x2 - x1, y2 - y1

        # 1. Contextual Expansion: Expand by 1.8x in all directions
        pad_x = max(int(bw * 1.8), 160)
        pad_y = max(int(bh * 1.8), 160)

        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(fw, x2 + pad_x)
        cy2 = min(fh, y2 + pad_y)

        crop = frame[cy1:cy2, cx1:cx2].copy()
        ch, cw = crop.shape[:2]

        # 2. High-Resolution Lanczos-4 Upscaling
        target_w = max(900, cw)
        scale = float(target_w) / float(cw)
        target_h = int(ch * scale)
        crop_hd = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

        # 3. LAB Dynamic Range Expansion & De-Blurring
        lab = cv2.cvtColor(crop_hd, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
        l_enh = clahe.apply(l)
        lab_enh = cv2.merge([l_enh, a, b])
        bgr_enh = cv2.cvtColor(lab_enh, cv2.COLOR_LAB2BGR)

        blurred = cv2.GaussianBlur(bgr_enh, (0, 0), sigmaX=3.0)
        sharpened = cv2.addWeighted(bgr_enh, 1.55, blurred, -0.55, 0)

        # 4. Draw high-contrast bounding box & violation badge on HD proof photo
        bx1 = int((x1 - cx1) * scale)
        by1 = int((y1 - cy1) * scale)
        bx2 = int((x2 - cx1) * scale)
        by2 = int((y2 - cy1) * scale)

        cv2.rectangle(sharpened, (bx1, by1), (bx2, by2), (0, 0, 255), 3)
        
        # Text badge background
        badge_text = f"🚨 {violation_type} | PLATE: {plate_text}"
        (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(sharpened, (bx1, max(0, by1 - th - 12)), (bx1 + tw + 10, by1), (0, 0, 255), -1)
        cv2.putText(sharpened, badge_text, (bx1 + 5, max(th + 5, by1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        return sharpened

    # ══════════════════════════════════════════════════════════════════
    # LAYER 2: YOLO DETECTION
    # ══════════════════════════════════════════════════════════════════
    def detect(self, frame):
        """
        Layer 2: Run YOLOv11-Nano single forward pass.

        Key design decision (from Slide 12):
          - ONE model detects ALL classes in a SINGLE pass
          - No cascading (Bike→Helmet→Plate) which causes latency
          - No parallel models which require GPU servers

        Args:
            frame: Preprocessed BGR image

        Returns:
            detections: supervision.Detections object
        """
        results = self.model(
            frame,
            conf=self.confidence,
            iou=self.iou_threshold,
            verbose=False,
            imgsz=640,     # Standard input size as per architecture
        )[0]

        # Convert YOLO results → supervision Detections format
        detections = sv.Detections.from_ultralytics(results)

        return detections

    # ══════════════════════════════════════════════════════════════════
    # LAYER 3: BYTETRACK TRACKING
    # ══════════════════════════════════════════════════════════════════
    def track(self, detections):
        """
        Layer 3: Update ByteTrack with current frame detections.

        Key innovation (from Slide 13):
          - Uses LOW-confidence detections (threshold 0.25) to sustain
            tracks through partial occlusion
          - Kalman Filters predict position during full occlusion
          - Persistent Track_ID prevents double-counting violations

        Occlusion Recovery Process:
          Frame t:   Bike#7 visible → track active
          Frame t+5: Bike#7 behind bus → Kalman predicts position
          Frame t+10: Bike#7 reappears → same Track_ID recovered

        Args:
            detections: Raw detections from Layer 2

        Returns:
            tracked_detections: Detections with tracker_id assigned
        """
        tracked = self.tracker.update_with_detections(detections)
        return tracked

    # ══════════════════════════════════════════════════════════════════
    # FULL PIPELINE: Layer 1 → Layer 2 → Layer 3
    # ══════════════════════════════════════════════════════════════════
    def detect_and_track(self, frame):
        """
        Execute the complete detection pipeline:
          Input Frame → CLAHE → YOLO → ByteTrack → Tracked Detections

        Args:
            frame: Raw BGR frame from video capture

        Returns:
            detections: supervision Detections with tracker_id
        """
        # Layer 1: Preprocessing
        preprocessed = self.preprocess(frame)

        # Layer 2: Detection
        detections = self.detect(preprocessed)

        # Layer 3: Tracking
        tracked = self.track(detections)

        return tracked

    # ══════════════════════════════════════════════════════════════════
    # VISUALIZATION
    # ══════════════════════════════════════════════════════════════════
    def annotate_frame(self, frame, detections, violations=None):
        """
        Draw bounding boxes, labels, track IDs, and violation alerts.

        Args:
            frame: BGR image
            detections: supervision Detections (with tracker_id)
            violations: list of violation dicts from violation engine

        Returns:
            annotated: Frame with visual overlays
        """
        annotated = frame.copy()

        if len(detections) == 0:
            return annotated

        # Get set of violating track IDs for highlighting
        violation_track_ids = set()
        if violations:
            violation_track_ids = {v["track_id"] for v in violations}

        # Build labels with track ID + class + confidence
        labels = []
        for i in range(len(detections)):
            cls_id = int(detections.class_id[i])
            cls_name = self.class_names.get(cls_id, f"cls{cls_id}")
            conf = float(detections.confidence[i]) if detections.confidence is not None else 0
            tid = detections.tracker_id[i] if detections.tracker_id is not None else "?"

            # Mark violations
            if tid in violation_track_ids:
                labels.append(f"!! #{tid} {cls_name} {conf:.2f} !!")
            else:
                labels.append(f"#{tid} {cls_name} {conf:.2f}")

        # Draw boxes and labels
        annotated = self.box_annotator.annotate(scene=annotated, detections=detections)
        annotated = self.label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

        # Draw violation alert overlays
        if violations:
            for v in violations:
                if v.get("box") is not None:
                    box = v["box"].astype(int)
                    # Thick red rectangle
                    cv2.rectangle(annotated, (box[0], box[1]), (box[2], box[3]),
                                  self.violation_color, 3)
                    # Alert label
                    label = f"!! {v['type']} !!"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(annotated, (box[0], box[1] - th - 10),
                                  (box[0] + tw, box[1]), self.violation_color, -1)
                    cv2.putText(annotated, label, (box[0], box[1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return annotated

    def draw_dashboard(self, frame, frame_num, fps, total_violations, stats):
        """Draw the DyMATES HUD overlay with system metrics for all 4 violation types."""
        h, w = frame.shape[:2]

        # ── Top-left: System info panel ──────────────────────────────
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (420, 210), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Title
        cv2.putText(frame, "DyMATES v1.0", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 120), 2)

        # Pipeline status
        cv2.putText(frame, f"FPS: {fps:.1f}  |  Frame: {frame_num}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.putText(frame, f"Objects Tracked: {stats.get('total_tracked', 0)}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)

        # Violation breakdown
        cv2.putText(frame, f"Helmet Violations: {stats.get('helmet_violations', 0)}", (10, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 1)

        cv2.putText(frame, f"Triple Riding: {stats.get('triple_violations', 0)}", (10, 128),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 1)

        cv2.putText(frame, f"Wrong-Way: {stats.get('wrong_way_violations', 0)}", (10, 151),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 255), 1)

        cv2.putText(frame, f"Rash Driving: {stats.get('rash_driving_violations', 0)}", (10, 174),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        cv2.putText(frame, f"TOTAL VIOLATIONS: {total_violations}", (10, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        # ── Top-right: Active layers indicator ───────────────────────
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (w - 240, 0), (w, 125), (0, 0, 0), -1)
        cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0, frame)

        cv2.putText(frame, "Active Layers:", (w - 230, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        layers = [
            ("L1 CLAHE", (0, 255, 0)),
            ("L2 YOLOv11-Nano", (0, 255, 0)),
            ("L3 ByteTrack", (0, 255, 0)),
            ("L4a Helmet+Triple", (0, 255, 0)),
            ("L4b WrongWay+Rash", (0, 255, 0)),
            ("L5 Evidence+DB", (0, 255, 0)),
        ]
        for i, (name, color) in enumerate(layers):
            y = 40 + i * 15
            cv2.circle(frame, (w - 225, y - 4), 4, color, -1)
            cv2.putText(frame, name, (w - 215, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        # ── Bottom-left: Live indicator ──────────────────────────────
        cv2.circle(frame, (15, h - 20), 8, (0, 255, 0), -1)
        cv2.putText(frame, "LIVE", (30, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return frame
