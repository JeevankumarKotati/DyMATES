import cv2
import re
import numpy as np

_ocr_model = None

def get_ocr_model():
    global _ocr_model
    if _ocr_model is None:
        print("[Layer 4.5] Initializing PaddleOCR Engine...")
        try:
            import logging
            logging.getLogger('ppocr').setLevel(logging.ERROR)
            from paddleocr import PaddleOCR
            _ocr_model = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, show_log=False)
        except ImportError:
            print("[Layer 4.5 Error] PaddleOCR not installed. Run: pip install paddlepaddle paddleocr")
            return None
    return _ocr_model


def preprocess_plate_crop(crop):
    """
    Apply advanced computer vision preprocessing to small/shadowed license plate crops:
    1. Rescale image by 2x using bicubic interpolation.
    2. Convert to grayscale.
    3. Apply CLAHE contrast limited adaptive histogram equalization.
    4. Apply Gaussian blur to reduce noise.
    """
    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return crop
    
    # 1. Upscale for better OCR character recognition
    scale_factor = 2.0 if h < 60 else 1.5
    resized = cv2.resize(crop, (int(w * scale_factor), int(h * scale_factor)), interpolation=cv2.INTER_CUBIC)
    
    # 2. Grayscale conversion
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    # 3. Apply CLAHE for shadow removal
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # 4. Convert back to 3-channel for PaddleOCR engine
    bgr_enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    return bgr_enhanced


def extract_license_plate_text(frame, bbox):
    """
    Given a frame and a bounding box [x1, y1, x2, y2] of a vehicle/plate,
    extract text using preprocessed PaddleOCR to simulate Automatic Number Plate Recognition (ANPR).
    """
    ocr = get_ocr_model()
    if ocr is None:
        return "OCR_OFFLINE"

    x1, y1, x2, y2 = map(int, bbox)
    
    # Add a tight padding
    padding = 6
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
    x2, y2 = min(w, x2 + padding), min(h, y2 + padding)

    raw_crop = frame[y1:y2, x1:x2]
    if raw_crop.size == 0 or raw_crop.shape[0] < 8 or raw_crop.shape[1] < 8:
        return "NOT FOUND"

    # Try preprocessed crop first
    prep_crop = preprocess_plate_crop(raw_crop)

    crops_to_try = [prep_crop, raw_crop]
    extracted_text = []

    for crop in crops_to_try:
        try:
            result = ocr.ocr(crop, cls=True)
        except Exception:
            continue

        if not result or result[0] is None:
            continue

        for line in result[0]:
            text = line[1][0]
            conf = line[1][1]
            if conf > 0.35:
                clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
                if len(clean_text) >= 3:
                    extracted_text.append(clean_text)
        
        if extracted_text:
            break

    if not extracted_text:
        return "NOT FOUND"

    # ALPR Heuristics for Indian License Plates:
    # Standard format pattern: e.g. MH12DE1432, KA05MP569
    standard_pattern = re.compile(r'^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}$')
    pattern_matches = [t for t in extracted_text if standard_pattern.match(t)]
    if pattern_matches:
        return max(pattern_matches, key=len)

    # Candidates containing both letters and digits
    both_candidates = [t for t in extracted_text if any(c.isalpha() for c in t) and any(c.isdigit() for c in t)]
    if both_candidates:
        return max(both_candidates, key=len)

    # Candidates containing digits
    digit_candidates = [t for t in extracted_text if any(c.isdigit() for c in t)]
    if digit_candidates:
        return max(digit_candidates, key=len)

    return max(extracted_text, key=len)


class MultiFrameOCRVoter:
    """
    Maintains a rolling buffer of license plate OCR readings per vehicle track_id.
    Returns the majority vote consensus text across multiple observed frames.
    """
    def __init__(self, max_history=10):
        self.max_history = max_history
        self.buffers = {}

    def add_reading(self, track_id, raw_plate_text):
        if track_id is None or track_id < 0 or raw_plate_text in ["NOT FOUND", "OCR_OFFLINE", "N/A"]:
            return raw_plate_text
        if track_id not in self.buffers:
            self.buffers[track_id] = []
        self.buffers[track_id].append(raw_plate_text)
        if len(self.buffers[track_id]) > self.max_history:
            self.buffers[track_id].pop(0)
        return self.get_consensus(track_id)

    def get_consensus(self, track_id):
        if track_id not in self.buffers or not self.buffers[track_id]:
            return "NOT FOUND"
        valid_readings = [p for p in self.buffers[track_id] if p not in ["NOT FOUND", "OCR_OFFLINE", "N/A"]]
        if not valid_readings:
            return self.buffers[track_id][-1]
        from collections import Counter
        counts = Counter(valid_readings)
        most_common, _ = counts.most_common(1)[0]
        return most_common

    def reset(self):
        self.buffers.clear()

