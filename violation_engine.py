"""
DyMATES - Violation Engine (Layer 4) — FULL VERSION
====================================================

Implements ALL violation types from the DyMATES architecture:

Static Violations (Slide 14):
  1. No Helmet — Hierarchical Linking (no-helmet ⊂ rider bbox)
     + Temporal Voting (30-frame window, ≥8 votes → confirmed)
  2. Triple Riding — Count riders per motorcycle bbox
     + Temporal Voting (20/30 frames with >2 riders → confirmed)

Dynamic Violations (Slide 15):
  → delegated to dynamic_violations.py (wrong-way + rash driving)

Supports both:
  - 2-class model: (With Helmet, Without Helmet) — basic helmet checking
  - Multi-class model: (Motorcycle, Rider, Helmet, No-Helmet, Plate) — full pipeline
"""

from collections import defaultdict
import numpy as np


# ══════════════════════════════════════════════════════════════════
# CONFIGURATION — Temporal Voting Parameters (Slide 14)
# ══════════════════════════════════════════════════════════════════

# Helmet violation
HELMET_VOTE_THRESHOLD = 2        # votes needed in window to confirm (lowered for fast reaction)
TEMPORAL_WINDOW = 20             # 20-frame sliding window
MIN_FRAMES_BEFORE_DECISION = 2   # minimum frames before any decision

# Triple riding violation
TRIPLE_RIDING_VOTE_THRESHOLD = 2   # frames with >2 riders to confirm
TRIPLE_RIDING_WINDOW = 20          # sliding window for rider count
TRIPLE_RIDING_MIN_FRAMES = 2       # minimum frames before checking

# Spatial overlap threshold for hierarchical linking
IOU_THRESHOLD = 0.15              # minimum overlap for rider/helmet association
CONTAINMENT_THRESHOLD = 0.3       # fraction of helmet bbox inside rider bbox


# ══════════════════════════════════════════════════════════════════
# CLASS NAME DETECTION HELPERS
# ══════════════════════════════════════════════════════════════════

def _normalize_class(cls_name):
    """Normalize class name to handle various dataset naming conventions."""
    return cls_name.lower().replace("_", " ").replace("-", " ").strip()


def _is_motorcycle(cls_name):
    """Check if class represents a motorcycle/bike/scooter."""
    name = _normalize_class(cls_name)
    return any(kw in name for kw in ["motorcycle", "motorbike", "bike", "moto", "two wheeler", "bicycle", "scooter"])


def _is_rider(cls_name):
    """Check if class represents a rider/person on motorcycle."""
    name = _normalize_class(cls_name)
    return any(kw in name for kw in ["rider", "person", "motorcyclist"])


def _is_no_helmet(cls_name):
    """Check if class represents no-helmet/without-helmet."""
    name = _normalize_class(cls_name)
    return any(kw in name for kw in ["without helmet", "no helmet", "nohelmet", "without", "no_helmet"])


def _is_helmet(cls_name):
    """Check if class represents helmet/with-helmet (safe)."""
    name = _normalize_class(cls_name)
    if _is_no_helmet(cls_name):
        return False
    return any(kw in name for kw in ["with helmet", "helmet"])


def _is_plate(cls_name):
    """Check if class represents license/number plate."""
    name = _normalize_class(cls_name)
    return any(kw in name for kw in ["plate", "license", "number"])


def detect_model_type(class_names):
    """
    Detect whether we have a 2-class, multi-class, or COCO model.
    Returns: '2class', 'multiclass', or 'coco'
    """
    has_motorcycle = any(_is_motorcycle(n) for n in class_names.values())
    has_rider = any(_is_rider(n) for n in class_names.values())
    has_helmet_classes = any(_is_helmet(n) or _is_no_helmet(n) for n in class_names.values())
    
    if has_motorcycle and has_helmet_classes:
        return "multiclass"   # Custom trained model with motorcycle + helmet classes
    elif has_helmet_classes and not has_motorcycle:
        return "2class"       # Pure helmet-only model (With/Without Helmet)
    elif has_motorcycle or has_rider:
        return "coco"         # COCO model — has motorcycle/person but no helmet classes
    return "2class"


# ══════════════════════════════════════════════════════════════════
# SPATIAL HELPERS — Hierarchical Bounding Box Linking & Vehicle Guard
# ══════════════════════════════════════════════════════════════════

def _bbox_center(box):
    """Get center point of bounding box."""
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _bbox_area(box):
    """Get area of bounding box."""
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _bbox_iou(box1, box2):
    """Compute IoU between two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = _bbox_area(box1)
    area2 = _bbox_area(box2)
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0


def _bbox_containment(inner, outer):
    """
    Compute what fraction of inner bbox is contained within outer bbox.
    Used for hierarchical linking: no-helmet contained in rider/motorcycle bbox.
    """
    x1 = max(inner[0], outer[0])
    y1 = max(inner[1], outer[1])
    x2 = min(inner[2], outer[2])
    y2 = min(inner[3], outer[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    inner_area = _bbox_area(inner)
    
    return inter / inner_area if inner_area > 0 else 0


def _is_inside_expanded(inner_box, outer_box, expand_ratio=0.3):
    """
    Check if inner bbox center is within an expanded version of outer bbox.
    Expansion allows for slight detection misalignment.
    """
    cx, cy = _bbox_center(inner_box)
    w = outer_box[2] - outer_box[0]
    h = outer_box[3] - outer_box[1]
    
    ex1 = outer_box[0] - w * expand_ratio
    ey1 = outer_box[1] - h * expand_ratio
    ex2 = outer_box[2] + w * expand_ratio
    ey2 = outer_box[3] + h * expand_ratio
    
    return ex1 <= cx <= ex2 and ey1 <= cy <= ey2


def _find_associated_motorcycle(nh_box, motorcycles):
    """
    Check if a without_helmet/rider detection is spatially associated with any motorcycle.
    CRITICAL DOMAIN GUARD: A helmet violation can ONLY occur for a motorcycle rider!
    Prevents false-positive traffic violations for pedestrians, coffee shop customers, bystanders, etc.
    
    Args:
        nh_box: bounding box [x1, y1, x2, y2] of without_helmet/rider
        motorcycles: list of (m_idx, tid, m_box, m_conf)
        
    Returns:
        (associated, tid, m_box, m_conf) or (False, -1, None, 0.0)
    """
    if not motorcycles:
        return False, -1, None, 0.0

    best_match = None
    best_score = -1.0

    nh_cx, nh_cy = _bbox_center(nh_box)

    for m_idx, tid, m_box, m_conf in motorcycles:
        m_x1, m_y1, m_x2, m_y2 = m_box
        m_w = max(1.0, m_x2 - m_x1)
        m_h = max(1.0, m_y2 - m_y1)

        # 1. Direct bounding box overlap / containment
        iou = _bbox_iou(nh_box, m_box)
        cont_nh = _bbox_containment(nh_box, m_box)
        cont_m = _bbox_containment(m_box, nh_box)
        inside = _is_inside_expanded(nh_box, m_box, expand_ratio=0.75)

        # 2. Vertical/Horizontal rider-above/on-motorcycle alignment
        # Head/torso is typically directly above, inside, or slightly offset on motorcycle
        horiz_ok = (m_x1 - m_w * 0.65 <= nh_cx <= m_x2 + m_w * 0.65)
        vert_ok = (m_y1 - m_h * 1.20 <= nh_cy <= m_y2 + m_h * 0.50)
        proximity_ok = horiz_ok and vert_ok

        if inside or cont_nh > 0.01 or cont_m > 0.01 or iou > 0.005 or proximity_ok:
            # Score based on spatial distance to center of motorcycle
            m_cx, m_cy = _bbox_center(m_box)
            dist = np.sqrt((nh_cx - m_cx)**2 + (nh_cy - m_cy)**2)
            score = 1.0 / (1.0 + dist)
            if score > best_score:
                best_score = score
                best_match = (True, tid, m_box, m_conf)

    if best_match:
        return best_match
    return False, -1, None, 0.0


# ══════════════════════════════════════════════════════════════════
# PER-TRACK STATE
# ══════════════════════════════════════════════════════════════════

track_state = defaultdict(lambda: {
    # Helmet violation state
    "no_helmet_votes": 0,
    "helmet_votes": 0,
    "total_frames": 0,
    "vote_history": [],
    "helmet_violation_confirmed": False,
    "last_box": None,
    "last_confidence": 0.0,
    # Triple riding state
    "rider_count_history": [],
    "triple_riding_confirmed": False,
})


# ══════════════════════════════════════════════════════════════════
# MULTI-CLASS VIOLATION CHECKING
# ══════════════════════════════════════════════════════════════════

def check_violations(detections, class_names):
    """
    Layer 4: Full violation analysis — handles both 2-class and multi-class models.
    
    For multi-class (Motorcycle, Rider, Helmet, No-Helmet, Plate):
      1. Find all motorcycles (tracked objects)
      2. For each motorcycle, find associated riders/helmets (spatial overlap & vehicle guard)
      3. Count riders per motorcycle → triple riding check
      4. Apply temporal voting for all violation types
    
    For 2-class (With Helmet, Without Helmet):
      → Falls back to simple temporal voting on each detection
    
    Args:
        detections: supervision Detections (with tracker_id from ByteTrack)
        class_names: dict mapping class_id → class_name
    
    Returns:
        list of confirmed violation dicts
    """
    if detections.tracker_id is None or len(detections) == 0:
        return []
    
    model_type = detect_model_type(class_names)
    
    if model_type == "multiclass":
        return _check_violations_multiclass(detections, class_names)
    elif model_type == "coco":
        return _check_violations_coco(detections, class_names)
    else:
        return _check_violations_2class(detections, class_names)


def _check_violations_multiclass(detections, class_names):
    """
    Full multi-class violation checking with hierarchical spatial linking & pedestrian guard.
    """
    violations = []
    
    # Group detections by category
    motorcycles = []  # (index, tid, bbox, conf)
    riders = []       # (index, bbox, conf)
    no_helmets = []   # (index, bbox, conf)
    helmets = []      # (index, bbox, conf)
    
    for i in range(len(detections)):
        cls_id = int(detections.class_id[i])
        cls_name = class_names.get(cls_id, "")
        bbox = detections.xyxy[i]
        conf = float(detections.confidence[i]) if detections.confidence is not None else 0.5
        tid = int(detections.tracker_id[i]) if detections.tracker_id is not None else -1
        
        if _is_motorcycle(cls_name):
            motorcycles.append((i, tid, bbox, conf))
        elif _is_rider(cls_name):
            riders.append((i, bbox, conf))
        elif _is_no_helmet(cls_name):
            no_helmets.append((i, bbox, conf))
        elif _is_helmet(cls_name):
            helmets.append((i, bbox, conf))
    
    # If no motorcycle is detected in the entire frame, NO helmet violations can occur!
    if not motorcycles:
        return violations
    
    # For each motorcycle, analyze associated riders and helmets
    for _, tid, moto_box, moto_conf in motorcycles:
        if tid < 0:
            continue
        
        state = track_state[tid]
        state["total_frames"] += 1
        state["last_box"] = moto_box
        state["last_confidence"] = moto_conf
        
        # ── Check if any without_helmet is spatially associated with THIS motorcycle ──
        has_no_helmet = False
        highest_nh_conf = moto_conf
        
        for _, nh_box, nh_conf in no_helmets:
            assoc, match_tid, match_box, _ = _find_associated_motorcycle(nh_box, motorcycles)
            if assoc and (match_tid == tid or np.array_equal(match_box, moto_box)):
                has_no_helmet = True
                highest_nh_conf = max(highest_nh_conf, nh_conf)
                break
        
        # ── Count riders/persons associated with THIS motorcycle ──
        associated_riders_count = 0
        all_people = riders + no_helmets + helmets
        for _, r_box, r_conf in all_people:
            assoc, match_tid, match_box, _ = _find_associated_motorcycle(r_box, motorcycles)
            if assoc and (match_tid == tid or np.array_equal(match_box, moto_box)):
                associated_riders_count += 1
        
        # ── TRIPLE RIDING CHECK (Slide 14) ──────────────────────────
        state["rider_count_history"].append(associated_riders_count)
        if len(state["rider_count_history"]) > TRIPLE_RIDING_WINDOW:
            state["rider_count_history"].pop(0)
        
        if (
            not state["triple_riding_confirmed"]
            and state["total_frames"] >= TRIPLE_RIDING_MIN_FRAMES
            and len(state["rider_count_history"]) >= TRIPLE_RIDING_MIN_FRAMES
        ):
            triple_frames = sum(1 for c in state["rider_count_history"] if c > 2)
            if triple_frames >= TRIPLE_RIDING_VOTE_THRESHOLD:
                state["triple_riding_confirmed"] = True
                violations.append({
                    "track_id": tid,
                    "type": "TRIPLE_RIDING",
                    "box": moto_box,
                    "confidence": moto_conf,
                    "rider_count": max(state["rider_count_history"]),
                    "votes": triple_frames,
                    "window": len(state["rider_count_history"]),
                })
                print(f"  [Layer 4] !! TRIPLE RIDING CONFIRMED: Track #{tid} "
                      f"({triple_frames}/{len(state['rider_count_history'])} frames with >2 riders)")
        
        # ── HELMET CHECK — Hierarchical Linking (Slide 14) ──────────
        if has_no_helmet:
            state["no_helmet_votes"] += 1
            state["vote_history"].append(True)
        else:
            state["helmet_votes"] += 1
            state["vote_history"].append(False)
        
        # Maintain sliding window
        if len(state["vote_history"]) > TEMPORAL_WINDOW:
            state["vote_history"].pop(0)
        
        # Temporal voting for helmet
        if (
            not state["helmet_violation_confirmed"]
            and state["total_frames"] >= MIN_FRAMES_BEFORE_DECISION
            and len(state["vote_history"]) >= MIN_FRAMES_BEFORE_DECISION
        ):
            violation_votes = sum(1 for v in state["vote_history"] if v)
            if violation_votes >= HELMET_VOTE_THRESHOLD:
                state["helmet_violation_confirmed"] = True
                violations.append({
                    "track_id": tid,
                    "type": "NO_HELMET",
                    "box": moto_box,
                    "confidence": highest_nh_conf,
                    "votes": violation_votes,
                    "window": len(state["vote_history"]),
                })
                print(f"  [Layer 4] !! NO HELMET CONFIRMED: Track #{tid} "
                      f"({violation_votes}/{len(state['vote_history'])} votes)")
    
    return violations


def _check_violations_coco(detections, class_names):
    """
    COCO model fallback: detects motorcycle + person but has NO helmet classes.
    """
    violations = []
    
    motorcycles = []  # (index, tid, bbox, conf)
    persons = []      # (index, bbox, conf)
    
    for i in range(len(detections)):
        cls_id = int(detections.class_id[i])
        cls_name = class_names.get(cls_id, "")
        bbox = detections.xyxy[i]
        conf = float(detections.confidence[i]) if detections.confidence is not None else 0.5
        tid = int(detections.tracker_id[i]) if detections.tracker_id is not None else -1
        
        if _is_motorcycle(cls_name):
            motorcycles.append((i, tid, bbox, conf))
        elif _is_rider(cls_name):  # _is_rider matches 'person'
            persons.append((i, bbox, conf))
            
    if not motorcycles:
        return violations
    
    for _, tid, moto_box, moto_conf in motorcycles:
        if tid < 0:
            continue
        
        state = track_state[tid]
        state["total_frames"] += 1
        state["last_box"] = moto_box
        state["last_confidence"] = moto_conf
        
        # Find persons near this motorcycle
        associated_persons = []
        for _, p_box, p_conf in persons:
            assoc, _, _, _ = _find_associated_motorcycle(p_box, motorcycles)
            if assoc:
                associated_persons.append((p_box, p_conf))
        
        rider_count = len(associated_persons)
        
        # ── TRIPLE RIDING CHECK ──────────────────────────────────────
        state["rider_count_history"].append(rider_count)
        if len(state["rider_count_history"]) > TRIPLE_RIDING_WINDOW:
            state["rider_count_history"].pop(0)
        
        if (
            not state["triple_riding_confirmed"]
            and state["total_frames"] >= TRIPLE_RIDING_MIN_FRAMES
            and len(state["rider_count_history"]) >= TRIPLE_RIDING_MIN_FRAMES
        ):
            triple_frames = sum(1 for c in state["rider_count_history"] if c > 2)
            if triple_frames >= TRIPLE_RIDING_VOTE_THRESHOLD:
                state["triple_riding_confirmed"] = True
                violations.append({
                    "track_id": tid,
                    "type": "TRIPLE_RIDING",
                    "box": moto_box,
                    "confidence": moto_conf,
                    "rider_count": max(state["rider_count_history"]),
                    "votes": triple_frames,
                    "window": len(state["rider_count_history"]),
                })
        
        # ── HELMET CHECK (COCO FALLBACK) ─────────────────────────────
        if associated_persons:
            state["no_helmet_votes"] += 1
            state["vote_history"].append(True)
        else:
            state["vote_history"].append(False)
        
        if len(state["vote_history"]) > TEMPORAL_WINDOW:
            state["vote_history"].pop(0)
        
        if (
            not state["helmet_violation_confirmed"]
            and state["total_frames"] >= MIN_FRAMES_BEFORE_DECISION
            and len(state["vote_history"]) >= MIN_FRAMES_BEFORE_DECISION
        ):
            violation_votes = sum(1 for v in state["vote_history"] if v)
            if violation_votes >= HELMET_VOTE_THRESHOLD:
                state["helmet_violation_confirmed"] = True
                violations.append({
                    "track_id": tid,
                    "type": "NO_HELMET",
                    "box": moto_box,
                    "confidence": moto_conf,
                    "votes": violation_votes,
                    "window": len(state["vote_history"]),
                })
    
    return violations


def _check_violations_2class(detections, class_names):
    """
    Simplified violation checking for 2-class model (With/Without Helmet).
    Uses direct temporal voting on each tracked detection.
    """
    violations = []
    
    for i in range(len(detections)):
        cls_id = int(detections.class_id[i])
        cls_name = class_names.get(cls_id, "")
        bbox = detections.xyxy[i]
        conf = float(detections.confidence[i]) if detections.confidence is not None else 0.5
        tid = int(detections.tracker_id[i]) if detections.tracker_id is not None else -1
        
        if tid < 0:
            continue
        
        state = track_state[tid]
        state["total_frames"] += 1
        state["last_box"] = bbox
        state["last_confidence"] = conf
        
        # Classify detection
        if _is_no_helmet(cls_name):
            state["no_helmet_votes"] += 1
            state["vote_history"].append(True)
        elif _is_helmet(cls_name):
            state["helmet_votes"] += 1
            state["vote_history"].append(False)
        else:
            state["vote_history"].append(False)
        
        # Maintain sliding window
        if len(state["vote_history"]) > TEMPORAL_WINDOW:
            state["vote_history"].pop(0)
        
        # Temporal voting
        if (
            not state["helmet_violation_confirmed"]
            and state["total_frames"] >= MIN_FRAMES_BEFORE_DECISION
            and len(state["vote_history"]) >= MIN_FRAMES_BEFORE_DECISION
        ):
            violation_votes = sum(1 for v in state["vote_history"] if v)
            if violation_votes >= HELMET_VOTE_THRESHOLD:
                state["helmet_violation_confirmed"] = True
                violations.append({
                    "track_id": tid,
                    "type": "NO_HELMET",
                    "box": bbox,
                    "confidence": conf,
                    "votes": violation_votes,
                    "window": len(state["vote_history"]),
                })
                print(f"  [Layer 4] !! NO HELMET CONFIRMED: Track #{tid} "
                      f"({violation_votes}/{len(state['vote_history'])} votes)")
    
    return violations


# ══════════════════════════════════════════════════════════════════
# SIMPLE IMAGE-MODE CHECKER (no tracking)
# ══════════════════════════════════════════════════════════════════

def check_violations_simple(detections, class_names):
    """
    Simplified violation check for image-by-image mode (no tracking).
    Directly flags 'Without Helmet' detections as NO_HELMET offenses.
    Links to motorcycle bounding box if available; otherwise flags directly on head/person box.
    """
    if detections.class_id is None or len(detections) == 0:
        return []
    
    violations = []
    no_helmets, helmets, riders, motorcycles = [], [], [], []
    
    for i in range(len(detections)):
        cls_name = class_names.get(int(detections.class_id[i]), "")
        bbox = detections.xyxy[i]
        conf = float(detections.confidence[i]) if detections.confidence is not None else 0.5
        tid = int(detections.tracker_id[i]) if detections.tracker_id is not None else -1
        
        if _is_no_helmet(cls_name):
            no_helmets.append((i, tid, bbox, conf))
        elif _is_helmet(cls_name):
            helmets.append((i, tid, bbox, conf))
        elif _is_rider(cls_name):
            riders.append((i, tid, bbox, conf))
        elif _is_motorcycle(cls_name):
            motorcycles.append((i, tid, bbox, conf))

    all_occupants = no_helmets + helmets + riders

    # 1. Standard explicit No-Helmet class check
    assigned_motorcycles = set()
    for _, nh_tid, nh_box, nh_conf in no_helmets:
        if motorcycles:
            assoc, match_tid, m_box, m_conf = _find_associated_motorcycle(nh_box, motorcycles)
            if assoc:
                m_key = tuple(m_box)
                if m_key not in assigned_motorcycles:
                    violations.append({
                        "track_id": match_tid if match_tid != -1 else nh_tid,
                        "type": "NO_HELMET",
                        "box": m_box,
                        "confidence": nh_conf,
                    })
                    assigned_motorcycles.add(m_key)
            else:
                # Flag directly on without_helmet box if not linked to existing motorcycle
                violations.append({
                    "track_id": nh_tid,
                    "type": "NO_HELMET",
                    "box": nh_box,
                    "confidence": nh_conf,
                })
        else:
            # No motorcycle box detected in image -> flag directly on without_helmet box
            violations.append({
                "track_id": nh_tid,
                "type": "NO_HELMET",
                "box": nh_box,
                "confidence": nh_conf,
            })
        
    # 2. Check Triple Riding
    # Case A: Motorcycles detected with >= 3 associated occupants
    triple_riding_motos = set()
    for _, m_tid, m_box, m_conf in motorcycles:
        assoc_occupants = [occ for occ in all_occupants if _find_associated_motorcycle(occ[2], [(0, m_tid, m_box, m_conf)])[0]]
        rider_count = len(assoc_occupants)
        if rider_count >= 3: 
            violations.append({
                "track_id": m_tid,
                "type": "TRIPLE_RIDING",
                "box": m_box,
                "confidence": m_conf,
                "rider_count": rider_count,
            })
            triple_riding_motos.add(tuple(m_box))

    # Case B: Spatial Cluster Fallback — if no motorcycle box was detected OR no motorcycle box had >=3 occupants,
    # check if >= 3 occupants form a spatial cluster in close proximity
    if len(all_occupants) >= 3 and not triple_riding_motos:
        boxes = [occ[2] for occ in all_occupants]
        confs = [occ[3] for occ in all_occupants]
        clusters = []
        for b in boxes:
            added = False
            b_cx, b_cy = _bbox_center(b)
            for cluster in clusters:
                if any(np.hypot(b_cx - _bbox_center(cb)[0], b_cy - _bbox_center(cb)[1]) < 350 for cb in cluster):
                    cluster.append(b)
                    added = True
                    break
            if not added:
                clusters.append([b])
        
        for cluster in clusters:
            if len(cluster) >= 3:
                min_x = min(cb[0] for cb in cluster)
                min_y = min(cb[1] for cb in cluster)
                max_x = max(cb[2] for cb in cluster)
                max_y = max(cb[3] for cb in cluster)
                cluster_w = max_x - min_x
                cluster_h = max_y - min_y
                enclosing_box = np.array([
                    max(0, min_x - cluster_w * 0.15),
                    max(0, min_y - cluster_h * 0.10),
                    max_x + cluster_w * 0.15,
                    max_y + cluster_h * 0.80
                ])
                avg_conf = float(np.mean(confs))
                violations.append({
                    "track_id": -1,
                    "type": "TRIPLE_RIDING",
                    "box": enclosing_box,
                    "confidence": avg_conf,
                    "rider_count": len(cluster),
                })

    return violations


# ══════════════════════════════════════════════════════════════════
# STATISTICS & STATE MANAGEMENT
# ══════════════════════════════════════════════════════════════════

def get_stats():
    """Return current pipeline statistics for dashboard overlay."""
    total_tracked = len(track_state)
    helmet_violations = sum(1 for s in track_state.values() if s["helmet_violation_confirmed"])
    triple_violations = sum(1 for s in track_state.values() if s["triple_riding_confirmed"])
    active_tracks = sum(1 for s in track_state.values() if s["total_frames"] > 0)
    
    return {
        "total_tracked": total_tracked,
        "helmet_violations": helmet_violations,
        "triple_violations": triple_violations,
        "active_tracks": active_tracks,
    }


def reset_state():
    """Reset all tracking state (used between runs)."""
    global track_state
    track_state.clear()
