"""
DyMATES - Dynamic Violation Detection Module (Layer 4b)

Implements behavioral analysis violations:
  1. Wrong-Way Detection  — vector angle >140° against lane direction (with Auto-Lane estimation)
  2. Rash Driving Detection — high lateral (x) variance = zig-zag pattern with EMA smoothing

Architecture Reference: DyMATES Slide 15 (Dynamic Violation Logic)
"""

import math
import numpy as np
from collections import defaultdict

# ══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════

# Wrong-Way Detection
WRONG_WAY_ANGLE_THRESHOLD = 140      # degrees — movement > this angle = wrong-way
WRONG_WAY_MIN_HISTORY = 6            # minimum centroid points before checking
WRONG_WAY_MIN_DISPLACEMENT = 25      # pixels — ignore micro-movements  
WRONG_WAY_VOTE_THRESHOLD = 4         # frames of wrong-way movement to confirm

# Rash Driving Detection
RASH_DRIVING_WINDOW = 30             # frames to analyze
RASH_DRIVING_VARIANCE_THRESHOLD = 450  # x-axis variance threshold (pixels²)
RASH_DRIVING_MIN_HISTORY = 12        # minimum frames before checking
RASH_DRIVING_VOTE_THRESHOLD = 6      # frames with high variance to confirm

DEFAULT_LANE_DIRECTION = 90.0  # degrees (90° = downward, 0° = rightward)

# ══════════════════════════════════════════════════════════════════
# PER-TRACK TRAJECTORY STATE & DOMINANT FLOW TRACKER
# ══════════════════════════════════════════════════════════════════

trajectory_state = defaultdict(lambda: {
    "raw_centroids": [],                # raw (cx, cy)
    "centroids": [],                    # EMA smoothed (cx, cy)
    "angles": [],                       # movement angles
    "wrong_way_votes": 0,               # count of wrong-way frames
    "wrong_way_confirmed": False,
    "rash_driving_votes": 0,            # count of high-variance frames
    "rash_driving_confirmed": False,
    "total_frames": 0,
    "last_box": None,
})

_global_vehicle_displacements = []


def estimate_dominant_lane_direction():
    """
    Auto-detect dominant traffic flow direction from all vehicle displacement vectors.
    Enables automatic adaptation to curved roads and non-standard camera angles.
    """
    if len(_global_vehicle_displacements) < 5:
        return DEFAULT_LANE_DIRECTION

    dxs = [d[0] for d in _global_vehicle_displacements[-50:]]
    dys = [d[1] for d in _global_vehicle_displacements[-50:]]
    
    mean_dx = np.mean(dxs)
    mean_dy = np.mean(dys)
    
    if abs(mean_dx) < 1.0 and abs(mean_dy) < 1.0:
        return DEFAULT_LANE_DIRECTION
        
    angle = math.degrees(math.atan2(mean_dy, mean_dx))
    if angle < 0:
        angle += 360
    return angle


# ══════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════

def _angle_difference(angle1, angle2):
    """Compute the absolute angular difference between two angles in [0, 180]."""
    diff = abs(angle1 - angle2) % 360
    if diff > 180:
        diff = 360 - diff
    return diff


def _smooth_centroid(prev_centroid, current_centroid, alpha=0.65):
    """Apply Exponential Moving Average (EMA) smoothing to eliminate detection jitter."""
    if prev_centroid is None:
        return current_centroid
    cx_s = alpha * current_centroid[0] + (1 - alpha) * prev_centroid[0]
    cy_s = alpha * current_centroid[1] + (1 - alpha) * prev_centroid[1]
    return (cx_s, cy_s)


# ══════════════════════════════════════════════════════════════════
# WRONG-WAY DETECTION
# ══════════════════════════════════════════════════════════════════

def check_wrong_way(track_id, state, lane_direction=DEFAULT_LANE_DIRECTION):
    """
    Wrong-Way Detection Algorithm with EMA smoothed centroids and vector analysis.
    """
    centroids = state["centroids"]
    
    if len(centroids) < WRONG_WAY_MIN_HISTORY:
        return None
    
    if state["wrong_way_confirmed"]:
        return None
    
    start = centroids[0]
    end = centroids[-1]
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    displacement = math.sqrt(dx**2 + dy**2)
    
    if displacement < WRONG_WAY_MIN_DISPLACEMENT:
        return None
    
    # Store displacement vector for auto-lane estimation
    if state["total_frames"] % 5 == 0:
        _global_vehicle_displacements.append((dx, dy))
        if len(_global_vehicle_displacements) > 200:
            _global_vehicle_displacements.pop(0)

    # Use auto-estimated lane direction if default or negative passed
    target_lane_dir = lane_direction if (lane_direction is not None and lane_direction >= 0) else estimate_dominant_lane_direction()

    # Compute smoothed movement angle
    recent = centroids[-8:]
    avg_dx = sum(recent[i+1][0] - recent[i][0] for i in range(len(recent)-1)) / max(1, len(recent)-1)
    avg_dy = sum(recent[i+1][1] - recent[i][1] for i in range(len(recent)-1)) / max(1, len(recent)-1)
    
    if abs(avg_dx) < 0.3 and abs(avg_dy) < 0.3:
        return None
    
    movement_angle = math.degrees(math.atan2(avg_dy, avg_dx))
    if movement_angle < 0:
        movement_angle += 360
    
    angle_diff = _angle_difference(movement_angle, target_lane_dir)
    
    if angle_diff > WRONG_WAY_ANGLE_THRESHOLD:
        state["wrong_way_votes"] += 1
    else:
        state["wrong_way_votes"] = max(0, state["wrong_way_votes"] - 1)
    
    if state["wrong_way_votes"] >= WRONG_WAY_VOTE_THRESHOLD:
        state["wrong_way_confirmed"] = True
        print(f"  [Layer 4b] !! WRONG-WAY CONFIRMED: Track #{track_id} "
              f"(angle_diff={angle_diff:.1f}°, votes={state['wrong_way_votes']})")
        return {
            "track_id": track_id,
            "type": "WRONG_WAY",
            "box": state["last_box"],
            "confidence": min(angle_diff / 180.0, 1.0),
            "angle_diff": angle_diff,
            "votes": state["wrong_way_votes"],
        }
    
    return None


# ══════════════════════════════════════════════════════════════════
# RASH DRIVING DETECTION
# ══════════════════════════════════════════════════════════════════

def check_rash_driving(track_id, state):
    """
    Rash Driving Detection Algorithm analyzing smoothed lateral variance and path jitter.
    """
    centroids = state["centroids"]
    
    if len(centroids) < RASH_DRIVING_MIN_HISTORY:
        return None
    
    if state["rash_driving_confirmed"]:
        return None
    
    window = centroids[-RASH_DRIVING_WINDOW:]
    x_coords = [c[0] for c in window]
    x_variance = np.var(x_coords)
    
    if len(window) >= 3:
        direction_changes = 0
        for i in range(2, len(window)):
            dx1 = window[i-1][0] - window[i-2][0]
            dx2 = window[i][0] - window[i-1][0]
            if dx1 * dx2 < 0:
                direction_changes += 1
        jitter_ratio = direction_changes / (len(window) - 2)
    else:
        jitter_ratio = 0.0
    
    is_rash = (x_variance > RASH_DRIVING_VARIANCE_THRESHOLD) or (jitter_ratio > 0.38)
    
    if is_rash:
        state["rash_driving_votes"] += 1
    else:
        state["rash_driving_votes"] = max(0, state["rash_driving_votes"] - 1)
    
    if state["rash_driving_votes"] >= RASH_DRIVING_VOTE_THRESHOLD:
        state["rash_driving_confirmed"] = True
        print(f"  [Layer 4b] !! RASH DRIVING CONFIRMED: Track #{track_id} "
              f"(x_var={x_variance:.1f}, jitter={jitter_ratio:.2f})")
        return {
            "track_id": track_id,
            "type": "RASH_DRIVING",
            "box": state["last_box"],
            "confidence": min(x_variance / (RASH_DRIVING_VARIANCE_THRESHOLD * 2), 1.0),
            "x_variance": float(x_variance),
            "jitter_ratio": float(jitter_ratio),
            "votes": state["rash_driving_votes"],
        }
    
    return None


# ══════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════

def check_dynamic_violations(detections, class_names, lane_direction=DEFAULT_LANE_DIRECTION):
    """
    Check all tracked objects for dynamic violations (wrong-way + rash driving).
    """
    if detections.tracker_id is None or len(detections) == 0:
        return []
    
    violations = []
    
    for i in range(len(detections)):
        tid = int(detections.tracker_id[i]) if detections.tracker_id is not None else -1
        if tid < 0:
            continue
        
        bbox = detections.xyxy[i]
        cls_id = int(detections.class_id[i])
        cls_name = class_names.get(cls_id, "").lower()
        
        is_vehicle = True
        if any(kw in cls_name for kw in ["helmet", "plate", "number"]):
            is_vehicle = False
        
        if not is_vehicle:
            continue
        
        state = trajectory_state[tid]
        state["total_frames"] += 1
        state["last_box"] = bbox
        
        raw_cx = (bbox[0] + bbox[2]) / 2.0
        raw_cy = (bbox[1] + bbox[3]) / 2.0
        state["raw_centroids"].append((raw_cx, raw_cy))
        
        prev_smoothed = state["centroids"][-1] if state["centroids"] else None
        smoothed = _smooth_centroid(prev_smoothed, (raw_cx, raw_cy), alpha=0.65)
        state["centroids"].append(smoothed)
        
        if len(state["centroids"]) > 120:
            state["raw_centroids"].pop(0)
            state["centroids"].pop(0)
        
        ww = check_wrong_way(tid, state, lane_direction)
        if ww:
            violations.append(ww)
        
        rd = check_rash_driving(tid, state)
        if rd:
            violations.append(rd)
    
    return violations


def get_dynamic_stats():
    """Return statistics about dynamic violation tracking."""
    total = len(trajectory_state)
    wrong_way = sum(1 for s in trajectory_state.values() if s["wrong_way_confirmed"])
    rash = sum(1 for s in trajectory_state.values() if s["rash_driving_confirmed"])
    estimated_angle = estimate_dominant_lane_direction()
    return {
        "trajectories_tracked": total,
        "wrong_way_violations": wrong_way,
        "rash_driving_violations": rash,
        "estimated_lane_angle": round(estimated_angle, 1),
    }


def reset_dynamic_state():
    """Reset all trajectory tracking state."""
    global trajectory_state, _global_vehicle_displacements
    trajectory_state.clear()
    _global_vehicle_displacements.clear()
