"""
DyMATES - Database Module (Layer 5: Evidence & Alert System)

Layer 5 Components:
  - SQLite-based violation evidence logging
  - Snapshot tracking with file paths
  - Violation summary reporting

Architecture Reference: Slide 16 (Evidence & Output)
"""

import sqlite3
import os
from datetime import datetime

DB_PATH = "violations.db"


def init_db():
    """Initialize the violations database with the required schema."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS violations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            track_id INTEGER NOT NULL,
            violation_type TEXT NOT NULL,
            confidence REAL,
            timestamp TEXT NOT NULL,
            frame_number INTEGER,
            snapshot_path TEXT,
            plate_text TEXT DEFAULT 'N/A',
            status TEXT DEFAULT 'PENDING'
        )
    """)
    conn.commit()
    conn.close()
    print("[Layer 5] SQLite database initialized (violations.db)")


def log_violation(track_id, violation_type, frame_number, snapshot_path, confidence=0.0, plate_text="N/A", recipient_email="jeevankumarkotati@gmail.com", skip_email=False):
    """Log a confirmed violation to the database with evidence AND optionally dispatch automated E-Challan email alert."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO violations 
           (track_id, violation_type, confidence, timestamp, frame_number, snapshot_path, plate_text) 
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            int(track_id),
            violation_type,
            round(confidence, 3),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            frame_number,
            snapshot_path,
            plate_text,
        ),
    )
    conn.commit()
    conn.close()
    print(f"  [Layer 5] DB LOG: Track #{track_id} → {violation_type} "
          f"(conf={confidence:.2f}, frame={frame_number}, plate={plate_text})")

    # Automated Email Alert Dispatch for single logged violation if not skipped
    if not skip_email:
        try:
            from email_alert_engine import send_violation_email
            send_violation_email(
                track_id=track_id,
                violation_type=violation_type,
                confidence=confidence,
                frame_number=frame_number,
                snapshot_path=snapshot_path,
                plate_text=plate_text,
                recipient_email=recipient_email
            )
        except Exception as e:
            print(f"  [Layer 5 Warning] Email alert dispatch error: {e}")


def log_image_violations_consolidated(violations_payload, recipient_email="jeevankumarkotati@gmail.com"):
    """
    Logs multiple violations from a single image to the database and dispatches ONE consolidated email alert.
    """
    if not violations_payload:
        return
    
    # 1. Log each violation to SQLite database without sending separate individual emails
    for v in violations_payload:
        log_violation(
            track_id=v.get("track_id", -1),
            violation_type=v.get("violation_type", v.get("type", "UNKNOWN")),
            frame_number=v.get("frame_number", 0),
            snapshot_path=v.get("snapshot_path", "N/A"),
            confidence=float(v.get("confidence", 0.0)),
            plate_text=v.get("plate_text", "N/A"),
            recipient_email=recipient_email,
            skip_email=True
        )

    # 2. Dispatch ONE consolidated email alert with all violations and proof snapshots
    try:
        from email_alert_engine import send_consolidated_violation_email
        send_consolidated_violation_email(
            violations_payload=violations_payload,
            recipient_email=recipient_email
        )
    except Exception as e:
        print(f"  [Layer 5 Warning] Consolidated email alert dispatch error: {e}")


_last_logged_times = {}

def log_violation_with_cooldown(track_id, violation_type, frame_number, snapshot_path, confidence=0.0, plate_text="N/A", cooldown_seconds=5.0, recipient_email="jeevankumarkotati@gmail.com"):
    """
    Log a violation only if it passes the anti-spam cooldown guard window.
    Prevents duplicate database entries and duplicate email alerts for the same vehicle track.
    """
    import time
    key = (track_id, violation_type)
    now = time.time()
    if track_id != -1 and key in _last_logged_times:
        if (now - _last_logged_times[key]) < cooldown_seconds:
            return False
    
    _last_logged_times[key] = now
    return log_violation(track_id, violation_type, frame_number, snapshot_path, confidence, plate_text, recipient_email)



def export_violations_csv(output_path="output/challans_report.csv"):
    """Export all logged database violations to a formatted CSV challan report."""
    import csv
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rows = get_all_violations()
    headers = ["Challan_ID", "Tracker_ID", "Violation_Type", "Confidence", "Timestamp", "Frame_Number", "Snapshot_Path", "License_Plate", "Status"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for r in rows:
            writer.writerow(r)
    print(f"[Layer 5] Exported {len(rows)} challan records to CSV: {output_path}")
    return output_path



def get_all_violations():
    """Retrieve all logged violations from the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM violations ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows


def get_violation_count():
    """Get total violation count."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM violations")
    count = cursor.fetchone()[0]
    conn.close()
    return count


def print_violations_summary():
    """Print a formatted summary table of all violations."""
    rows = get_all_violations()
    if not rows:
        print("\n[Layer 5] No violations recorded in database.\n")
        return

    print(f"\n{'='*95}")
    print(f"  DyMATES — Violation Evidence Database")
    print(f"{'='*95}")
    print(f"{'ID':<5} {'Track#':<8} {'Violation':<18} {'Conf':<8} {'Time':<22} {'Frame':<8} {'Plate':<12} {'Status'}")
    print(f"{'-'*95}")
    for row in rows:
        try:
            conf = float(row[3]) if row[3] is not None else 0.0
            conf_str = f"{conf:.3f}"
        except Exception:
            conf_str = str(row[3])
        plate = str(row[7]) if len(row) > 7 and row[7] is not None else "N/A"
        status = str(row[8]) if len(row) > 8 and row[8] is not None else "PENDING"
        print(
            f"{row[0]:<5} {row[1]:<8} {row[2]:<18} {conf_str:<8} {str(row[4]):<22} {row[5]:<8} {plate:<12} {status}"
        )
    print(f"{'='*95}")
    print(f"Total violations logged: {len(rows)}")
    print(f"Database: {DB_PATH}\n")


if __name__ == "__main__":
    init_db()
    print_violations_summary()
