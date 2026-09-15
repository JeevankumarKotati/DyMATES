import streamlit as st
import cv2
import tempfile
import os
import sqlite3
import pandas as pd
import time
import uuid
from datetime import datetime

# Local pipeline modules
from detector import DyMATESDetector
from violation_engine import check_violations, check_violations_simple, get_stats, reset_state
from dynamic_violations import check_dynamic_violations, get_dynamic_stats, reset_dynamic_state
from ocr_engine import extract_license_plate_text
from database import init_db, log_violation, log_image_violations_consolidated

# ---------------------------------------------------------
# UI Configuration
# ---------------------------------------------------------
st.set_page_config(page_title="DyMATES Dashboard", page_icon="🚥", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    MainMenu {visibility: hidden;} footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

init_db()
os.makedirs("snapshots", exist_ok=True)

# ---------------------------------------------------------
# Model Resolution — find the best available model
# ---------------------------------------------------------
def resolve_model_path():
    """Find the best available model weights, preferring trained models."""
    candidates = [
        "models/best.pt",
        "runs/detect/runs/dymates_v1/weights/best.pt",
        "runs/detect/runs/dymates_v1/weights/last.pt",
        "runs/dymates_v1/weights/best.pt",
        "runs/dymates_v1/weights/last.pt",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    # Fallback to pretrained COCO
    return "yolo11n.pt"


@st.cache_resource
def load_detector(conf, model_path):
    return DyMATESDetector(model_path=model_path, confidence=conf)

# ---------------------------------------------------------
# Session state for stop control
# ---------------------------------------------------------
if "stop_video" not in st.session_state:
    st.session_state.stop_video = False
if "processing" not in st.session_state:
    st.session_state.processing = False

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/8144/8144023.png", width=60)
    st.title("DyMATES Core")
    st.markdown("---")
    
    st.subheader("⚙️ Neural Settings")
    conf_thresh = st.slider("Detection Confidence", 0.1, 1.0, 0.25, 0.05)
    lane_dir = st.slider("Lane Vector Alignment (°)", 0, 360, 90)
    frame_skip = st.slider("Process Every Nth Frame", 1, 10, 2, 
                           help="Higher = faster but fewer frames analyzed. 1 = every frame.")
    
    st.markdown("---")
    model_path = resolve_model_path()
    model_name = os.path.basename(model_path)
    st.info(f"System Status: **ONLINE**\n\nModel: **{model_name}**\n\nDatabase: **Connected**")

detector = load_detector(conf_thresh, model_path)

# Show detected class names for debugging
with st.sidebar:
    with st.expander("🔍 Model Classes"):
        for cid, cname in detector.class_names.items():
            st.text(f"  {cid}: {cname}")

# ---------------------------------------------------------
# Top Header + Global Metrics
# ---------------------------------------------------------
col_title, col_met1, col_met2, col_met3 = st.columns([3, 1, 1, 1])

with col_title:
    st.title("🚦 DyMATES Intelligence")
    st.caption("Dynamic Multi-Agent Traffic Enforcement System")

with col_met1:
    conn = sqlite3.connect("violations.db")
    count_tickets = pd.read_sql_query("SELECT COUNT(*) FROM violations", conn).iloc[0, 0]
    conn.close()
    st.metric(label="Total Tickets Issued", value=str(count_tickets))

with col_met2:
    st.metric(label="Camera Feed", value="Live", delta="Active", delta_color="normal")

with col_met3:
    st.metric(label="Network", value="Secure", delta="Encrypted", delta_color="normal")

st.markdown("---")

# ---------------------------------------------------------
# App Layout: TABS
# ---------------------------------------------------------
tab_video, tab_image, tab_db = st.tabs(["▶ Live Video Processing", "📸 Image Analysis", "🗄️ Database Explorer"])

# ----------------- TAB 1: VIDEO -----------------
with tab_video:
    file = st.file_uploader("Upload Surveillance Footage (.MP4)", type=["mp4", "avi"])
    
    if file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(file.read())
        tfile.close()

        # --- Preview and controls ---
        col_vid, col_console = st.columns([2.5, 1.5])
        
        with col_vid:
            st.markdown("### 📡 Camera Feed")
            video_placeholder = st.empty()
            
            # Show video preview only when NOT processing
            if not st.session_state.processing:
                video_placeholder.video(tfile.name)
            
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                run_btn = st.button("🚀 Execute Intelligent Audit", use_container_width=True, type="primary")
            with btn_col2:
                stop_btn = st.button("⛔ Stop Processing", use_container_width=True, type="secondary")

        with col_console:
            st.markdown("### 📊 Live Analytics")
            m1, m2 = st.columns(2)
            fps_placeholder = m1.empty()
            trk_placeholder = m2.empty()
            fps_placeholder.metric("Processing FPS", "0.0")
            trk_placeholder.metric("Tracked Vehicles", "0")
            
            m3, m4 = st.columns(2)
            frame_placeholder = m3.empty()
            viol_count_placeholder = m4.empty()
            frame_placeholder.metric("Frame", "0")
            viol_count_placeholder.metric("Violations Found", "0")
            
            st.markdown("#### Detection Volume")
            chart_placeholder = st.empty()
            
            st.markdown("#### 🚨 LIVE ALERTS")
            alert_placeholder = st.empty()
            alert_placeholder.info("Awaiting traffic violations...", icon="👀")
            
        if stop_btn:
            st.session_state.stop_video = True

        if run_btn:
            st.session_state.stop_video = False
            st.session_state.processing = True
            
            reset_state()
            reset_dynamic_state()
            cap = cv2.VideoCapture(tfile.name)
            
            if not cap.isOpened():
                st.error("❌ Could not open video file. Please try a different file.")
            else:
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
                
                frame_count = 0
                processed_count = 0
                start_time = time.time()
                density_data = []
                logged_vid_vids = set()
                total_violations_found = 0
                
                # Progress bar
                progress_bar = st.progress(0, text="Processing video...")

                while cap.isOpened():
                    # Check stop signal
                    if st.session_state.stop_video:
                        st.warning("⚠️ Processing stopped by user.")
                        break
                    
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_count += 1
                    
                    # Frame skipping for performance
                    if frame_count % frame_skip != 0:
                        continue
                    
                    processed_count += 1
                    
                    # ── Full Pipeline: L1→L2→L3→L4 ──────────────────
                    detections = detector.detect_and_track(frame)
                    sv = check_violations(detections, detector.class_names)
                    dv = check_dynamic_violations(detections, detector.class_names, lane_dir)
                    all_violations = sv + dv
                    
                    # ── Annotation ───────────────────────────────────
                    annotated = detector.annotate_frame(frame.copy(), detections, all_violations)
                    video_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
                    
                    # ── Live Metrics ─────────────────────────────────
                    elapsed = time.time() - start_time
                    current_fps = processed_count / elapsed if elapsed > 0 else 0
                    
                    fps_placeholder.metric("Processing FPS", f"{current_fps:.1f}")
                    trk_placeholder.metric("Tracked Vehicles", str(len(detections)))
                    frame_placeholder.metric("Frame", f"{frame_count}/{total_frames}")
                    viol_count_placeholder.metric("Violations Found", str(total_violations_found))
                    
                    # Progress
                    if total_frames > 0:
                        progress_bar.progress(min(frame_count / total_frames, 1.0), 
                                            text=f"Frame {frame_count}/{total_frames}")
                    
                    # ── Chart ────────────────────────────────────────
                    density_data.append(len(detections))
                    if len(density_data) > 50:
                        density_data.pop(0)
                    if processed_count % 3 == 0:  # Update chart less frequently
                        chart_placeholder.area_chart(density_data, height=150)
                    
                    # ── Process ALL violations (not just last) ───────
                    for v in all_violations:
                        if "box" not in v or v["box"] is None:
                            continue
                            
                        bx = v["box"].astype(int)
                        plate = extract_license_plate_text(frame, bx)
                        total_violations_found += 1
                        
                        # Log to database (once per track per violation type)
                        trk_flag = v.get('track_id', -1)
                        v_key = (trk_flag, v['type'])
                        if v_key not in logged_vid_vids and trk_flag != -1:
                            snap_path = f"snapshots/vid_viol_{v['type']}_{trk_flag}_{frame_count}.jpg"
                            hd_snap = detector.extract_hd_violation_snapshot(frame, bx, v['type'], plate)
                            if hd_snap is not None and hd_snap.size > 0:
                                cv2.imwrite(snap_path, hd_snap)
                            else:
                                snap_path = "N/A"
                            log_violation(trk_flag, v['type'], frame_count, snap_path, 
                                        float(v.get('confidence', 0.0)), plate)
                            logged_vid_vids.add(v_key)

                        # Flash alert for latest violation
                        with alert_placeholder.container():
                            st.error(f"**🚨 VIOLATION CONFIRMED: {v['type']}**", icon="🚨")
                            ca, cb = st.columns(2)
                            ca.metric(label="OCR Plate Match", value=plate)
                            cb.metric(label="Radar Tracking ID", value=f"#{v.get('track_id', '-')}")
                            cc, cd = st.columns(2)
                            cc.metric(label="Confidence", value=f"{v.get('confidence', 0.0):.2f}")
                            cd.metric(label="Frame #", value=str(frame_count))
                            
                            hd_alert_snap = detector.extract_hd_violation_snapshot(frame, bx, v['type'], plate)
                            if hd_alert_snap is not None and hd_alert_snap.size > 0:
                                st.image(cv2.cvtColor(hd_alert_snap, cv2.COLOR_BGR2RGB), 
                                        caption="High-Definition Evidence Proof", width=340)

                cap.release()
                st.session_state.processing = False
                progress_bar.progress(1.0, text="Complete!")
                
                # ── Final Summary ────────────────────────────────
                st.success("✅ Audit Sequence Complete!")
                
                s_stats = get_stats()
                d_stats = get_dynamic_stats()
                
                sc1, sc2, sc3, sc4, sc5 = st.columns(5)
                sc1.metric("Frames Processed", processed_count)
                sc2.metric("Helmet Violations", s_stats.get('helmet_violations', 0))
                sc3.metric("Triple Riding", s_stats.get('triple_violations', 0))
                sc4.metric("Wrong-Way", d_stats.get('wrong_way_violations', 0))
                sc5.metric("Rash Driving", d_stats.get('rash_driving_violations', 0))

# ----------------- TAB 2: IMAGE -----------------
with tab_image:
    img_file = st.file_uploader("Upload Snapshot (.JPG / .PNG)", type=["jpg", "jpeg", "png"])
    
    if img_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        tfile.write(img_file.read())
        tfile.close()
        
        frame = cv2.imread(tfile.name)
        
        with st.spinner("Executing High-Resolution Automotive Surface Enhancement & Detection..."):
            # Layer 1: Enhance vehicle quality preserving geometry, stance, dynamic range, and background
            preprocessed = detector.preprocess(frame)
            detections = detector.detect(preprocessed)
            violations = check_violations_simple(detections, detector.class_names)
            
            annotated = preprocessed.copy()
            
            # Process all confirmed violations first
            violation_boxes = []
            image_violations_payload = []

            for v in violations:
                box = v["box"].astype(int)
                # ANPR OCR on enhanced high-res crop
                plate = extract_license_plate_text(preprocessed, box)
                conf = float(v.get('confidence', 0.0))
                uid = str(uuid.uuid4())[:6]
                snap_path = f"snapshots/violation_img_{v['type']}_{uid}.jpg"
                
                # Generate 900px+ High-Definition Contextual Evidence Snapshot
                hd_snap = detector.extract_hd_violation_snapshot(preprocessed, box, v['type'], plate)
                if hd_snap is not None and hd_snap.size > 0:
                    cv2.imwrite(snap_path, hd_snap)
                else: 
                    snap_path = "N/A"
                
                v['cached_plate'] = plate
                v['cached_snap'] = hd_snap
                violation_boxes.append(tuple(box))

                image_violations_payload.append({
                    "track_id": v.get("track_id", -1),
                    "violation_type": v["type"],
                    "confidence": conf,
                    "frame_number": 0,
                    "snapshot_path": snap_path,
                    "plate_text": plate,
                })

            # Log all violations to database and send ONE consolidated email for this image
            if image_violations_payload:
                log_image_violations_consolidated(image_violations_payload)

            # 1. Draw safe / normal objects (with_helmet, motorcycle, license_plate) in green/amber
            for i in range(len(detections)):
                cls_id = int(detections.class_id[i])
                cls_name = detector.class_names.get(cls_id, f"obj{cls_id}")
                conf = float(detections.confidence[i])
                box = detections.xyxy[i].astype(int)
                
                # Skip drawing raw without_helmet if it's already highlighted as a red violation below
                if any(abs(box[0]-vb[0]) < 10 and abs(box[1]-vb[1]) < 10 for vb in violation_boxes):
                    continue

                if "with_helmet" in cls_name or "helmet" in cls_name:
                    cv2.rectangle(annotated, (box[0], box[1]), (box[2], box[3]), (0, 220, 0), 2)
                    cv2.putText(annotated, f"[Safe] Helmet {conf:.2f}", (box[0], max(20, box[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 2)
                elif "motorcycle" in cls_name or "bike" in cls_name:
                    cv2.rectangle(annotated, (box[0], box[1]), (box[2], box[3]), (255, 165, 0), 2)
                    cv2.putText(annotated, f"Vehicle {conf:.2f}", (box[0], max(20, box[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)
                elif "plate" in cls_name or "license" in cls_name:
                    cv2.rectangle(annotated, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 2)
                    cv2.putText(annotated, f"Plate {conf:.2f}", (box[0], max(20, box[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            # 2. Draw confirmed violations in BRIGHT RED
            for v in violations:
                box = v["box"].astype(int)
                plate = v.get('cached_plate', 'N/A')
                conf = float(v.get('confidence', 0.0))
                cv2.rectangle(annotated, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 3)
                cv2.putText(annotated, f"🚨 {v['type']} ({conf:.2f}) | LP: {plate}", (box[0], max(25, box[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                
        # Layout
        st.markdown("### 🎯 Threat Assessment & Frame Analysis")
        colA, colB = st.columns([2, 1.2])
        
        with colA:
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="🔍 Rendered Detection Overlay (Proportional Fit)", width=620)
            
        with colB:
            if not violations:
                st.success("✅ Frame Clear. No violations detected.")
            else:
                st.error(f"🚨 ALERT: Identified {len(violations)} Offense(s)")
                for idx, v in enumerate(violations):
                    with st.expander(f"Offense #{idx+1}: {v['type']}", expanded=True):
                        m_col1, m_col2 = st.columns(2)
                        m_col1.metric("License Plate OCR", v.get('cached_plate', 'N/A'))
                        m_col2.metric("Neural Confidence", f"{v.get('confidence', 0.0):.2f}")
                        
                        if 'cached_snap' in v and v['cached_snap'].size > 0:
                            st.image(cv2.cvtColor(v['cached_snap'], cv2.COLOR_BGR2RGB), caption="Cropped Evidence Snapshot", width=280)

# ----------------- TAB 3: DATABASE -----------------
with tab_db:
    st.markdown("### 🗄️ System Database & Analytics Explorer")
    
    conn = sqlite3.connect("violations.db")
    df = pd.read_sql_query("SELECT id as Ticket_ID, track_id as Tracker_Hash, violation_type as Infraction, plate_text as License_Plate, timestamp as Time, snapshot_path as Filepath FROM violations ORDER BY id DESC", conn)
    conn.close()
    
    col_db1, col_db2 = st.columns([3, 1])
    with col_db1:
        st.dataframe(df, use_container_width=True, hide_index=True)
    with col_db2:
        st.markdown("#### 📥 Export Data")
        if not df.empty:
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📄 Download Official CSV Challans",
                data=csv_data,
                file_name="dyMATES_official_challans.csv",
                mime="text/csv",
                use_container_width=True
            )
            st.success(f"Total Records: **{len(df)}**")
        else:
            st.info("No records to export.")
            
    if not df.empty:
        st.markdown("---")
        ch_col1, ch_col2 = st.columns(2)
        with ch_col1:
            st.markdown("#### 📊 Violation Type Breakdown")
            v_counts = df['Infraction'].value_counts()
            st.bar_chart(v_counts)
        with ch_col2:
            st.markdown("#### 📸 Evidence Frame Snapshot")
            selected_id = st.selectbox("Select Ticket ID to view evidence:", df['Ticket_ID'].tolist())
            target = df[df['Ticket_ID'] == selected_id].iloc[0]
            
            if os.path.exists(target['Filepath']):
                st.image(target['Filepath'], width=600, caption=f"Ticket #{selected_id} | OCR Plate: {target['License_Plate']}")
            else:
                st.error("Evidence image file not found on local disk.")

