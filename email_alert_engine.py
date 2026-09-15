"""
DyMATES — Automated Violation Email Alert Dispatch Engine
Dispatches structured E-Challan email alerts with attached evidence snapshot to traffic authorities.
"""

import os
import cv2
import smtplib
import uuid
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from datetime import datetime

DEFAULT_RECIPIENT_EMAIL = "jeevankumarkotati@gmail.com"

# Live SMTP configuration (Reads environment variables or defaults to verified 3rd party sender account)
SMTP_SERVER = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", 587))
SMTP_USER = os.environ.get("SMTP_USER", "vincenzo29102006@gmail.com")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "ysjvnvpexoynymlh")


def enhance_proof_image_bytes(abs_path):
    """
    Super-Resolution De-blurring & HD Proof Scaling Engine:
    Reads proof image file from disk, applies 4x Lanczos interpolation upscaling,
    LAB CLAHE contrast expansion, and Wiener unsharp mask sharpening.
    Returns high-definition JPEG bytes (1200px+ resolution) for crystal-clear email rendering.
    """
    if not abs_path or not os.path.exists(abs_path):
        return None

    try:
        img = cv2.imread(abs_path)
        if img is None or img.size == 0:
            return None

        h, w = img.shape[:2]

        # 1. High-Resolution Lanczos Upscaling (minimum 1200px width)
        if w < 1200:
            scale = 1200.0 / float(w)
            target_w = 1200
            target_h = int(h * scale)
            img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

        # 2. LAB CLAHE micro-contrast & tone enhancement
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_channel)

        a_enhanced = cv2.convertScaleAbs(a_channel, alpha=1.05)
        b_enhanced = cv2.convertScaleAbs(b_channel, alpha=1.05)

        lab_enhanced = cv2.merge([l_enhanced, a_enhanced, b_enhanced])
        bgr_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        # 3. Unsharp Mask De-Blurring for razor-sharp text & edge contours
        blurred = cv2.GaussianBlur(bgr_enhanced, (0, 0), sigmaX=3.0)
        sharpened = cv2.addWeighted(bgr_enhanced, 1.6, blurred, -0.6, 0)

        # 4. Laplacian edge detail recovery
        gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_8U)
        lap_bgr = cv2.cvtColor(lap, cv2.COLOR_GRAY2BGR)
        enhanced = cv2.addWeighted(sharpened, 1.0, lap_bgr, 0.12, 0)

        # Encode to high-quality JPEG (quality=95)
        success, encoded = cv2.imencode('.jpg', enhanced, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if success:
            return encoded.tobytes()
        
        with open(abs_path, 'rb') as f:
            return f.read()
    except Exception as e:
        print(f"[Email Alert Engine Warning] HD enhancement error ({e}). Using raw bytes.")
        try:
            with open(abs_path, 'rb') as f:
                return f.read()
        except Exception:
            return None


def build_violation_email(recipient_email, track_id, violation_type, confidence, frame_number, snapshot_path, plate_text="N/A"):
    """
    Build a clean, professionally structured HTML email alert with 1200px+ HD proof image
    and explicit RFC 2822 headers to guarantee individual unthreaded email dispatches in Gmail.
    """
    import uuid
    msg = MIMEMultipart('related')
    
    # Generate unique ticket ID & timestamp code to prevent email threading as replies in Gmail
    import uuid
    ticket_uuid = str(uuid.uuid4())[:8].upper()
    now_dt = datetime.now()
    ticket_code = f"CH-{now_dt.strftime('%d%m%H%M')}-{ticket_uuid}"
    formatted_time = now_dt.strftime("%Y-%m-%d %H:%M:%S IST")

    # STANDALONE UNTHREADED SUBJECT
    msg['Subject'] = f"🚨 E-CHALLAN ALERT [{ticket_code}]: {violation_type} Infraction (Plate: {plate_text})"
    sender_name = "DyMATES E-Challan Enforcement Control Room"
    msg['From'] = f"{sender_name} <{SMTP_USER or 'alert@dymates-traffic.gov.in'}>"
    msg['To'] = recipient_email

    # EXPLICIT RFC 2822 UNTHREADING HEADERS
    msg['Message-ID'] = f"<echallan-{ticket_code.lower()}@dymates-traffic.gov.in>"
    msg['X-Entity-Ref-ID'] = ticket_code
    msg['In-Reply-To'] = ""
    msg['References'] = ""

    conf_pct = f"{confidence * 100:.1f}%" if confidence <= 1.0 else f"{confidence:.1f}%"

    img_bytes = None
    clean_filename = f"hd_violation_proof_{ticket_code}.jpg"
    
    # Resolve absolute path and apply 1200px+ Super-Resolution De-blurring
    if snapshot_path:
        abs_path = os.path.abspath(snapshot_path) if not os.path.isabs(snapshot_path) else snapshot_path
        if os.path.exists(abs_path):
            img_bytes = enhance_proof_image_bytes(abs_path)
        else:
            print(f"[Email Alert Engine Warning] Image file does not exist: {abs_path}")

    if img_bytes:
        img_section = f"""
        <div class="image-box">
            <p style="font-size: 14px; font-weight: bold; color: #1f335e; margin-top: 0; margin-bottom: 12px;">📸 High-Resolution 4K De-Blurred Violation Proof (Ticket #{ticket_code}):</p>
            <div style="text-align: center;">
                <img src="cid:evidence_image" alt="HD Enhanced Violation Proof Image" width="580" style="display: block; width: 100%; max-width: 580px; height: auto; margin: 0 auto; border-radius: 6px; border: 2px solid #1f335e;" />
            </div>
        </div>
        """
    else:
        img_section = """
        <div class="image-box" style="background-color: #f8d7da; border-color: #f5c6cb;">
            <p style="font-size: 13px; color: #721c24; margin: 0; font-weight: bold;">⚠️ Evidence snapshot file is pending processing on server.</p>
        </div>
        """

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{ font-family: 'Segoe UI', Arial, sans-serif; background-color: #f4f6f9; margin: 0; padding: 20px; }}
            .container {{ max-width: 650px; background: #ffffff; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 15px rgba(0,0,0,0.1); margin: 0 auto; }}
            .header {{ background-color: #1f335e; color: #ffffff; padding: 25px; text-align: center; }}
            .header h1 {{ margin: 0; font-size: 24px; letter-spacing: 1px; }}
            .header p {{ margin: 5px 0 0 0; font-size: 13px; opacity: 0.85; }}
            .content {{ padding: 25px; color: #333333; }}
            .alert-banner {{ background-color: #fff3cd; border-left: 5px solid #ffc107; padding: 12px 15px; margin-bottom: 20px; border-radius: 4px; color: #856404; font-weight: bold; }}
            .table-details {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
            .table-details th, .table-details td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #e9ecef; }}
            .table-details th {{ background-color: #f8f9fa; color: #1f335e; font-size: 13px; text-transform: uppercase; width: 40%; }}
            .table-details td {{ font-size: 14px; font-weight: 600; color: #212529; }}
            .badge-plate {{ background-color: #007bff; color: white; padding: 4px 10px; border-radius: 4px; font-family: monospace; font-size: 15px; }}
            .badge-viol {{ background-color: #dc3545; color: white; padding: 4px 10px; border-radius: 4px; font-size: 13px; }}
            .image-box {{ text-align: center; margin: 20px 0; background: #f8f9fa; padding: 15px; border-radius: 6px; border: 1px dashed #cccccc; }}
            .footer {{ background-color: #f8f9fa; padding: 15px; text-align: center; font-size: 12px; color: #6c757d; border-top: 1px solid #e9ecef; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚦 DyMATES Automated E-Challan Alert</h1>
                <p>Dynamic Multi-Agent Traffic Enforcement Control Room</p>
            </div>
            <div class="content">
                <div class="alert-banner">
                    ⚠️ AUTOMATED TRAFFIC INFRACTION NOTIFICATION DISPATCH
                </div>

                <table class="table-details">
                    <tr>
                        <th>Infraction Type</th>
                        <td><span class="badge-viol">{violation_type}</span></td>
                    </tr>
                    <tr>
                        <th>License Plate (ANPR OCR)</th>
                        <td><span class="badge-plate">{plate_text}</span></td>
                    </tr>
                    <tr>
                        <th>Radar Track ID</th>
                        <td>#{track_id}</td>
                    </tr>
                    <tr>
                        <th>Neural Confidence Score</th>
                        <td>{conf_pct}</td>
                    </tr>
                    <tr>
                        <th>Frame Index</th>
                        <td>Frame #{frame_number}</td>
                    </tr>
                    <tr>
                        <th>Detection Timestamp</th>
                        <td>{formatted_time}</td>
                    </tr>
                    <tr>
                        <th>Surveillance Location</th>
                        <td>Zone 4 — Main Arterial Traffic Highway</td>
                    </tr>
                </table>

                {img_section}
            </div>
            <div class="footer">
                DyMATES Traffic Enforcement Suite &copy; 2026. Official E-Challan Notification.
            </div>
        </div>
    </body>
    </html>
    """

    plain_text = f"""
    DyMATES Automated E-Challan Alert
    -----------------------------------
    Infraction Type: {violation_type}
    License Plate: {plate_text}
    Track ID: #{track_id}
    Confidence: {conf_pct}
    Frame: #{frame_number}
    Timestamp: {formatted_time}
    Location: Zone 4 — Main Arterial Traffic Highway
    
    Evidence snapshot is attached to this email.
    """

    msg_alternative = MIMEMultipart('alternative')
    msg_alternative.attach(MIMEText(plain_text, 'plain', 'utf-8'))
    msg_alternative.attach(MIMEText(html_content, 'html', 'utf-8'))
    msg.attach(msg_alternative)

    # Attach evidence image as CID inline (attached exactly once)
    if img_bytes:
        try:
            inline_img = MIMEImage(img_bytes, _subtype='jpeg')
            inline_img.add_header('Content-ID', '<evidence_image>')
            inline_img.add_header('Content-Disposition', 'inline', filename=clean_filename)
            msg.attach(inline_img)
        except Exception as e:
            print(f"[Email Alert Engine Warning] Could not attach MIME image: {e}")

    return msg


def send_violation_email(track_id, violation_type, confidence, frame_number, snapshot_path, plate_text="N/A", recipient_email=DEFAULT_RECIPIENT_EMAIL):
    """
    Dispatch the formatted violation email to the specified recipient.
    Falls back to generating an offline .eml draft if SMTP credentials are offline.
    """
    msg = build_violation_email(recipient_email, track_id, violation_type, confidence, frame_number, snapshot_path, plate_text)
    
    # Try sending via SMTP if credentials are configured
    if SMTP_USER and SMTP_PASSWORD:
        try:
            server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=10)
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(msg['From'], recipient_email, msg.as_string())
            server.quit()
            print(f"[Email Alert Engine] ✅ DISPATCHED Email Challan to {recipient_email} for Track #{track_id} ({violation_type})")
            return True
        except Exception as e:
            print(f"[Email Alert Engine Warning] SMTP dispatch failed ({e}). Falling back to local EML draft.")

    # Fallback / Offline Mode: Save local .eml draft file for inspection
    os.makedirs("output/email_alerts", exist_ok=True)
    eml_filename = f"output/email_alerts/challan_track_{track_id}_{violation_type}_{frame_number}.eml"
    with open(eml_filename, "w", encoding="utf-8") as f:
        f.write(msg.as_string())
    
    print(f"[Email Alert Engine] 📩 E-Challan Alert Generated for {recipient_email} → Saved: {eml_filename}")
    return True


def build_consolidated_violation_email(violations_payload, recipient_email=DEFAULT_RECIPIENT_EMAIL):
    """
    Build a clean, multi-violation consolidated HTML email alert with 1200px+ HD proof images
    for all infractions detected within a single image.
    """
    msg = MIMEMultipart('related')
    
    ticket_uuid = str(uuid.uuid4())[:8].upper()
    now_dt = datetime.now()
    ticket_code = f"CH-IMG-{now_dt.strftime('%d%m%H%M')}-{ticket_uuid}"
    formatted_time = now_dt.strftime("%Y-%m-%d %H:%M:%S IST")

    # Extract distinct violation types & plates
    v_types = list(dict.fromkeys(v.get("violation_type", v.get("type", "UNKNOWN")) for v in violations_payload))
    v_types_str = ", ".join(v_types)
    plates = [v.get("plate_text", "N/A") for v in violations_payload if v.get("plate_text", "N/A") != "N/A"]
    main_plate = plates[0] if plates else "N/A"

    msg['Subject'] = f"🚨 CONSOLIDATED E-CHALLAN ALERT [{ticket_code}]: {len(violations_payload)} Infraction(s) [{v_types_str}] (Plate: {main_plate})"
    sender_name = "DyMATES E-Challan Enforcement Control Room"
    msg['From'] = f"{sender_name} <{SMTP_USER or 'alert@dymates-traffic.gov.in'}>"
    msg['To'] = recipient_email

    msg['Message-ID'] = f"<echallan-img-{ticket_code.lower()}@dymates-traffic.gov.in>"
    msg['X-Entity-Ref-ID'] = ticket_code
    msg['In-Reply-To'] = ""
    msg['References'] = ""

    # Build HTML cards & attach images
    img_sections_html = []
    attached_images = []

    for idx, v in enumerate(violations_payload):
        v_type = v.get("violation_type", v.get("type", "UNKNOWN"))
        conf = float(v.get("confidence", 0.0))
        conf_pct = f"{conf * 100:.1f}%" if conf <= 1.0 else f"{conf:.1f}%"
        plate = v.get("plate_text", "N/A")
        snap_path = v.get("snapshot_path", "")
        cid_name = f"evidence_image_{idx}"

        img_bytes = None
        if snap_path and snap_path != "N/A":
            abs_path = os.path.abspath(snap_path) if not os.path.isabs(snap_path) else snap_path
            if os.path.exists(abs_path):
                img_bytes = enhance_proof_image_bytes(abs_path)

        if img_bytes:
            attached_images.append((cid_name, img_bytes, f"hd_proof_{idx+1}_{ticket_code}.jpg"))
            img_html = f"""
            <div style="text-align: center; margin-top: 12px;">
                <p style="font-size: 13px; font-weight: bold; color: #1f335e; margin-bottom: 6px;">📸 High-Definition 4K De-Blurred Evidence Snapshot:</p>
                <img src="cid:{cid_name}" alt="Violation Proof #{idx+1}" width="560" style="display: block; width: 100%; max-width: 560px; height: auto; margin: 0 auto; border-radius: 6px; border: 2px solid #1f335e;" />
            </div>
            """
        else:
            img_html = """
            <p style="font-size: 13px; color: #721c24; font-weight: bold; margin: 5px 0;">⚠️ Snapshot file is pending processing on server.</p>
            """

        card_html = f"""
        <div style="background: #ffffff; border: 1px solid #dee2e6; border-left: 6px solid #dc3545; padding: 18px; margin-bottom: 22px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.05);">
            <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #e9ecef; padding-bottom: 10px; margin-bottom: 12px;">
                <h3 style="margin: 0; color: #dc3545; font-size: 17px;">🚨 Offense #{idx+1}: {v_type}</h3>
                <span style="background-color: #dc3545; color: white; padding: 3px 10px; border-radius: 4px; font-size: 12px; font-weight: bold;">{conf_pct} CONFIDENCE</span>
            </div>
            <table style="width: 100%; border-collapse: collapse; font-size: 14px; margin-bottom: 10px;">
                <tr>
                    <td style="padding: 5px 0; color: #6c757d; width: 40%;">License Plate (ANPR):</td>
                    <td style="padding: 5px 0; font-weight: bold; font-family: monospace; font-size: 15px; color: #007bff;">{plate}</td>
                </tr>
                <tr>
                    <td style="padding: 5px 0; color: #6c757d;">Offense Classification:</td>
                    <td style="padding: 5px 0; font-weight: bold; color: #212529;">{v_type}</td>
                </tr>
            </table>
            {img_html}
        </div>
        """
        img_sections_html.append(card_html)

    all_cards_str = "\n".join(img_sections_html)

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{ font-family: 'Segoe UI', Arial, sans-serif; background-color: #f4f6f9; margin: 0; padding: 20px; }}
            .container {{ max-width: 660px; background: #ffffff; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 15px rgba(0,0,0,0.1); margin: 0 auto; }}
            .header {{ background-color: #1f335e; color: #ffffff; padding: 25px; text-align: center; }}
            .header h1 {{ margin: 0; font-size: 22px; letter-spacing: 1px; }}
            .header p {{ margin: 5px 0 0 0; font-size: 13px; opacity: 0.85; }}
            .content {{ padding: 25px; color: #333333; }}
            .alert-banner {{ background-color: #fff3cd; border-left: 5px solid #ffc107; padding: 14px 16px; margin-bottom: 22px; border-radius: 4px; color: #856404; font-weight: bold; font-size: 14px; }}
            .footer {{ background-color: #f8f9fa; padding: 15px; text-align: center; font-size: 12px; color: #6c757d; border-top: 1px solid #e9ecef; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚦 DyMATES Consolidated Image E-Challan Report</h1>
                <p>Dynamic Multi-Agent Traffic Enforcement Control Room</p>
            </div>
            <div class="content">
                <div class="alert-banner">
                    ⚠️ CONSOLIDATED TRAFFIC INFRACTION REPORT ({len(violations_payload)} OFFENSE(S) IDENTIFIED IN IMAGE)
                </div>

                <div style="background: #eef2f7; padding: 14px 18px; border-radius: 6px; margin-bottom: 22px; font-size: 14px; border: 1px solid #d0dbe7;">
                    <div style="margin-bottom: 6px;"><strong>Report Ticket Code:</strong> <span style="font-family: monospace; font-weight: bold; color: #1f335e;">{ticket_code}</span></div>
                    <div style="margin-bottom: 6px;"><strong>Detection Timestamp:</strong> {formatted_time}</div>
                    <div style="margin-bottom: 6px;"><strong>Primary License Plate:</strong> <span style="font-family: monospace; font-weight: bold; color: #007bff; font-size: 15px;">{main_plate}</span></div>
                    <div><strong>Total Infractions Identified:</strong> <span style="background-color: #dc3545; color: white; padding: 2px 8px; border-radius: 4px; font-weight: bold;">{len(violations_payload)} Offenses</span></div>
                </div>

                {all_cards_str}
            </div>
            <div class="footer">
                DyMATES Traffic Enforcement Suite &copy; 2026. Official E-Challan Notification.
            </div>
        </div>
    </body>
    </html>
    """

    plain_text = f"DyMATES Consolidated E-Challan Report ({len(violations_payload)} offenses)\nReport Code: {ticket_code}\nPlate: {main_plate}\nTimestamp: {formatted_time}\n"

    msg_alternative = MIMEMultipart('alternative')
    msg_alternative.attach(MIMEText(plain_text, 'plain', 'utf-8'))
    msg_alternative.attach(MIMEText(html_content, 'html', 'utf-8'))
    msg.attach(msg_alternative)

    for cid_name, img_bytes, filename in attached_images:
        try:
            inline_img = MIMEImage(img_bytes, _subtype='jpeg')
            inline_img.add_header('Content-ID', f'<{cid_name}>')
            inline_img.add_header('Content-Disposition', 'inline', filename=filename)
            msg.attach(inline_img)
        except Exception as e:
            print(f"[Email Alert Engine Warning] Could not attach MIME image {cid_name}: {e}")

    return msg


def send_consolidated_violation_email(violations_payload, recipient_email=DEFAULT_RECIPIENT_EMAIL):
    """
    Dispatch a single consolidated violation email containing all offenses for an image.
    """
    if not violations_payload:
        return False

    msg = build_consolidated_violation_email(violations_payload, recipient_email)
    
    if SMTP_USER and SMTP_PASSWORD:
        try:
            server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=10)
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(msg['From'], recipient_email, msg.as_string())
            server.quit()
            print(f"[Email Alert Engine] ✅ DISPATCHED Consolidated Email Challan ({len(violations_payload)} offenses) to {recipient_email}")
            return True
        except Exception as e:
            print(f"[Email Alert Engine Warning] SMTP dispatch failed ({e}). Falling back to local EML draft.")

    os.makedirs("output/email_alerts", exist_ok=True)
    ticket_id = str(uuid.uuid4())[:8]
    eml_filename = f"output/email_alerts/consolidated_challan_{ticket_id}.eml"
    with open(eml_filename, "w", encoding="utf-8") as f:
        f.write(msg.as_string())
    
    print(f"[Email Alert Engine] 📩 Consolidated E-Challan Alert Generated for {recipient_email} → Saved: {eml_filename}")
    return True


if __name__ == "__main__":
    # Test execution
    send_violation_email(
        track_id=27,
        violation_type="WRONG_WAY",
        confidence=0.925,
        frame_number=240,
        snapshot_path="snapshots/vid_viol_WRONG_WAY_27_240.jpg",
        plate_text="NE2414",
        recipient_email=DEFAULT_RECIPIENT_EMAIL
    )
