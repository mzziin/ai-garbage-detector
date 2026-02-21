"""
AI-Based Illegal Garbage Dumping Detection System — Streamlit Dashboard

Run with: streamlit run app.py
"""

import streamlit as st
import os
import sys
import json
import subprocess
import time
import uuid
from datetime import datetime, timedelta
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from config import DEVICE, DETECTION_MODEL, POSE_MODEL, UPLOADS_DIR, EVIDENCE_DIR
from database import (
    init_db, get_cameras, get_camera, get_incident_count, get_incidents,
    get_incident, get_video_uploads, get_video_upload, get_video_detections
)
from camera_manager import add_camera, remove_camera, list_cameras, get_camera_source, test_camera_source

# ============================================================
# Page Config
# ============================================================
st.set_page_config(
    page_title="Garbage Dump Detection System",
    page_icon="🗑️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database
init_db()

# Session state for monitor processes
if "monitor_processes" not in st.session_state:
    st.session_state.monitor_processes = {}


# ============================================================
# Sidebar Navigation
# ============================================================
st.sidebar.title("🗑️ Garbage Detection")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["📊 Dashboard", "📹 Live Monitor", "📤 Video Upload", "📷 Camera Management", "🔍 Incident Viewer"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Device:** `{DEVICE.upper()}`")
st.sidebar.markdown(f"**Detection Model:** `{DETECTION_MODEL}`")
st.sidebar.markdown(f"**Pose Model:** `{POSE_MODEL}`")


# ============================================================
# Dashboard Page
# ============================================================
def page_dashboard():
    st.title("📊 Dashboard")
    st.markdown("Real-time overview of garbage dumping detection system.")

    # Stats cards
    col1, col2, col3, col4 = st.columns(4)

    total_incidents = get_incident_count(today_only=False)
    today_incidents = get_incident_count(today_only=True)
    cameras = get_cameras()
    uploads = get_video_uploads()

    with col1:
        st.metric("Total Incidents", total_incidents)
    with col2:
        st.metric("Today's Incidents", today_incidents)
    with col3:
        st.metric("Registered Cameras", len(cameras))
    with col4:
        st.metric("Video Uploads", len(uploads))

    st.markdown("---")

    # Recent incidents
    st.subheader("Recent Incidents")
    incidents = get_incidents(limit=20)

    if incidents:
        for incident in incidents:
            with st.expander(
                f"🚨 Incident #{incident['id']} — "
                f"{incident['timestamp']} — "
                f"Confidence: {incident['confidence']:.1%}" if incident['confidence'] else
                f"🚨 Incident #{incident['id']} — {incident['timestamp']}"
            ):
                col_img, col_info = st.columns([1, 2])

                with col_img:
                    if incident["snapshot_path"] and os.path.exists(incident["snapshot_path"]):
                        img = Image.open(incident["snapshot_path"])
                        st.image(img, caption="Evidence Snapshot", use_container_width=True)
                    else:
                        st.info("No snapshot available")

                with col_info:
                    st.markdown(f"**Description:** {incident['description']}")
                    st.markdown(f"**Camera ID:** {incident['camera_id']}")
                    st.markdown(f"**Confidence:** {incident['confidence']:.1%}" if incident['confidence'] else "**Confidence:** N/A")
                    if incident["objects_detected"]:
                        try:
                            objects = json.loads(incident["objects_detected"])
                            st.markdown(f"**Objects Detected:** {', '.join(objects)}")
                        except (json.JSONDecodeError, TypeError):
                            st.markdown(f"**Objects Detected:** {incident['objects_detected']}")
    else:
        st.info("No incidents detected yet. Start a live monitor or upload a video to begin detection.")

    # Incident trend chart
    if incidents:
        st.markdown("---")
        st.subheader("Incident Trend")

        df = pd.DataFrame(incidents)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["date"] = df["timestamp"].dt.date

        daily_counts = df.groupby("date").size().reset_index(name="incidents")
        fig = px.bar(daily_counts, x="date", y="incidents",
                     title="Incidents Per Day",
                     labels={"date": "Date", "incidents": "Number of Incidents"})
        fig.update_layout(xaxis_title="Date", yaxis_title="Incidents")
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Live Monitor Page
# ============================================================
def page_live_monitor():
    st.title("📹 Live Monitor")

    cameras = list_cameras()

    if not cameras:
        st.warning("No cameras registered. Go to **Camera Management** to add one first.")
        return

    # Camera selection
    camera_options = {f"{c['name']} ({c['source_type']})": c['id'] for c in cameras}
    selected_label = st.selectbox("Select Camera", list(camera_options.keys()))
    selected_camera_id = camera_options[selected_label]
    selected_camera = get_camera(selected_camera_id)

    # Choose mode based on camera type
    is_webcam = selected_camera["source_type"] == "webcam"

    if is_webcam:
        st.info(
            "**Webcam detected** — The live feed will open in a **separate OpenCV window** "
            "(outside the browser). Press **Q** in that window to stop."
        )
    else:
        st.info(
            "**IP / Mobile camera detected** — The live feed will stream "
            "**directly in the dashboard** below with real-time detection overlays."
        )

    st.markdown("---")

    # ---- Mode A: OpenCV window (webcam) ----
    if is_webcam:
        _render_opencv_monitor(selected_camera_id, selected_camera)
    # ---- Mode B: In-dashboard stream (IP / mobile / RTSP) ----
    else:
        _render_dashboard_stream(selected_camera_id, selected_camera)


def _render_opencv_monitor(selected_camera_id, selected_camera):
    """Launch the live_monitor.py subprocess for webcam feeds."""
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🟢 Start Live Monitor", type="primary", use_container_width=True):
            source = get_camera_source(selected_camera_id)
            if source is None:
                st.error("Could not determine camera source.")
                return

            cmd = [
                sys.executable, "live_monitor.py",
                "--source", str(source),
                "--camera-id", str(selected_camera_id),
                "--camera-name", selected_camera["name"]
            ]

            try:
                process = subprocess.Popen(
                    cmd,
                    cwd=os.path.dirname(os.path.abspath(__file__)),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                )
                st.session_state.monitor_processes[selected_camera_id] = {
                    "process": process,
                    "pid": process.pid,
                    "camera_name": selected_camera["name"],
                    "started_at": datetime.now().strftime("%H:%M:%S")
                }
                st.success(
                    f"✅ Live Monitor launched for **{selected_camera['name']}**! "
                    f"Look for the OpenCV window. Press **Q** in that window to stop."
                )
            except Exception as e:
                st.error(f"Failed to launch monitor: {e}")

    with col2:
        if st.button("🔴 Stop All Monitors", use_container_width=True):
            stopped = 0
            for cam_id, info in list(st.session_state.monitor_processes.items()):
                proc = info["process"]
                if proc.poll() is None:
                    proc.terminate()
                    stopped += 1
                del st.session_state.monitor_processes[cam_id]
            if stopped:
                st.info(f"Stopped {stopped} monitor(s).")
            else:
                st.info("No active monitors to stop.")

    # Show active monitors
    st.markdown("---")
    st.subheader("Active Monitors")

    active = False
    for cam_id, info in list(st.session_state.monitor_processes.items()):
        proc = info["process"]
        is_running = proc.poll() is None

        if is_running:
            active = True
            st.markdown(
                f"🟢 **{info['camera_name']}** — PID: {info['pid']} — "
                f"Started: {info['started_at']}"
            )
        else:
            st.markdown(
                f"⚫ **{info['camera_name']}** — Stopped"
            )
            del st.session_state.monitor_processes[cam_id]

    if not active:
        st.info("No active monitors. Select a camera and click Start.")


def _render_dashboard_stream(selected_camera_id, selected_camera):
    """Stream IP/mobile camera feed directly in the Streamlit dashboard."""
    import cv2
    import numpy as np

    col1, col2 = st.columns(2)
    with col1:
        start_stream = st.button("🟢 Start Live Stream", type="primary", use_container_width=True)
    with col2:
        stop_stream = st.button("🔴 Stop Stream", use_container_width=True)

    if stop_stream:
        st.session_state.pop("dashboard_streaming", None)
        st.info("Stream stopped.")
        return

    if start_stream:
        st.session_state["dashboard_streaming"] = True

    if not st.session_state.get("dashboard_streaming", False):
        st.markdown("---")
        st.info("Click **Start Live Stream** to begin viewing the camera feed with detection.")
        st.subheader("Camera Info")
        st.json({
            "ID": selected_camera["id"],
            "Name": selected_camera["name"],
            "Type": selected_camera["source_type"],
            "Source": selected_camera.get("source_url") or "N/A",
        })
        return

    # --- Streaming active ---
    st.markdown("---")

    source = get_camera_source(selected_camera_id)
    if source is None:
        st.error("Could not determine camera source.")
        return

    # Status indicators
    status_col1, status_col2, status_col3 = st.columns(3)
    status_text = status_col1.empty()
    fps_text = status_col2.empty()
    detection_text = status_col3.empty()

    status_text.markdown("🟡 **Connecting...**")

    # Frame display area
    frame_placeholder = st.empty()

    # Load models (cached in session state to avoid reloading on every rerun)
    if "detector_instance" not in st.session_state:
        with st.spinner("Loading AI models (first time only)..."):
            from detector import Detector
            st.session_state["detector_instance"] = Detector()

    detector = st.session_state["detector_instance"]

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        st.error(f"❌ Could not connect to camera: {source}")
        st.session_state.pop("dashboard_streaming", None)
        return

    # Warm-up
    for _ in range(5):
        cap.read()

    status_text.markdown("🟢 **Live**")
    prev_time = time.time()

    # Performance settings
    INFER_EVERY_N = 3          # Run AI every Nth frame; show raw feed in between
    INFER_WIDTH = 640          # Resize frame to this width before inference
    USE_POSE = (DEVICE == "cuda")  # Pose estimation only on GPU (too slow on CPU)

    frame_count = 0
    last_persons = []
    last_waste = []
    last_poses = []
    last_alert = False
    scale_x, scale_y = 1.0, 1.0  # Scale factors to map detections back to original

    from config import PROXIMITY_THRESHOLD

    try:
        while st.session_state.get("dashboard_streaming", False):
            ret, frame = cap.read()
            if not ret:
                status_text.markdown("🟡 **Reconnecting...**")
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(source)
                continue

            frame_count += 1

            # Calculate FPS
            now = time.time()
            fps = 1.0 / max(now - prev_time, 0.001)
            prev_time = now
            fps_text.markdown(f"**FPS:** {fps:.1f}")

            # Run AI inference only every Nth frame
            if frame_count % INFER_EVERY_N == 0:
                orig_h, orig_w = frame.shape[:2]
                ratio = INFER_WIDTH / orig_w
                small_h = int(orig_h * ratio)
                small = cv2.resize(frame, (INFER_WIDTH, small_h))

                scale_x = orig_w / INFER_WIDTH
                scale_y = orig_h / small_h

                detections = detector.detect_for_video(small)
                last_persons = detections["persons"]
                last_waste = detections["waste"]

                # Scale boxes back to original resolution
                for det in last_persons + last_waste:
                    det["box"] = [
                        int(det["box"][0] * scale_x),
                        int(det["box"][1] * scale_y),
                        int(det["box"][2] * scale_x),
                        int(det["box"][3] * scale_y),
                    ]

                if USE_POSE:
                    last_poses = detector.detect_pose_for_video(small)
                    for pose in last_poses:
                        pose["keypoints"][:, 0] *= scale_x
                        pose["keypoints"][:, 1] *= scale_y
                else:
                    last_poses = []

                # Proximity check
                last_alert = False
                if last_persons and last_waste:
                    for w in last_waste:
                        wx = (w["box"][0] + w["box"][2]) // 2
                        wy = (w["box"][1] + w["box"][3]) // 2
                        for p in last_persons:
                            px = (p["box"][0] + p["box"][2]) // 2
                            py = (p["box"][1] + p["box"][3]) // 2
                            dist = ((wx - px)**2 + (wy - py)**2) ** 0.5
                            if dist <= PROXIMITY_THRESHOLD:
                                last_alert = True
                                break
                        if last_alert:
                            break

                detection_text.markdown(
                    f"**Persons:** {len(last_persons)} | **Waste:** {len(last_waste)}"
                )

            # Draw cached detections on current frame
            display = frame.copy()

            for det in last_persons:
                x1, y1, x2, y2 = det["box"]
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display, f"Person ({det['confidence']:.2f})",
                            (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            for det in last_waste:
                x1, y1, x2, y2 = det["box"]
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(display, f"{det['class_name']} ({det['confidence']:.2f})",
                            (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Draw pose skeletons
            skeleton = [
                (5, 7), (7, 9), (6, 8), (8, 10), (5, 6),
                (5, 11), (6, 12), (11, 12),
                (11, 13), (13, 15), (12, 14), (14, 16)
            ]
            for pose in last_poses:
                kps = pose["keypoints"]
                for i, j in skeleton:
                    if kps[i][2] > 0.3 and kps[j][2] > 0.3:
                        pt1 = (int(kps[i][0]), int(kps[i][1]))
                        pt2 = (int(kps[j][0]), int(kps[j][1]))
                        cv2.line(display, pt1, pt2, (255, 165, 0), 2)

            # Alert overlay
            if last_alert:
                h, ww = display.shape[:2]
                cv2.rectangle(display, (0, 0), (ww - 1, h - 1), (0, 0, 255), 6)
                cv2.putText(display, "!! POTENTIAL DUMPING !!",
                            (ww // 2 - 200, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

            # OSD
            mode = "AI" if frame_count % INFER_EVERY_N == 0 else "PASS"
            cv2.putText(display, f"FPS: {fps:.1f} | {DEVICE.upper()} | {mode}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Convert BGR to RGB for Streamlit
            display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(display_rgb, channels="RGB", use_container_width=True)

    except Exception as e:
        st.error(f"Stream error: {e}")
    finally:
        cap.release()
        st.session_state.pop("dashboard_streaming", None)
        status_text.markdown("⚫ **Stopped**")


# ============================================================
# Video Upload Page
# ============================================================
def page_video_upload():
    st.title("📤 Video Upload & Analysis")
    st.markdown("Upload a video file to analyze it for garbage dumping events.")

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mkv", "mov", "wmv"],
        help="Supported formats: MP4, AVI, MKV, MOV, WMV"
    )

    if uploaded_file is not None:
        # Save uploaded file
        file_ext = os.path.splitext(uploaded_file.name)[1]
        unique_name = f"{uuid.uuid4().hex}{file_ext}"
        save_path = os.path.join(UPLOADS_DIR, unique_name)

        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"✅ Uploaded: **{uploaded_file.name}** ({uploaded_file.size / 1024 / 1024:.1f} MB)")

        # Process button
        if st.button("🔍 Analyze Video", type="primary"):
            from database import insert_video_upload
            import cv2

            # Get total frames
            cap = cv2.VideoCapture(save_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            upload_id = insert_video_upload(unique_name, uploaded_file.name, total_frames)

            st.markdown("---")
            st.subheader("Processing...")

            progress_bar = st.progress(0)
            status_text = st.empty()

            def progress_callback(processed, total):
                if total > 0:
                    pct = min(processed / total, 1.0)
                    progress_bar.progress(pct)
                    status_text.text(f"Processed {processed}/{total} sampled frames...")

            from video_processor import process_video
            results = process_video(upload_id, save_path, progress_callback)

            progress_bar.progress(1.0)
            status_text.text("✅ Processing complete!")

            # Show results
            st.markdown("---")
            st.subheader("Results")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Frames", results["total_frames"])
            with col2:
                st.metric("Frames Analyzed", results["processed_frames"])
            with col3:
                st.metric("Incidents Found", results["incidents_found"])

            if results["detections"]:
                st.markdown("### Detected Incidents")
                for det in results["detections"]:
                    with st.expander(
                        f"Frame {det['frame_number']} — "
                        f"Time: {det['timestamp']:.1f}s — "
                        f"Confidence: {det['confidence']:.1%}"
                    ):
                        if det["snapshot_path"] and os.path.exists(det["snapshot_path"]):
                            img = Image.open(det["snapshot_path"])
                            st.image(img, caption=f"Frame {det['frame_number']}", use_container_width=True)
                        st.markdown(f"**Objects:** {', '.join(det['objects'])}")
                        st.markdown(f"**Confidence:** {det['confidence']:.1%}")
            else:
                st.info("No dumping incidents detected in this video.")

    # Previously uploaded videos
    st.markdown("---")
    st.subheader("Previous Uploads")

    uploads = get_video_uploads()
    if uploads:
        for upload in uploads:
            status_icon = {
                "queued": "⏳",
                "processing": "🔄",
                "completed": "✅",
                "failed": "❌"
            }.get(upload["status"], "❓")

            with st.expander(
                f"{status_icon} {upload['original_name']} — "
                f"Status: {upload['status']} — "
                f"Incidents: {upload['incidents_found']}"
            ):
                st.markdown(f"**Upload ID:** {upload['id']}")
                st.markdown(f"**Status:** {upload['status']}")
                st.markdown(f"**Total Frames:** {upload['total_frames']}")
                st.markdown(f"**Processed Frames:** {upload['processed_frames']}")
                st.markdown(f"**Incidents Found:** {upload['incidents_found']}")
                st.markdown(f"**Uploaded:** {upload['created_at']}")

                # Show detections for completed uploads
                if upload["status"] == "completed":
                    detections = get_video_detections(upload["id"])
                    if detections:
                        st.markdown("**Detections:**")
                        for det in detections:
                            col_img, col_info = st.columns([1, 2])
                            with col_img:
                                if det["snapshot_path"] and os.path.exists(det["snapshot_path"]):
                                    img = Image.open(det["snapshot_path"])
                                    st.image(img, use_container_width=True)
                            with col_info:
                                st.markdown(f"Frame: {det['frame_number']}")
                                st.markdown(f"Time: {det['timestamp_in_video']:.1f}s")
                                st.markdown(f"Confidence: {det['confidence']:.1%}")
    else:
        st.info("No videos uploaded yet.")


# ============================================================
# Camera Management Page
# ============================================================
def page_camera_management():
    st.title("📷 Camera Management")
    st.markdown("Add, test, and manage camera sources.")

    # Add new camera form
    st.subheader("Add New Camera")

    with st.form("add_camera_form"):
        cam_name = st.text_input("Camera Name", placeholder="e.g., Front Gate Camera")

        cam_type = st.selectbox("Camera Type", ["webcam", "ip_stream", "rtsp"])

        if cam_type == "webcam":
            device_idx = st.number_input("Device Index", min_value=0, max_value=10, value=0,
                                         help="0 = default webcam, 1 = second camera, etc.")
            source_url = None
        else:
            device_idx = 0
            placeholder = (
                "http://192.168.1.5:8080/video" if cam_type == "ip_stream"
                else "rtsp://username:password@ip:port/stream"
            )
            source_url = st.text_input("Camera URL", placeholder=placeholder)

        col1, col2 = st.columns(2)
        with col1:
            submitted = st.form_submit_button("➕ Add Camera", type="primary",
                                              use_container_width=True)
        with col2:
            test_btn = st.form_submit_button("🔍 Test Connection", use_container_width=True)

        if submitted:
            if not cam_name:
                st.error("Please enter a camera name.")
            elif cam_type != "webcam" and not source_url:
                st.error("Please enter a camera URL.")
            else:
                camera_id = add_camera(cam_name, cam_type, source_url, device_idx)
                st.success(f"✅ Camera **{cam_name}** added with ID {camera_id}!")
                st.rerun()

        if test_btn:
            test_source = device_idx if cam_type == "webcam" else source_url
            if test_source is not None and (cam_type == "webcam" or test_source):
                with st.spinner("Testing camera connection..."):
                    success, message = test_camera_source(test_source)
                if success:
                    st.success(f"✅ {message}")
                else:
                    st.error(f"❌ {message}")
            else:
                st.warning("Please enter a camera URL to test.")

    # Existing cameras
    st.markdown("---")
    st.subheader("Registered Cameras")

    cameras = list_cameras()
    if cameras:
        for cam in cameras:
            col1, col2, col3 = st.columns([3, 2, 1])

            with col1:
                source_display = cam.get("source_url") or f"Device {cam.get('device_index', 0)}"
                st.markdown(
                    f"**{cam['name']}** — `{cam['source_type']}` — `{source_display}`"
                )

            with col2:
                st.markdown(f"ID: {cam['id']} | Added: {cam['created_at']}")

            with col3:
                if st.button("🗑️ Delete", key=f"del_cam_{cam['id']}"):
                    remove_camera(cam["id"])
                    st.success(f"Deleted camera: {cam['name']}")
                    st.rerun()
    else:
        st.info("No cameras registered yet. Add one above.")

    # Mobile camera instructions
    st.markdown("---")
    st.subheader("📱 Mobile Camera Setup")
    st.markdown("""
    To use your phone as a camera source:

    1. Install **IP Webcam** app on your Android phone (from Play Store)
    2. Open the app and tap **Start Server**
    3. Note the URL shown (e.g., `http://192.168.1.5:8080`)
    4. Add a new camera above with type **ip_stream** and URL: `http://YOUR_IP:8080/video`
    5. Make sure your phone and laptop are on the **same WiFi network**
    """)


# ============================================================
# Incident Viewer Page
# ============================================================
def page_incident_viewer():
    st.title("🔍 Incident Viewer")
    st.markdown("Browse and filter all detected incidents.")

    # Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        cameras = get_cameras()
        camera_filter = st.selectbox(
            "Filter by Camera",
            ["All Cameras"] + [f"{c['name']} (ID: {c['id']})" for c in cameras]
        )

    with col2:
        date_from = st.date_input("From Date", value=datetime.now().date() - timedelta(days=30))

    with col3:
        date_to = st.date_input("To Date", value=datetime.now().date())

    # Parse camera filter
    camera_id_filter = None
    if camera_filter != "All Cameras":
        try:
            camera_id_filter = int(camera_filter.split("ID: ")[1].rstrip(")"))
        except (IndexError, ValueError):
            pass

    # Fetch incidents
    incidents = get_incidents(
        limit=100,
        camera_id=camera_id_filter,
        date_from=str(date_from),
        date_to=str(date_to + timedelta(days=1))
    )

    st.markdown("---")
    st.markdown(f"**Showing {len(incidents)} incidents**")

    if incidents:
        for incident in incidents:
            confidence_str = f"{incident['confidence']:.1%}" if incident['confidence'] else "N/A"

            with st.expander(
                f"🚨 #{incident['id']} | {incident['timestamp']} | "
                f"Confidence: {confidence_str} | Camera: {incident['camera_id']}"
            ):
                col_img, col_info = st.columns([1, 2])

                with col_img:
                    if incident["snapshot_path"] and os.path.exists(incident["snapshot_path"]):
                        img = Image.open(incident["snapshot_path"])
                        st.image(img, caption="Evidence", use_container_width=True)
                    else:
                        st.info("No snapshot available")

                with col_info:
                    st.markdown(f"**Incident ID:** {incident['id']}")
                    st.markdown(f"**Timestamp:** {incident['timestamp']}")
                    st.markdown(f"**Camera ID:** {incident['camera_id']}")
                    st.markdown(f"**Confidence:** {confidence_str}")
                    st.markdown(f"**Description:** {incident['description']}")

                    if incident["objects_detected"]:
                        try:
                            objects = json.loads(incident["objects_detected"])
                            st.markdown(f"**Objects:** {', '.join(objects)}")
                        except (json.JSONDecodeError, TypeError):
                            st.markdown(f"**Objects:** {incident['objects_detected']}")
    else:
        st.info("No incidents found for the selected filters.")


# ============================================================
# Page Router
# ============================================================
if page == "📊 Dashboard":
    page_dashboard()
elif page == "📹 Live Monitor":
    page_live_monitor()
elif page == "📤 Video Upload":
    page_video_upload()
elif page == "📷 Camera Management":
    page_camera_management()
elif page == "🔍 Incident Viewer":
    page_incident_viewer()
