"""
Live Monitor — Standalone OpenCV window for real-time garbage dump detection.

Usage:
    python live_monitor.py --source 0 --camera-id 1
    python live_monitor.py --source "http://192.168.1.5:8080/video" --camera-id 2
    python live_monitor.py --source "rtsp://..." --camera-id 3

Press Q to quit.
"""

import argparse
import sys
import os
import time
import cv2
import numpy as np
from datetime import datetime

from config import DEVICE, EVIDENCE_DIR
from detector import Detector
from tracker import ObjectTracker
from dump_analyzer import DumpAnalyzer
from database import insert_incident


def parse_args():
    parser = argparse.ArgumentParser(description="Live Garbage Dump Detection Monitor")
    parser.add_argument("--source", type=str, default="0",
                        help="Camera source: device index (0,1,...) or URL (http/rtsp)")
    parser.add_argument("--camera-id", type=int, default=None,
                        help="Camera ID from database (for incident logging)")
    parser.add_argument("--camera-name", type=str, default="Unknown Camera",
                        help="Camera name for display")
    return parser.parse_args()


def draw_detections(frame, detections, tracked_data, poses, events, fps):
    """Draw all detection overlays on the frame."""
    h, w = frame.shape[:2]

    # Draw person bounding boxes (green)
    for det in detections["persons"]:
        x1, y1, x2, y2 = det["box"]
        tid = det["track_id"]
        conf = det["confidence"]
        label = f"Person #{tid} ({conf:.2f})"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)

    # Draw waste bounding boxes
    for det in detections["waste"]:
        x1, y1, x2, y2 = det["box"]
        tid = det["track_id"]
        conf = det["confidence"]
        name = det["class_name"]

        # Check if this is new waste (red) or pre-existing (yellow)
        is_new = any(w.track_id == tid for w in tracked_data.get("new_waste", []))
        color = (0, 0, 255) if is_new else (0, 255, 255)
        label_prefix = "NEW " if is_new else ""

        label = f"{label_prefix}{name} #{tid} ({conf:.2f})"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2)

    # Draw pose skeletons
    skeleton_connections = [
        (5, 7), (7, 9),    # left arm
        (6, 8), (8, 10),   # right arm
        (5, 6),            # shoulders
        (5, 11), (6, 12),  # torso
        (11, 12),          # hips
        (11, 13), (13, 15), # left leg
        (12, 14), (14, 16)  # right leg
    ]

    for pose in poses:
        kps = pose["keypoints"]
        for i, j in skeleton_connections:
            if kps[i][2] > 0.3 and kps[j][2] > 0.3:
                pt1 = (int(kps[i][0]), int(kps[i][1]))
                pt2 = (int(kps[j][0]), int(kps[j][1]))
                cv2.line(frame, pt1, pt2, (255, 165, 0), 2)

        # Draw keypoints
        for kp in kps:
            if kp[2] > 0.3:
                cv2.circle(frame, (int(kp[0]), int(kp[1])), 3, (0, 255, 255), -1)

    # Flash red border if dump event detected
    if events:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)
        cv2.putText(frame, "!! ILLEGAL DUMPING DETECTED !!", (w // 2 - 250, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    # OSD overlay — top-left info panel
    overlay_lines = [
        f"FPS: {fps:.1f}",
        f"Device: {DEVICE.upper()}",
        f"Persons: {len(detections['persons'])}",
        f"Waste Objects: {len(detections['waste'])}",
        f"New Waste: {len(tracked_data.get('new_waste', []))}",
        f"Tracks: {len(tracked_data.get('persons', {})) + len(tracked_data.get('waste', {}))}",
    ]

    # Semi-transparent background for OSD
    overlay_h = 25 * len(overlay_lines) + 15
    overlay_bg = frame[5:5 + overlay_h, 5:280].copy()
    cv2.rectangle(frame, (5, 5), (280, 5 + overlay_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay_bg, 0.3, frame[5:5 + overlay_h, 5:280], 0.7, 0,
                    frame[5:5 + overlay_h, 5:280])

    for i, line in enumerate(overlay_lines):
        cv2.putText(frame, line, (10, 25 + i * 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1)

    return frame


def save_snapshot(frame, camera_id):
    """Save an incident snapshot and return the file path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"incident_{camera_id}_{timestamp}.jpg"
    filepath = os.path.join(EVIDENCE_DIR, filename)
    cv2.imwrite(filepath, frame)
    return filepath


def main():
    args = parse_args()

    # Parse source — integer for webcam, string for URL
    source = args.source
    try:
        source = int(source)
    except ValueError:
        pass  # It's a URL string

    camera_id = args.camera_id
    camera_name = args.camera_name

    print(f"[LiveMonitor] Starting with source: {source}")
    print(f"[LiveMonitor] Camera ID: {camera_id}, Name: {camera_name}")
    print(f"[LiveMonitor] Device: {DEVICE}")

    # Initialize components
    print("[LiveMonitor] Loading AI models...")
    detector = Detector()
    tracker = ObjectTracker()
    analyzer = DumpAnalyzer()

    # Open camera
    print(f"[LiveMonitor] Opening camera source: {source}")
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[LiveMonitor] ERROR: Could not open camera source: {source}")
        sys.exit(1)

    print("[LiveMonitor] Camera opened successfully. Press Q to quit.")

    window_name = f"Live Monitor - {camera_name}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    fps = 0.0
    frame_time = time.time()
    incident_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[LiveMonitor] Failed to read frame. Retrying...")
                time.sleep(0.1)
                continue

            # Calculate FPS
            now = time.time()
            fps = 1.0 / max(now - frame_time, 0.001)
            frame_time = now

            # Stage 1: Object detection with tracking
            detections = detector.detect(frame)

            # Stage 2: Pose estimation
            poses = detector.detect_pose(frame)

            # Stage 3 & 4: Update tracker (tracking + temporal analysis)
            tracked_data = tracker.update(detections)

            # Stage 5, 6, 7: Analyze for dump events
            confirmed_events = analyzer.analyze(tracked_data, poses)

            # Log confirmed incidents
            for event in confirmed_events:
                incident_count += 1
                snapshot_path = save_snapshot(frame, camera_id)
                confidence = event.final_confidence()

                # Build description
                objects = []
                waste_tid = event.waste_track_id
                for det in detections["waste"]:
                    if det["track_id"] == waste_tid:
                        objects.append(det["class_name"])
                        break

                description = (
                    f"Illegal dumping detected. "
                    f"Person #{event.person_track_id} dropped waste near "
                    f"tracked object #{waste_tid}. "
                    f"Confidence: {confidence:.1%}"
                )

                if camera_id is not None:
                    insert_incident(
                        camera_id=camera_id,
                        confidence=confidence,
                        snapshot_path=snapshot_path,
                        description=description,
                        objects_detected=objects
                    )
                    print(f"[LiveMonitor] INCIDENT #{incident_count} logged! "
                          f"Confidence: {confidence:.1%}, Snapshot: {snapshot_path}")
                else:
                    print(f"[LiveMonitor] INCIDENT detected (no camera_id to log). "
                          f"Confidence: {confidence:.1%}")

            # Draw everything on frame
            display_frame = draw_detections(
                frame.copy(), detections, tracked_data, poses, confirmed_events, fps
            )

            # Show frame
            cv2.imshow(window_name, display_frame)

            # Check for quit key
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("[LiveMonitor] Quit requested.")
                break

    except KeyboardInterrupt:
        print("[LiveMonitor] Interrupted.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"[LiveMonitor] Stopped. Total incidents detected: {incident_count}")


if __name__ == "__main__":
    main()
