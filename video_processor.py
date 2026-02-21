"""
Video Processor — Analyzes uploaded video files for garbage dumping events.

Processes frames at a configurable sample rate, runs the detection pipeline,
and logs any detected incidents to the database.
"""

import os
import cv2
import json
from datetime import datetime

from config import EVIDENCE_DIR, VIDEO_FRAME_SAMPLE_RATE, DEVICE
from detector import Detector
from tracker import ObjectTracker
from dump_analyzer import DumpAnalyzer
from database import (
    update_video_upload, insert_video_detection
)


def process_video(upload_id, video_path, progress_callback=None):
    """
    Process an uploaded video file for garbage dump detection.

    Args:
        upload_id: Database ID of the video_uploads record
        video_path: Full path to the video file
        progress_callback: Optional callable(processed_frames, total_frames)
                          for progress updates (used by Streamlit)

    Returns:
        dict with:
            - "total_frames": int
            - "processed_frames": int
            - "incidents_found": int
            - "detections": list of detection dicts
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        update_video_upload(upload_id, status="failed")
        return {"total_frames": 0, "processed_frames": 0,
                "incidents_found": 0, "detections": []}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    update_video_upload(upload_id, status="processing", processed_frames=0)

    # Initialize detection components
    detector = Detector()
    tracker = ObjectTracker()
    analyzer = DumpAnalyzer()

    processed_count = 0
    incidents_found = 0
    all_detections = []
    frame_number = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_number += 1

            # Sample every Nth frame
            if frame_number % VIDEO_FRAME_SAMPLE_RATE != 0:
                continue

            processed_count += 1
            timestamp_in_video = frame_number / fps

            # Run detection (without persistent tracking for sampled frames)
            detections = detector.detect_for_video(frame)
            poses = detector.detect_pose_for_video(frame)

            # For video analysis, we use a simplified approach:
            # Check if person + waste are in proximity in the same frame
            persons = detections["persons"]
            waste = detections["waste"]

            detection_made = False

            if persons and waste:
                # Check proximity between each person-waste pair
                for waste_obj in waste:
                    wx = (waste_obj["box"][0] + waste_obj["box"][2]) // 2
                    wy = (waste_obj["box"][1] + waste_obj["box"][3]) // 2

                    for person in persons:
                        px = (person["box"][0] + person["box"][2]) // 2
                        py = (person["box"][1] + person["box"][3]) // 2

                        dist = ((wx - px) ** 2 + (wy - py) ** 2) ** 0.5

                        from config import PROXIMITY_THRESHOLD
                        if dist <= PROXIMITY_THRESHOLD:
                            # Check pose for throwing action
                            pose_score = 0.0
                            for pose in poses:
                                # Simple check: is any wrist below the elbow?
                                kps = pose["keypoints"]
                                for wrist_idx, elbow_idx in [(9, 7), (10, 8)]:
                                    if (kps[wrist_idx][2] > 0.3 and
                                            kps[elbow_idx][2] > 0.3 and
                                            kps[wrist_idx][1] > kps[elbow_idx][1]):
                                        pose_score = max(pose_score, 0.6)

                            # Calculate confidence
                            detection_conf = (waste_obj["confidence"] + person["confidence"]) / 2
                            confidence = detection_conf * 0.5 + pose_score * 0.3 + 0.2

                            if confidence > 0.3:
                                # Save snapshot
                                snapshot_path = _save_video_snapshot(
                                    frame, upload_id, frame_number
                                )

                                objects_detected = [waste_obj["class_name"]]

                                insert_video_detection(
                                    upload_id=upload_id,
                                    frame_number=frame_number,
                                    timestamp_in_video=round(timestamp_in_video, 2),
                                    confidence=round(confidence, 3),
                                    snapshot_path=snapshot_path,
                                    objects_detected=objects_detected
                                )

                                incidents_found += 1
                                all_detections.append({
                                    "frame_number": frame_number,
                                    "timestamp": round(timestamp_in_video, 2),
                                    "confidence": round(confidence, 3),
                                    "objects": objects_detected,
                                    "snapshot_path": snapshot_path
                                })
                                detection_made = True
                                break  # One detection per waste object

                    if detection_made:
                        break  # Move to next frame after first detection

            # Update progress
            update_video_upload(
                upload_id,
                processed_frames=processed_count,
                incidents_found=incidents_found
            )

            if progress_callback:
                progress_callback(processed_count, total_frames // VIDEO_FRAME_SAMPLE_RATE)

    except Exception as e:
        print(f"[VideoProcessor] Error processing video: {e}")
        update_video_upload(upload_id, status="failed")
        raise
    finally:
        cap.release()

    # Mark as completed
    update_video_upload(upload_id, status="completed",
                        processed_frames=processed_count,
                        incidents_found=incidents_found)

    return {
        "total_frames": total_frames,
        "processed_frames": processed_count,
        "incidents_found": incidents_found,
        "detections": all_detections
    }


def _save_video_snapshot(frame, upload_id, frame_number):
    """Save a detection snapshot from video processing."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"video_{upload_id}_frame_{frame_number}_{timestamp}.jpg"
    filepath = os.path.join(EVIDENCE_DIR, filename)
    cv2.imwrite(filepath, frame)
    return filepath
