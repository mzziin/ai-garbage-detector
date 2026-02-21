import math
import numpy as np
from ultralytics import YOLO
from config import (
    DEVICE, DETECTION_MODEL, POSE_MODEL,
    DETECTION_CONFIDENCE, POSE_CONFIDENCE,
    PERSON_CLASS_ID, WASTE_CLASS_IDS
)


class Detector:
    """Wraps YOLOv8 object detection and pose estimation models."""

    def __init__(self):
        print(f"[Detector] Loading models on device: {DEVICE}")
        self.detection_model = YOLO(DETECTION_MODEL)
        self.pose_model = YOLO(POSE_MODEL)
        print(f"[Detector] Detection model: {DETECTION_MODEL}")
        print(f"[Detector] Pose model: {POSE_MODEL}")

    @staticmethod
    def _safe_track_id(boxes, i):
        """Safely extract track ID, guarding against None tensor and NaN values."""
        if boxes.id is None:
            return -1
        val = boxes.id[i].item()
        if math.isnan(val):
            return -1
        return int(val)

    def detect(self, frame):
        """
        Run object detection on a frame.

        Returns:
            dict with keys:
                - "persons": list of dicts {box, confidence, track_id}
                - "waste": list of dicts {box, confidence, class_id, class_name, track_id}
                - "raw_results": ultralytics Results object
        """
        results = self.detection_model.track(
            frame,
            conf=DETECTION_CONFIDENCE,
            device=DEVICE,
            persist=True,
            verbose=False
        )

        persons = []
        waste_objects = []

        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes

            if boxes is not None and len(boxes) > 0:
                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i].item())
                    conf = float(boxes.conf[i].item())
                    box = boxes.xyxy[i].cpu().numpy().astype(int).tolist()
                    track_id = self._safe_track_id(boxes, i)

                    entry = {
                        "box": box,  # [x1, y1, x2, y2]
                        "confidence": conf,
                        "track_id": track_id,
                        "class_id": cls_id,
                        "class_name": self.detection_model.names[cls_id]
                    }

                    if cls_id == PERSON_CLASS_ID:
                        persons.append(entry)
                    elif cls_id in WASTE_CLASS_IDS:
                        waste_objects.append(entry)

        return {
            "persons": persons,
            "waste": waste_objects,
            "raw_results": results
        }

    def detect_pose(self, frame):
        """
        Run pose estimation on a frame.

        Returns:
            list of dicts, each with:
                - "keypoints": numpy array of shape (17, 3) — x, y, confidence per keypoint
                - "box": [x1, y1, x2, y2]
                - "confidence": float
                - "track_id": int
        
        YOLOv8 Pose Keypoint indices:
            0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
            5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
            9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
            13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
        """
        results = self.pose_model.track(
            frame,
            conf=POSE_CONFIDENCE,
            device=DEVICE,
            persist=True,
            verbose=False
        )

        poses = []

        if results and len(results) > 0:
            result = results[0]
            
            if result.keypoints is not None and result.boxes is not None:
                keypoints_data = result.keypoints.data.cpu().numpy()
                boxes = result.boxes

                for i in range(len(keypoints_data)):
                    kps = keypoints_data[i]  # shape (17, 3)
                    box = boxes.xyxy[i].cpu().numpy().astype(int).tolist()
                    conf = float(boxes.conf[i].item())
                    track_id = self._safe_track_id(boxes, i)

                    poses.append({
                        "keypoints": kps,
                        "box": box,
                        "confidence": conf,
                        "track_id": track_id
                    })

        return poses

    def detect_for_video(self, frame):
        """
        Run detection without tracking (for uploaded video analysis).
        Tracking state should not persist across non-sequential frames.

        Returns same format as detect() but without track_ids.
        """
        results = self.detection_model(
            frame,
            conf=DETECTION_CONFIDENCE,
            device=DEVICE,
            verbose=False
        )

        persons = []
        waste_objects = []

        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes

            if boxes is not None and len(boxes) > 0:
                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i].item())
                    conf = float(boxes.conf[i].item())
                    box = boxes.xyxy[i].cpu().numpy().astype(int).tolist()

                    entry = {
                        "box": box,
                        "confidence": conf,
                        "track_id": -1,
                        "class_id": cls_id,
                        "class_name": self.detection_model.names[cls_id]
                    }

                    if cls_id == PERSON_CLASS_ID:
                        persons.append(entry)
                    elif cls_id in WASTE_CLASS_IDS:
                        waste_objects.append(entry)

        return {
            "persons": persons,
            "waste": waste_objects,
            "raw_results": results
        }

    def detect_pose_for_video(self, frame):
        """Run pose estimation without tracking (for uploaded video analysis)."""
        results = self.pose_model(
            frame,
            conf=POSE_CONFIDENCE,
            device=DEVICE,
            verbose=False
        )

        poses = []

        if results and len(results) > 0:
            result = results[0]

            if result.keypoints is not None and result.boxes is not None:
                keypoints_data = result.keypoints.data.cpu().numpy()
                boxes = result.boxes

                for i in range(len(keypoints_data)):
                    kps = keypoints_data[i]
                    box = boxes.xyxy[i].cpu().numpy().astype(int).tolist()
                    conf = float(boxes.conf[i].item())

                    poses.append({
                        "keypoints": kps,
                        "box": box,
                        "confidence": conf,
                        "track_id": -1
                    })

        return poses
