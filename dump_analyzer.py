import time
import math
import numpy as np
from collections import defaultdict
from config import (
    PROXIMITY_THRESHOLD, FRAME_ACCUMULATION_THRESHOLD, COOLDOWN_SECONDS,
    WEIGHT_DETECTION, WEIGHT_POSE, WEIGHT_TEMPORAL, WEIGHT_ACCUMULATION,
    ARM_ANGLE_THRESHOLD, WRIST_DISPLACEMENT_THRESHOLD,
    WASTE_PERSISTENCE_FRAMES, DEFAULT_SAFE_ZONES
)


# YOLOv8 Pose keypoint indices
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_ELBOW = 7
KP_RIGHT_ELBOW = 8
KP_LEFT_WRIST = 9
KP_RIGHT_WRIST = 10
KP_LEFT_HIP = 11
KP_RIGHT_HIP = 12


class DumpEvent:
    """Tracks the accumulation of evidence for a potential dump event."""

    def __init__(self, waste_track_id, person_track_id):
        self.waste_track_id = waste_track_id
        self.person_track_id = person_track_id
        self.accumulated_frames = 0
        self.detection_confidences = []
        self.pose_scores = []
        self.temporal_scores = []
        self.first_detected = time.time()
        self.confirmed = False

    def accumulate(self, detection_conf, pose_score, temporal_score):
        """Add one frame of evidence."""
        self.accumulated_frames += 1
        self.detection_confidences.append(detection_conf)
        self.pose_scores.append(pose_score)
        self.temporal_scores.append(temporal_score)

    def reset(self):
        """Reset accumulation if conditions stop being met."""
        self.accumulated_frames = 0
        self.detection_confidences.clear()
        self.pose_scores.clear()
        self.temporal_scores.clear()

    def is_confirmed(self):
        """Check if enough frames have accumulated to confirm the event."""
        return self.accumulated_frames >= FRAME_ACCUMULATION_THRESHOLD

    def final_confidence(self):
        """Calculate weighted final confidence score."""
        if not self.detection_confidences:
            return 0.0

        avg_detection = sum(self.detection_confidences) / len(self.detection_confidences)
        avg_pose = sum(self.pose_scores) / len(self.pose_scores) if self.pose_scores else 0.0
        avg_temporal = sum(self.temporal_scores) / len(self.temporal_scores) if self.temporal_scores else 0.0
        accumulation_score = min(self.accumulated_frames / FRAME_ACCUMULATION_THRESHOLD, 1.0)

        score = (
            WEIGHT_DETECTION * avg_detection +
            WEIGHT_POSE * avg_pose +
            WEIGHT_TEMPORAL * avg_temporal +
            WEIGHT_ACCUMULATION * accumulation_score
        )
        return round(min(score, 1.0), 3)


class DumpAnalyzer:
    """
    Core intelligence module that combines detection, pose, tracking, and
    temporal data to determine if an illegal dumping event has occurred.
    """

    def __init__(self, safe_zones=None):
        self.active_events = {}  # key: (waste_track_id, person_track_id) -> DumpEvent
        self.cooldown_log = {}   # key: region_key -> last_incident_time
        self.safe_zones = safe_zones or DEFAULT_SAFE_ZONES
        self.previous_wrist_positions = {}  # person_track_id -> {left: (x,y), right: (x,y)}

    def analyze(self, tracked_data, poses):
        """
        Analyze current frame data to detect dump events.

        Args:
            tracked_data: dict from ObjectTracker.update() with persons, new_waste, etc.
            poses: list of pose dicts from Detector.detect_pose()

        Returns:
            list of confirmed DumpEvent objects (may be empty)
        """
        confirmed_events = []
        new_waste = tracked_data["new_waste"]
        persons = tracked_data["persons"]

        # Build a map of person track_id -> pose data
        pose_map = {}
        for pose in poses:
            tid = pose["track_id"]
            if tid != -1:
                pose_map[tid] = pose

        # Check each new waste object
        for waste_obj in new_waste:
            # Skip if in a safe zone
            if self._is_in_safe_zone(waste_obj.box):
                continue

            # Find nearby person
            closest_person = None
            closest_dist = float("inf")
            for pid, person in persons.items():
                dist = self._distance(waste_obj.center(), person.center())
                if dist < closest_dist:
                    closest_dist = dist
                    closest_person = person

            if closest_person is None or closest_dist > PROXIMITY_THRESHOLD:
                # No person nearby — reset any active event for this waste
                self._reset_events_for_waste(waste_obj.track_id)
                continue

            # Calculate scores
            detection_conf = waste_obj.confidence
            pose_score = self._analyze_pose(closest_person.track_id, pose_map)
            temporal_score = self._temporal_score(waste_obj)

            # Get or create event
            event_key = (waste_obj.track_id, closest_person.track_id)
            if event_key not in self.active_events:
                self.active_events[event_key] = DumpEvent(
                    waste_obj.track_id, closest_person.track_id
                )

            event = self.active_events[event_key]
            event.accumulate(detection_conf, pose_score, temporal_score)

            # Check if confirmed
            if event.is_confirmed() and not event.confirmed:
                # Check cooldown
                region_key = self._region_key(waste_obj.box)
                if not self._is_in_cooldown(region_key):
                    event.confirmed = True
                    self.cooldown_log[region_key] = time.time()
                    confirmed_events.append(event)

        # Clean up old events that haven't accumulated
        self._cleanup_stale_events(tracked_data["frame_count"])

        return confirmed_events

    def _analyze_pose(self, person_track_id, pose_map):
        """
        Analyze pose keypoints to detect throwing/dropping actions.
        Returns a score between 0.0 and 1.0.
        """
        if person_track_id not in pose_map:
            return 0.0

        pose = pose_map[person_track_id]
        keypoints = pose["keypoints"]  # shape (17, 3)
        score = 0.0

        # Check arm angles — look for downward arm extension
        arm_score = self._check_arm_angle(keypoints)
        score += arm_score * 0.5

        # Check wrist displacement between frames
        wrist_score = self._check_wrist_displacement(person_track_id, keypoints)
        score += wrist_score * 0.5

        # Update previous wrist positions
        left_wrist = keypoints[KP_LEFT_WRIST]
        right_wrist = keypoints[KP_RIGHT_WRIST]
        self.previous_wrist_positions[person_track_id] = {
            "left": (left_wrist[0], left_wrist[1]),
            "right": (right_wrist[0], right_wrist[1])
        }

        return min(score, 1.0)

    def _check_arm_angle(self, keypoints):
        """
        Check if either arm is in a throwing/dropping position.
        Measures angle between shoulder-elbow-wrist.
        Returns score 0.0 to 1.0.
        """
        best_score = 0.0

        for side in ["left", "right"]:
            if side == "left":
                shoulder = keypoints[KP_LEFT_SHOULDER]
                elbow = keypoints[KP_LEFT_ELBOW]
                wrist = keypoints[KP_LEFT_WRIST]
            else:
                shoulder = keypoints[KP_RIGHT_SHOULDER]
                elbow = keypoints[KP_RIGHT_ELBOW]
                wrist = keypoints[KP_RIGHT_WRIST]

            # Skip if keypoints have low confidence
            if shoulder[2] < 0.3 or elbow[2] < 0.3 or wrist[2] < 0.3:
                continue

            # Calculate angle at elbow
            angle = self._angle_between_points(
                (shoulder[0], shoulder[1]),
                (elbow[0], elbow[1]),
                (wrist[0], wrist[1])
            )

            # Check if wrist is below elbow (dropping motion)
            wrist_below_elbow = wrist[1] > elbow[1]

            if angle < ARM_ANGLE_THRESHOLD and wrist_below_elbow:
                # Strong throwing/dropping signal
                best_score = max(best_score, 1.0)
            elif angle < ARM_ANGLE_THRESHOLD * 1.5:
                best_score = max(best_score, 0.5)

        return best_score

    def _check_wrist_displacement(self, person_track_id, keypoints):
        """
        Check if wrists have moved significantly downward between frames.
        Returns score 0.0 to 1.0.
        """
        if person_track_id not in self.previous_wrist_positions:
            return 0.0

        prev = self.previous_wrist_positions[person_track_id]
        best_score = 0.0

        for side, kp_idx in [("left", KP_LEFT_WRIST), ("right", KP_RIGHT_WRIST)]:
            curr_wrist = keypoints[kp_idx]
            if curr_wrist[2] < 0.3:
                continue

            prev_pos = prev[side]
            dy = curr_wrist[1] - prev_pos[1]  # Positive = downward

            if dy > WRIST_DISPLACEMENT_THRESHOLD:
                # Significant downward movement
                normalized = min(dy / (WRIST_DISPLACEMENT_THRESHOLD * 3), 1.0)
                best_score = max(best_score, normalized)

        return best_score

    def _temporal_score(self, waste_obj):
        """
        Score based on how "new" the waste object is and how long it has persisted.
        Returns 0.0 to 1.0.
        """
        if not waste_obj.is_new:
            return 0.0

        # Higher score if waste has persisted for more frames
        persistence_ratio = min(waste_obj.frames_seen / WASTE_PERSISTENCE_FRAMES, 1.0)
        return persistence_ratio

    def _is_in_safe_zone(self, box):
        """Check if a bounding box center falls within any safe zone."""
        cx = (box[0] + box[2]) // 2
        cy = (box[1] + box[3]) // 2

        for zone in self.safe_zones:
            if (zone["x1"] <= cx <= zone["x2"] and
                    zone["y1"] <= cy <= zone["y2"]):
                return True
        return False

    def _is_in_cooldown(self, region_key):
        """Check if a region is still in cooldown after a recent incident."""
        if region_key not in self.cooldown_log:
            return False
        elapsed = time.time() - self.cooldown_log[region_key]
        return elapsed < COOLDOWN_SECONDS

    def _region_key(self, box, grid_size=150):
        """Map a bounding box to a region key for cooldown tracking."""
        cx = (box[0] + box[2]) // 2
        cy = (box[1] + box[3]) // 2
        return (cx // grid_size, cy // grid_size)

    def _reset_events_for_waste(self, waste_track_id):
        """Reset any active events involving this waste object."""
        keys_to_reset = [
            k for k in self.active_events if k[0] == waste_track_id
        ]
        for k in keys_to_reset:
            self.active_events[k].reset()

    def _cleanup_stale_events(self, current_frame, max_age_seconds=60):
        """Remove events that are too old and never confirmed."""
        now = time.time()
        keys_to_remove = []
        for key, event in self.active_events.items():
            if event.confirmed:
                keys_to_remove.append(key)
            elif now - event.first_detected > max_age_seconds:
                keys_to_remove.append(key)
        for key in keys_to_remove:
            del self.active_events[key]

    @staticmethod
    def _distance(p1, p2):
        """Euclidean distance between two points."""
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    @staticmethod
    def _angle_between_points(a, b, c):
        """
        Calculate angle at point b formed by points a-b-c.
        Returns angle in degrees.
        """
        ba = (a[0] - b[0], a[1] - b[1])
        bc = (c[0] - b[0], c[1] - b[1])

        dot = ba[0] * bc[0] + ba[1] * bc[1]
        mag_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2)
        mag_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2)

        if mag_ba == 0 or mag_bc == 0:
            return 180.0

        cos_angle = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))
        return math.degrees(math.acos(cos_angle))

    def reset(self):
        """Reset all analyzer state."""
        self.active_events.clear()
        self.cooldown_log.clear()
        self.previous_wrist_positions.clear()
