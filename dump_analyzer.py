import time
import math
import numpy as np
from collections import defaultdict
from config import (
    PROXIMITY_THRESHOLD, FRAME_ACCUMULATION_THRESHOLD, COOLDOWN_SECONDS,
    WEIGHT_DETECTION, WEIGHT_TEMPORAL, WEIGHT_ACCUMULATION,
    WASTE_PERSISTENCE_FRAMES, DEFAULT_SAFE_ZONES
)


class DumpEvent:
    """Tracks the accumulation of evidence for a potential dump event."""

    def __init__(self, waste_track_id, person_track_id):
        self.waste_track_id = waste_track_id
        self.person_track_id = person_track_id
        self.accumulated_frames = 0
        self.detection_confidences = []
        self.temporal_scores = []
        self.first_detected = time.time()
        self.confirmed = False

    def accumulate(self, detection_conf, temporal_score):
        """Add one frame of evidence."""
        self.accumulated_frames += 1
        self.detection_confidences.append(detection_conf)
        self.temporal_scores.append(temporal_score)

    def reset(self):
        """Reset accumulation if conditions stop being met."""
        self.accumulated_frames = 0
        self.detection_confidences.clear()
        self.temporal_scores.clear()

    def is_confirmed(self):
        """Check if enough frames have accumulated to confirm the event."""
        return self.accumulated_frames >= FRAME_ACCUMULATION_THRESHOLD

    def final_confidence(self):
        """Calculate weighted final confidence score."""
        if not self.detection_confidences:
            return 0.0

        avg_detection = sum(self.detection_confidences) / len(self.detection_confidences)
        avg_temporal = sum(self.temporal_scores) / len(self.temporal_scores) if self.temporal_scores else 0.0
        accumulation_score = min(self.accumulated_frames / FRAME_ACCUMULATION_THRESHOLD, 1.0)

        score = (
            WEIGHT_DETECTION * avg_detection +
            WEIGHT_TEMPORAL * avg_temporal +
            WEIGHT_ACCUMULATION * accumulation_score
        )
        return round(min(score, 1.0), 3)


class DumpAnalyzer:
    """
    Core intelligence module that combines detection, tracking, and
    temporal data to determine if an illegal dumping event has occurred.

    Logic: person near new waste → person leaves → waste stays = DUMPING
    """

    def __init__(self, safe_zones=None):
        self.active_events = {}  # key: (waste_track_id, person_track_id) -> DumpEvent
        self.cooldown_log = {}   # key: region_key -> last_incident_time
        self.safe_zones = safe_zones or DEFAULT_SAFE_ZONES

    def analyze(self, tracked_data):
        """
        Analyze current frame data to detect dump events.

        Args:
            tracked_data: dict from ObjectTracker.update() with persons, new_waste, etc.

        Returns:
            list of confirmed DumpEvent objects (may be empty)
        """
        confirmed_events = []
        new_waste = tracked_data["new_waste"]
        persons = tracked_data["persons"]

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
            temporal_score = self._temporal_score(waste_obj)

            # Get or create event
            event_key = (waste_obj.track_id, closest_person.track_id)
            if event_key not in self.active_events:
                self.active_events[event_key] = DumpEvent(
                    waste_obj.track_id, closest_person.track_id
                )

            event = self.active_events[event_key]
            event.accumulate(detection_conf, temporal_score)

            # Check if confirmed
            if event.is_confirmed() and not event.confirmed:
                # Check cooldown
                region_key = self._region_key(waste_obj.box)
                if not self._is_in_cooldown(region_key):
                    event.confirmed = True
                    self.cooldown_log[region_key] = time.time()
                    confirmed_events.append(event)

        # Clean up old events that haven't accumulated
        self._cleanup_stale_events()

        return confirmed_events

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

    def _cleanup_stale_events(self, max_age_seconds=60):
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

    def reset(self):
        """Reset all analyzer state."""
        self.active_events.clear()
        self.cooldown_log.clear()
