import numpy as np
import cv2
import string

from Module.imageprocessing.ImageProcessor import (
    find_components,
    mask_to_well,
    extract_curve,
    filter_worms,
)
from Module.imageprocessing.foreground import extract_foreground


# ------------------------------------------------------------
#  Worm Detection Object
# ------------------------------------------------------------
class WormDetection:
    """Single detection inside a well."""

    def __init__(self, centroid, end1, end2, area, length, region='core'):
        self.centroid = np.array(centroid, dtype=float)
        self.end1 = np.array(end1, dtype=float) if end1 is not None else None
        self.end2 = np.array(end2, dtype=float) if end2 is not None else None
        self.area = area
        self.length = length
        self.region = region


# ------------------------------------------------------------
#  Worm Track Object
# ------------------------------------------------------------
class WormTrack:
    """Tracks a single worm over time inside a well."""

    def __init__(
        self,
        track_id,
        centroid,
        area,
        length,
        min_skel_length,
        min_area,
        min_motion,
        well_id,
        role="secondary",
    ):
        # Identity
        self.id = track_id
        self.well_id = well_id
        self.role = role  # "primary" or "secondary"

        # Geometry
        self.centroid = np.array(centroid, dtype=float)
        self.prev_centroid = None
        self.velocity = np.zeros(2)

        self.area = area
        self.length = length

        # Head / tail
        self.head = None
        self.tail = None
        self.head_tail_confident = False

        # Region
        self.region = None
        self.region_history = []

        # Lifetime
        self.missed = 0
        self.age = 0

        # NEW: visibility bookkeeping
        self.last_seen_frame = None
        self.offscreen = False

        # Thresholds
        self.min_skel_length = min_skel_length
        self.min_area = min_area
        self.min_motion = min_motion

    def update_motion(self, new_centroid):
        if self.prev_centroid is not None:
            self.velocity = new_centroid - self.prev_centroid
        else:
            self.velocity[:] = 0

        self.prev_centroid = self.centroid.copy()
        self.centroid = new_centroid

    def compute_head_tail(self, detection):
        if detection.end1 is None or detection.end2 is None:
            self.head = self.tail = None
            self.head_tail_confident = False
            return

        if detection.length < self.min_skel_length or detection.area < self.min_area:
            self.head = self.tail = None
            self.head_tail_confident = False
            return

        pts = np.array([detection.end1, detection.end2])
        motion = (
            self.centroid - self.prev_centroid
            if self.prev_centroid is not None
            else np.zeros(2)
        )
        speed = np.linalg.norm(motion)

        if speed >= self.min_motion:
            dot = [np.dot(pts[i] - self.centroid, motion) for i in range(2)]
            head_idx = np.argmax(dot)
            self.head, self.tail = pts[head_idx], pts[1 - head_idx]
            self.head_tail_confident = True
        elif self.head is not None and self.tail is not None:
            d0 = np.linalg.norm(pts[0] - self.head) + np.linalg.norm(pts[1] - self.tail)
            d1 = np.linalg.norm(pts[1] - self.head) + np.linalg.norm(pts[0] - self.tail)
            self.head, self.tail = (pts[0], pts[1]) if d0 <= d1 else (pts[1], pts[0])
            self.head_tail_confident = True
        else:
            self.head = self.tail = None
            self.head_tail_confident = False

    def assign(self, detection, frame_idx):
        self.update_motion(detection.centroid)
        self.area = detection.area
        self.length = detection.length
        self.missed = 0
        self.age += 1
        self.offscreen = False
        self.last_seen_frame = frame_idx

        if hasattr(detection, "region"):
            self.region = detection.region
            self.region_history.append(detection.region)

        self.compute_head_tail(detection)

    def mark_missed(self):
        self.missed += 1
        self.age += 1
        self.velocity[:] = 0
        self.head = self.tail = None
        self.head_tail_confident = False

    def export_row(self, video_idx, frame_idx):
        cx, cy = (-1, -1) if self.centroid is None else map(int, self.centroid)
        hx, hy = (-1, -1) if self.head is None else map(int, self.head)
        tx, ty = (-1, -1) if self.tail is None else map(int, self.tail)

        return (
            video_idx,
            frame_idx,
            int(self.id[1:]),
            self.well_id,
            cx, cy,
            hx, hy,
            tx, ty,
            self.region if self.region is not None else -1,
            self.age,
            1 if self.role == "primary" else 0,
        )


# ------------------------------------------------------------
#  Worm Tracker (ONE PER WELL)
# ------------------------------------------------------------
class WormTracker:
    """Tracks multiple worms inside a single well."""

    def __init__(
        self,
        well_id,
        max_dist=50,
        max_missed=5,
        min_new_area=5,
        min_new_dist=10,
        min_skel_length=150,
        min_motion=2.0,
        min_area=150,
        area_weight=50,
    ):
        self.well_id = well_id
        self.label = string.ascii_uppercase[well_id]

        # Active and dormant tracks
        self.tracks = []
        self.lost_tracks = []

        self.next_id = 1
        self.frame_idx = 0

        # Parameters
        self.max_dist = max_dist
        self.max_missed = max_missed
        self.min_new_area = min_new_area
        self.min_new_dist = min_new_dist
        self.area_weight = area_weight
        self.min_skel_length = min_skel_length
        self.min_motion = min_motion
        self.min_area = min_area

        # Birth‑suppression parameters
        self.birth_suppression_dist = min_new_dist
        self.birth_suppression_frames = max_missed
        self.min_secondary_persistence = self.max_missed

        self.primary_track = None
        self.primary_initialized = False
        self.primary_base_dist = max_dist      # px
        self.primary_max_speed = 0.5

    # ---------------- ID / cost helpers ----------------

    def _make_id(self):
        tid = f"{self.label}{self.next_id}"
        self.next_id += 1
        return tid

    def _cost(self, track, det):
        if track.prev_centroid is None:
            return 0.0

        dist = np.linalg.norm(track.prev_centroid - det.centroid)
        pred = track.prev_centroid + track.velocity
        vel_error = np.linalg.norm(pred - det.centroid)
        area_ratio = abs(track.area - det.area) / max(track.area, 1)

        return dist + vel_error + self.area_weight * area_ratio

    def _can_create_new(self, det):
        # Very small blobs never create tracks
        if det.area < self.min_new_area:
            return False

        # Must be sufficiently far from ALL existing tracks
        for t in self.tracks + self.lost_tracks:
            if t.centroid is not None:
                if np.linalg.norm(t.centroid - det.centroid) < self.min_new_dist:
                    return False

        # Must not be too similar in size to primary (prevents ID stealing)
        if self.primary_track is not None:
            area_ratio = det.area / max(self.primary_track.area, 1)
            if area_ratio > 0.6:
                return False

        return True

    def _suppressed_birth(self, det):
        """
        Prevent spawning new worms near recently lost tracks.
        """
        for t in self.lost_tracks:
            if t.centroid is None:
                continue
            if np.linalg.norm(t.centroid - det.centroid) < self.birth_suppression_dist:
                return True
        return False

    def _initialize_primary(self, detections):
        """
        Choose the largest detection as the main worm.
        Only called once.
        """
        if self.primary_initialized:
            return

        if len(detections) == 0:
            return

        # Largest detection by area
        primary_det = max(detections, key=lambda d: d.area)

        t = WormTrack(
            track_id=f"{self.label}1",      # FIXED first ID
            centroid=primary_det.centroid,
            area=primary_det.area,
            length=primary_det.length,
            min_skel_length=self.min_skel_length,
            min_area=self.min_area,
            min_motion=self.min_motion,
            well_id=self.well_id,
            role="primary",
        )
        t.assign(primary_det, self.frame_idx)

        self.primary_track = t
        self.tracks.append(t)
        self.primary_initialized = True

    # ---------------- Main update ----------------

    def update(self, detections):
        """
        Update tracker state using current detections.
        Tracking, not background, is the authority.
        """

        self._initialize_primary(detections)
        matched = set()
        active_tracks = []

        if self.primary_track is not None:
            best_det = None

            if self.primary_track.offscreen:
                if self.primary_track.last_seen_frame is None:
                    dt = self.max_missed
                else:
                    dt = self.frame_idx - self.primary_track.last_seen_frame

                max_allowed_dist = self.primary_base_dist + self.primary_max_speed * dt
            else:
                max_allowed_dist = self.primary_base_dist

            best_cost = max_allowed_dist

            for det in detections:
                cost = self._cost(self.primary_track, det)
                if cost < best_cost:
                    best_cost = cost
                    best_det = det

            if best_det is not None:
                self.primary_track.assign(best_det, self.frame_idx)
                matched.add(id(best_det))
                active_tracks.append(self.primary_track)
            else:
                self.primary_track.mark_missed()

                if self.primary_track.missed >= self.max_missed:
                    self.primary_track.offscreen = True

                active_tracks.append(self.primary_track)

        # ---------- 2. Recover lost tracks ----------
        still_lost = []

        for track in self.lost_tracks:
            best_det = None
            best_cost = self.max_dist * 2  # looser gating

            for det in detections:
                if id(det) in matched:
                    continue
                cost = self._cost(track, det)
                if cost < best_cost:
                    best_cost = cost
                    best_det = det

            if best_det is not None:
                track.assign(best_det)
                matched.add(id(best_det))
                active_tracks.append(track)
            else:
                still_lost.append(track)

        self.lost_tracks = [t for t in still_lost if t.missed <= self.max_missed]

        # ---------- 3. Create new tracks (guarded) ----------
        for det in detections:
            if id(det) in matched:
                continue
            if not self._can_create_new(det):
                continue
            if self._suppressed_birth(det):
                continue

            t = WormTrack(
                self._make_id(),
                det.centroid,
                det.area,
                det.length,
                self.min_skel_length,
                self.min_area,
                self.min_motion,
                self.well_id,
                role='secondary'
            )
            t.assign(det, self.frame_idx)
            active_tracks.append(t)
            self.tracks.append(t)

        # ---------- 4. Cleanup ----------

        self.tracks = [
            t for t in self.tracks
            if (
                t.role == "primary"
                or t.age >= self.min_secondary_persistence
                or t.missed <= self.max_missed
            )
        ]

        self.frame_idx += 1
        return active_tracks

    def detect(self, thresh, mask=None, region_mask=None):
        fg = thresh.astype(np.uint8).copy()

        if mask is not None:
            fg[mask == 0] = 0

        k = max(3, 9)
        if k % 2 == 0:
            k += 1

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)

        _, labels, stats, centroids = find_components(fg)

        detections = []

        for label in range(1, stats.shape[0]):
            x, y, w, h, area = stats[label]
            if area < self.min_new_area:
                continue

            cx, cy = centroids[label]

            instance_mask = (labels[y : y + h, x : x + w] == label).astype(np.uint8) * 255

            region = "core"
            if region_mask is not None:
                region_slice = region_mask[y : y + h, x : x + w]
                wall_pixels = np.sum((instance_mask > 0) & (region_slice == 2))
                core_pixels = np.sum((instance_mask > 0) & (region_slice == 1))
                if wall_pixels > core_pixels:
                    region = "wall"

            curve_centroid, end1, end2, length = extract_curve(
                instance_mask,
                (x, y, w, h),
                area,
            )

            detections.append(
                WormDetection(
                    centroid=(cx, cy),
                    end1=end1,
                    end2=end2,
                    area=area,
                    length=length,
                    region=region,
                )
            )

        return detections
