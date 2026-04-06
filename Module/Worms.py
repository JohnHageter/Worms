import numpy as np
import string


class WormDetection:
    def __init__(self, centroid, end1=None, end2=None, area=0, well_id=None):
        self.centroid = np.array(centroid, dtype=float)
        self.end1 = np.array(end1, dtype=float) if end1 is not None else None
        self.end2 = np.array(end2, dtype=float) if end2 is not None else None
        self.area = area
        self.well_id = well_id


class WormTrack:
    def __init__(self, track_id, well_id, centroid, area):
        self.id = track_id
        self.well_id = well_id

        # current positions
        self.centroid = np.array(centroid, dtype=float)
        self.head = None
        self.tail = None

        # previous positions
        self.prev_centroid = None
        self.prev_head = None
        self.prev_tail = None

        self.area = area
        self.velocity = np.array([0.0, 0.0])
        self.missed = 0
        self.head_tail_confident = False

    def update_motion(self, new_centroid):
        if self.prev_centroid is not None:
            self.velocity = new_centroid - self.prev_centroid
        else:
            self.velocity = np.array([0.0, 0.0])
        self.prev_centroid = self.centroid.copy()
        self.centroid = new_centroid

    def compute_head_tail(self, detection, min_area=150, min_motion=2.0):
        """Assign head/tail only if area and motion are sufficient"""
        if detection.end1 is None or detection.end2 is None:
            self.head_tail_confident = False
            self.head = None
            self.tail = None
            return

        if self.area < min_area or np.linalg.norm(self.velocity) < min_motion:
            # Not confident enough to assign head/tail
            self.head_tail_confident = False
            self.head = None
            self.tail = None
            return

        pts = np.array([detection.end1, detection.end2])

        if self.head_tail_confident and self.head is not None and self.tail is not None:
            # Keep continuity with previous head/tail
            head_idx = np.argmin(np.linalg.norm(pts - self.head, axis=1))
            tail_idx = np.argmin(np.linalg.norm(pts - self.tail, axis=1))
            if head_idx == tail_idx:
                tail_idx = 1 - head_idx
            proposed_head, proposed_tail = pts[head_idx], pts[tail_idx]

            # velocity override
            if np.linalg.norm(self.velocity) >= 3.0:
                motion_point = self.centroid + self.velocity
                motion_head_idx = np.argmin(np.linalg.norm(pts - motion_point, axis=1))
                motion_tail_idx = 1 - motion_head_idx
                proposed_head, proposed_tail = (
                    pts[motion_head_idx],
                    pts[motion_tail_idx],
                )

            # store previous positions
            self.prev_head = self.head.copy()
            self.prev_tail = self.tail.copy()

            self.head = proposed_head
            self.tail = proposed_tail
            return

        # First confident assignment
        motion_point = self.centroid + self.velocity
        dists = np.linalg.norm(pts - motion_point, axis=1)
        if abs(dists[0] - dists[1]) < 3.0:
            self.head_tail_confident = False
            return

        head_idx = np.argmin(dists)
        tail_idx = 1 - head_idx

        self.prev_head = self.head.copy() if self.head is not None else None
        self.prev_tail = self.tail.copy() if self.tail is not None else None

        self.head = pts[head_idx]
        self.tail = pts[tail_idx]
        self.head_tail_confident = True

    def assign(self, detection):
        self.update_motion(detection.centroid)
        self.area = detection.area
        self.missed = 0
        self.compute_head_tail(detection)

    def mark_missed(self):
        self.missed += 1
        self.velocity = np.array([0.0, 0.0])
        if self.missed > 0:
            self.head_tail_confident = False
            self.head = None
            self.tail = None


class WellTracker:
    def __init__(
        self, well_id, max_dist=50, max_missed=5, min_new_area=5, area_weight=50
    ):
        self.well_id = well_id
        self.label = string.ascii_uppercase[well_id]
        self.tracks = []
        self.lost_tracks = []
        self.next_id = 1
        self.max_dist = max_dist
        self.max_missed = max_missed
        self.min_new_area = min_new_area
        self.area_weight = area_weight

    def _make_id(self):
        tid = f"{self.label}{self.next_id}"
        self.next_id += 1
        return tid

    def _cost(self, track, detection):
        """Distance + predicted motion + area similarity"""
        if track.prev_centroid is None:
            return 0
        dist = np.linalg.norm(track.prev_centroid - detection.centroid)
        pred = track.prev_centroid + track.velocity
        vel_error = np.linalg.norm(pred - detection.centroid)
        area_ratio = abs(track.area - detection.area) / max(track.area, 1)
        return dist + vel_error + self.area_weight * area_ratio

    def _can_create_new(self, detection):
        """Only create new track if area is large enough and far from existing tracks"""
        if detection.area < self.min_new_area:
            return False
        for t in self.tracks + self.lost_tracks:
            if t.centroid is not None and not np.isnan(t.centroid[0]):
                if np.linalg.norm(t.centroid - detection.centroid) < self.max_dist:
                    return False
        return True

    def update(self, detections):
        matched = set()
        active_tracks = []

        # 1. Try to match existing active tracks
        for track in self.tracks:
            best = None
            best_cost = self.max_dist
            for det in detections:
                if id(det) in matched:
                    continue
                cost = self._cost(track, det)
                if cost < best_cost:
                    best_cost = cost
                    best = det
            if best is not None:
                track.assign(best)
                matched.add(id(best))
                active_tracks.append(track)
            else:
                track.mark_missed()
                if track.missed <= self.max_missed:
                    self.lost_tracks.append(track)

        # 2. Try to rematch lost tracks
        still_lost = []
        for track in self.lost_tracks:
            best = None
            best_dist = self.max_dist * 10  # allow rematch over bigger distance
            for det in detections:
                if id(det) in matched:
                    continue
                cost = self._cost(track, det)
                if cost < best_dist:
                    best = det
                    best_dist = cost
            if best is not None:
                track.assign(best)
                matched.add(id(best))
                active_tracks.append(track)
            else:
                still_lost.append(track)
        self.lost_tracks = still_lost

        # 3. Create new tracks for unmatched detections (buds)
        for det in detections:
            if id(det) not in matched and self._can_create_new(det):
                t = WormTrack(self._make_id(), self.well_id, det.centroid, det.area)
                t.assign(det)
                active_tracks.append(t)
                self.tracks.append(t)

        # 4. Clean up dead tracks
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]

        return active_tracks


class WormTracker:
    def __init__(self, max_dist=50, max_missed=5):
        self.well_trackers = {}
        self.max_dist = max_dist
        self.max_missed = max_missed

    def update(self, detections):
        wells = {}
        for d in detections:
            wells.setdefault(d.well_id, []).append(d)
        all_tracks = []
        for well_id, dets in wells.items():
            if well_id not in self.well_trackers:
                self.well_trackers[well_id] = WellTracker(
                    well_id, self.max_dist, self.max_missed
                )
            tracks = self.well_trackers[well_id].update(dets)
            all_tracks.extend(tracks)
        return all_tracks
