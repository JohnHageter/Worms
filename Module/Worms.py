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
        self.centroid = np.array(centroid, dtype=float)
        self.prev_centroid = None
        self.area = area
        self.velocity = np.array([0.0,0.0])
        self.head = None
        self.tail = None
        self.missed = 0
        self.head_tail_confident = False  # new flag

    def update_motion(self, new_centroid):
        if self.prev_centroid is not None:
            self.velocity = new_centroid - self.prev_centroid
        else:
            self.velocity = np.array([0.0,0.0])
        self.prev_centroid = new_centroid
        self.centroid = new_centroid

    def compute_head_tail(self, detection):
        if detection.end1 is None or detection.end2 is None:
            self.head_tail_confident = False
            self.head = None
            self.tail = None
            return

        pts = np.array([detection.end1, detection.end2])

        # ---- if previous assignment exists ----
        if self.head_tail_confident and self.head is not None and self.tail is not None:
            # assign based on minimum distance to previous head/tail independently
            head_idx = np.argmin(np.linalg.norm(pts - self.head, axis=1))
            tail_idx = np.argmin(np.linalg.norm(pts - self.tail, axis=1))

            # ensure head and tail are not the same point
            if head_idx == tail_idx:
                tail_idx = 1 - head_idx

            proposed_head, proposed_tail = pts[head_idx], pts[tail_idx]

            # ---- velocity override ----
            if np.linalg.norm(self.velocity) >= 2.0:
                motion_point = self.centroid + self.velocity
                motion_head_idx = np.argmin(np.linalg.norm(pts - motion_point, axis=1))
                motion_tail_idx = 1 - motion_head_idx
                proposed_head, proposed_tail = pts[motion_head_idx], pts[motion_tail_idx]

            self.head = tuple(proposed_head)
            self.tail = tuple(proposed_tail)
            return

        # ---- strict assignment for first confident detection ----
        if np.linalg.norm(self.velocity) < 2.0:
            self.head_tail_confident = False
            return

        motion_point = self.centroid + self.velocity
        dists = np.linalg.norm(pts - motion_point, axis=1)
        if abs(dists[0] - dists[1]) < 2.0:
            self.head_tail_confident = False
            return

        head_idx = np.argmin(dists)
        tail_idx = 1 - head_idx
        self.head = tuple(pts[head_idx])
        self.tail = tuple(pts[tail_idx])
        self.head_tail_confident = True

    def assign(self, detection):
        self.update_motion(detection.centroid)
        self.area = detection.area
        self.missed = 0
        self.compute_head_tail(detection)

    def mark_missed(self):
        self.centroid = np.array([np.nan, np.nan])
        self.velocity = np.array([0.0,0.0])
        self.missed += 1
        # If lost, clear confidence
        if self.missed > 0:
            self.head_tail_confident = False
            self.head = None
            self.tail = None

class WellTracker:
    def __init__(self, well_id, max_dist=50, max_missed=5):
        self.well_id = well_id
        self.label = string.ascii_uppercase[well_id]
        self.tracks = []
        self.next_id = 1
        self.max_dist = max_dist
        self.max_missed = max_missed

    def _make_id(self):
        tid = f"{self.label}{self.next_id}"
        self.next_id += 1
        return tid

    def _cost(self, track, detection):
        if track.prev_centroid is None:
            return 0
        dist = np.linalg.norm(track.prev_centroid - detection.centroid)
        pred = track.prev_centroid + track.velocity
        vel_error = np.linalg.norm(pred - detection.centroid)
        area_ratio = abs(track.area - detection.area)/max(track.area,1)
        return dist + vel_error + 50*area_ratio

    def update(self, detections):
        matched = set()
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
            else:
                track.mark_missed()
        for det in detections:
            if id(det) not in matched:
                t = WormTrack(self._make_id(), self.well_id, det.centroid, det.area)
                t.assign(det)
                self.tracks.append(t)
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]
        return self.tracks

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
                self.well_trackers[well_id] = WellTracker(well_id, self.max_dist, self.max_missed)
            tracks = self.well_trackers[well_id].update(dets)
            all_tracks.extend(tracks)
        for well_id, wt in self.well_trackers.items():
            if well_id not in wells:
                for t in wt.tracks:
                    t.mark_missed()
        return all_tracks