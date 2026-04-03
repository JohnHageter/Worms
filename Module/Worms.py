from typing import Optional, Tuple
from scipy.optimize import linear_sum_assignment
import numpy as np
from typing import Optional, Tuple
import numpy as np


class WormTrack:
    def __init__(self, track_id, worm, frame_idx):
        self.id = track_id
        self.last_centroid = np.array(worm["centroid"], dtype=float)

        self.head = None
        self.tail = None

        self.last_frame = frame_idx
        self.age = 1
        self.missed = 0
        self.well_id = worm["well_id"]

        self.velocity = np.zeros(2)

        self.orientation_locked = False
        self.motion_history = []


class WormTracker:
    def __init__(self, max_dist=80, max_missed=10, fission_dist=50):
        self.max_dist = max_dist
        self.max_missed = max_missed
        self.fission_dist = fission_dist  # max distance to parent for fission
        self.tracks = []
        self.next_id = 0

    def _match_tracks(self, tracks, worms):
        """
        Match existing tracks to current detections using Hungarian algorithm
        with predicted positions.
        """
        if len(tracks) == 0 or len(worms) == 0:
            return [], set(), set()

        n_tracks = len(tracks)
        n_worms = len(worms)
        cost_matrix = np.zeros((n_tracks, n_worms), dtype=float)

        for i, t in enumerate(tracks):
            predicted = t.last_centroid + t.velocity
            for j, w in enumerate(worms):
                cost_matrix[i, j] = np.linalg.norm(predicted - w["centroid"])

        track_idx, worm_idx = linear_sum_assignment(cost_matrix)

        matches = []
        used_tracks, used_worms = set(), set()
        for ti, wi in zip(track_idx, worm_idx):
            if cost_matrix[ti, wi] <= self.max_dist:
                matches.append((ti, wi))
                used_tracks.add(ti)
                used_worms.add(wi)

        return matches, used_tracks, used_worms

    def assign_head_tail(
        self,
        t,
        centroid: Tuple[int, int],
        endA: Optional[Tuple[int, int]],
        endB: Optional[Tuple[int, int]],
    ):
        """
        Hybrid head/tail assignment:
        - Fast motion → use velocity
        - Slow motion → assign by closest distance to previous assignment
        """
        if endA is None or endB is None:
            return t.head, t.tail

        centroid = np.array(centroid, dtype=float)
        prev_centroid = t.last_centroid

        # Update velocity
        step = centroid - prev_centroid
        t.velocity = 0.8 * t.velocity + 0.2 * step
        speed = np.linalg.norm(t.velocity)

        # Motion history smoothing
        if not hasattr(t, "motion_history"):
            t.motion_history = []
        t.motion_history.append(speed)
        if len(t.motion_history) > 10:
            t.motion_history.pop(0)
        avg_speed = np.mean(t.motion_history)

        # Orientation not locked
        if not getattr(t, "orientation_locked", False):
            if t.age < 5 or avg_speed < 1.0:
                return None, None
            t.orientation_locked = True
            vA = np.array(endA) - centroid
            vB = np.array(endB) - centroid
            if np.dot(vA, t.velocity) > np.dot(vB, t.velocity):
                t.head, t.tail = endA, endB
            else:
                t.head, t.tail = endB, endA
            return t.head, t.tail

        # Orientation locked → hybrid assignment
        motion_threshold = 1.0
        if avg_speed >= motion_threshold:
            vA = np.array(endA) - centroid
            vB = np.array(endB) - centroid
            if np.dot(vA, t.velocity) > np.dot(vB, t.velocity):
                new_head, new_tail = endA, endB
            else:
                new_head, new_tail = endB, endA
        else:
            distA_to_head = np.linalg.norm(np.array(endA) - np.array(t.head))
            distB_to_head = np.linalg.norm(np.array(endB) - np.array(t.head))
            if distA_to_head < distB_to_head:
                new_head, new_tail = endA, endB
            else:
                new_head, new_tail = endB, endA

        return new_head, new_tail

    def update(self, worms, frame_idx):
        """
        Update tracker with new detections.
        Only create new IDs if a worm fissions.
        """
        by_well = {}
        for w in worms:
            by_well.setdefault(w["well_id"], []).append(w)

        for well_id, dets in by_well.items():
            well_tracks = [t for t in self.tracks if t.well_id == well_id]

            # 1. Match existing tracks
            matches, used_tracks, used_dets = self._match_tracks(well_tracks, dets)

            for ti, di in matches:
                t = well_tracks[ti]
                w = dets[di]

                head, tail = self.assign_head_tail(
                    t, w["centroid"], w["end1"], w["end2"]
                )
                if head is not None:
                    t.head = head
                if tail is not None:
                    t.tail = tail

                t.last_centroid = np.array(w["centroid"], dtype=float)
                t.last_frame = frame_idx
                t.age += 1
                t.missed = 0

            # 2. Increase missed count for unmatched tracks
            for i, t in enumerate(well_tracks):
                if i not in used_tracks:
                    t.missed += 1

            # 3. Handle unmatched detections → fission only
            for di, w in enumerate(dets):
                if di in used_dets:
                    continue

                # Check all existing tracks (including recently missed ones)
                too_close = False
                for t in well_tracks:
                    dist = np.linalg.norm(t.last_centroid - w["centroid"])
                    if dist < self.fission_dist:
                        too_close = True
                        break

                min_area_for_new_track = 60
                # Only create new ID if far from all existing tracks and large enough
                if not too_close and w["area"] >= min_area_for_new_track:
                    self.tracks.append(WormTrack(self.next_id, w, frame_idx))
                    self.next_id += 1

        # 4. Remove tracks missed too long
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]

        return self.tracks
