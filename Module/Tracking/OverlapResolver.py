from dataclasses import dataclass
from typing import Dict, Tuple, Any
import numpy as np


@dataclass
class OverlapRecord:
    track_ids: Tuple[int, int]
    start_frame: int
    last_frame: int
    entry_centroids: Dict[int, np.ndarray]
    entry_velocities: Dict[int, np.ndarray]


class OverlapResolver:

    def __init__(
        self,
        *,
        overlap_distance: float,
        min_approach_speed: float,
        max_overlap_frames: int,
    ):
        self.overlap_distance = overlap_distance
        self.min_approach_speed = min_approach_speed
        self.max_overlap_frames = max_overlap_frames
        self.active_overlaps: Dict[frozenset, OverlapRecord] = {}

    def register(self, t1, t2, frame_idx: int):
        key = frozenset((t1.id, t2.id))

        if key not in self.active_overlaps:
            self.active_overlaps[key] = OverlapRecord(
                track_ids=(t1.id, t2.id),
                start_frame=frame_idx,
                last_frame=frame_idx,
                entry_centroids={
                    t1.id: t1.centroid.copy(),
                    t2.id: t2.centroid.copy(),
                },
                entry_velocities={
                    t1.id: t1.velocity.copy(),
                    t2.id: t2.velocity.copy(),
                },
            )
        else:
            self.active_overlaps[key].last_frame = frame_idx


    def resolve(
        self,
        tracks: Dict[int, Any],
        det_centroids: list[np.ndarray],
        frame_idx: int,
    ) -> Dict[int, int]:
        """
        Returns: mapping detection_index -> track_id
        """

        forced_assignments: Dict[int, int] = {}

        for key, record in list(self.active_overlaps.items()):

            if frame_idx - record.last_frame <= self.max_overlap_frames:
                continue

            id1, id2 = record.track_ids

            if id1 not in tracks or id2 not in tracks:
                del self.active_overlaps[key]
                continue

            t1 = tracks[id1]
            t2 = tracks[id2]

            p1 = record.entry_centroids[id1] + record.entry_velocities[id1]
            p2 = record.entry_centroids[id2] + record.entry_velocities[id2]

            cand1 = []
            cand2 = []

            for i, c in enumerate(det_centroids):
                d1 = np.linalg.norm(c - t1.centroid)
                d2 = np.linalg.norm(c - t2.centroid)

                if d1 < self.overlap_distance * 2:
                    cand1.append((i, c))

                if d2 < self.overlap_distance * 2:
                    cand2.append((i, c))

            if len(cand1) == 0 or len(cand2) == 0:
                continue


            best_pair = None
            best_cost = float("inf")

            for i, c_i in cand1:
                for j, c_j in cand2:

                    if i == j:
                        continue  


                    if (
                        np.linalg.norm(c_i - p1) > self.overlap_distance * 3
                        or np.linalg.norm(c_j - p2) > self.overlap_distance * 3
                    ):
                        continue

                    cost_12 = np.linalg.norm(c_i - p1) + np.linalg.norm(c_j - p2)
                    cost_21 = np.linalg.norm(c_i - p2) + np.linalg.norm(c_j - p1)

                    if cost_12 < best_cost:
                        best_cost = cost_12
                        best_pair = (i, id1, j, id2)

                    if cost_21 < best_cost:
                        best_cost = cost_21
                        best_pair = (i, id2, j, id1)

            if best_pair is None:
                continue

            di1, tid1, di2, tid2 = best_pair

   
            if di1 in forced_assignments or di2 in forced_assignments:
                continue

            forced_assignments[di1] = tid1
            forced_assignments[di2] = tid2

            del self.active_overlaps[key]

        return forced_assignments
