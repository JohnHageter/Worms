from dataclasses import replace
from typing import List, Dict, Optional, Tuple
from imgui import end
import numpy as np

from Module.Tracking.Configuration import TrackingParameters
from Module.Tracking.Orientation import assign_head_tail
from Module.Tracking.Skeletonize import skeleton_longest_path
from Module.Tracking.StateManager import TrackState
from Module.Tracking.OverlapResolver import OverlapResolver
from Module.Tracking.Detection import BlobDetection
from Module.Tracking.utils import Track, TentativeTrack, masks_touch


class MultiOrganismTracker:
    def __init__(self, params):
        self.p: TrackingParameters = params
        self.frame_idx = 0
        self.next_id = 1

        self.active_tracks: Dict[int, Track] = {}
        self.lost_tracks: Dict[int, Track] = {}
        self.tentative_tracks: List[TentativeTrack] = []

        self.overlap_resolver = OverlapResolver(
            overlap_distance=params.overlap_distance,
            min_approach_speed=params.min_speed_for_overlap,
            max_overlap_frames=params.max_overlap_frames,
        )

    def _det_centroid(self, det: BlobDetection) -> np.ndarray:
        x0, y0, w, h = det.bbox
        ys, xs = np.where(det.mask)

        if xs.size == 0:
            cx = x0 + w / 2
            cy = y0 + h / 2
        else:
            cx = x0 + xs.mean()
            cy = y0 + ys.mean()

        return np.array([cx, cy], dtype=float)

    def _det_area(self, det: BlobDetection) -> float:
        return float(np.count_nonzero(det.mask))

    def _det_endpoints(self, det: BlobDetection) -> Optional[np.ndarray]:

        endpoints_local, path_local, length = skeleton_longest_path(det.mask)

        if endpoints_local is not None and path_local is not None and length >= self.p.min_length_for_skeleton:
            x0, y0, _, _ = det.bbox
            endpoints_xy = endpoints_local + np.array([x0,y0], dtype=np.float32)
            path_xy = path_local + np.array([x0, y0], dtype=np.float32)
        else:
            endpoints_xy = None
            path_xy = None

        return endpoints_xy

    def _associate(self, detections, det_centroids, det_areas):
        detections = list(detections)
        assignments = {}
        used_tracks = set()
        used_dets = set()

        for di, c in enumerate(det_centroids):
            candidates = []
            if di in used_dets:
                continue

            for tid, t in self.active_tracks.items():
                delta = c - t.predicted
                dist = np.linalg.norm(delta)

                if t.state.current == TrackState.OVERLAPPING:
                    if dist > self.p.overlap_distance:
                        continue

                if dist > self.p.max_match_distance:
                    continue

                angle_cost = 0.0
                vnorm = np.linalg.norm(t.velocity)
                if vnorm > 1e-6:
                    angle_cost = 1.0 - np.dot(
                        t.velocity / vnorm,
                        delta / (dist + 1e-6),
                    )

                cost = dist + self.p.motion_angle_weight * angle_cost
                candidates.append((tid, cost))

            if not candidates:
                continue

            candidates.sort(key=lambda x: x[1])
            best_tid, best_cost = candidates[0]

            if len(candidates) > 1:
                second_cost = candidates[1][1]
                if best_cost / second_cost > 0.9:
                    detections[di] = replace(detections[di], fate="ignored_ambiguous")
                    continue

            if best_tid in used_tracks:
                continue

            assignments[best_tid] = di
            used_tracks.add(best_tid)
            used_dets.add(di)

            detections[di] = replace(detections[di], fate="matched_active")

        unassigned_dets = [i for i in range(len(detections)) if i not in used_dets]
        return assignments, unassigned_dets, detections

    def _update_assigned_tracks(self, assignments, detections, det_centroids, det_areas, det_endpoints):
        for tid, di in assignments.items():
            t = self.active_tracks[tid]
            t.last_detection = detections[di]
            new_c = det_centroids[di]

            t.velocity = new_c - t.centroid
            t.velocity_history.append(t.velocity.copy())

            t.centroid = new_c
            t.centroid_history.append(new_c.copy())

            t.predicted = t.centroid + t.velocity
            t.area = det_areas[di]
            t.last_seen = self.frame_idx
            t.age += 1
            t.missed = 0
            t.endpoints = det_endpoints[di]

            if (t.endpoints is not None 
                and t.endpoints.shape == (2,2) 
                and not t.state.current == TrackState.OVERLAPPING):
                t.ht_state = assign_head_tail(params=self.p, 
                                              centroid=t.centroid, 
                                              velocity=t.velocity, 
                                              endpoints_xy=t.endpoints, 
                                              prev_state=t.ht_state, 
                                              shape_hint=None
                                              )

            if not t.state.current == TrackState.OVERLAPPING:
                if t.expected_area is None:
                    t.expected_area = t.area
                else:
                    alpha = 0.1  
                    t.expected_area = (1 - alpha) * t.expected_area + alpha * t.area

            if t.state.current == TrackState.LOST:
                t.state.transition(
                    new_state=TrackState.ACTIVE,
                    frame_idx=self.frame_idx,
                    reason="reacquired",
                )

    def _update_unmatched_tracks(self, assignments):
        assigned_ids = set(assignments.keys())

        for tid, t in list(self.active_tracks.items()):
            if t.age == 1:
                continue

            if t.state.current in (
                TrackState.INTERACTING,
                TrackState.OVERLAPPING,
            ):
                continue

            if tid not in assigned_ids:
                t.missed += 1

                if t.missed == 1:
                    t.state.transition(
                        new_state=TrackState.LOST,
                        frame_idx=self.frame_idx,
                        reason="track_unmatched",
                    )

                if t.missed > self.p.max_missed_frames:
                    t.state.transition(
                        new_state=TrackState.TERMINATED,
                        frame_idx=self.frame_idx,
                        reason="exceeded_missed_frames",
                    )
                    self.active_tracks.pop(tid)
                    self.lost_tracks.pop(tid, None)
                else:
                    self.lost_tracks[tid] = t

    def _handle_tentatives(self, unassigned_dets, detections, det_centroids, det_areas, frame_used_dets):
        detections = list(detections)
        matched_tentative_ids = set()
        used_dets = set()

        if any(
            t.state.current in (TrackState.INTERACTING, TrackState.OVERLAPPING)
            for t in self.active_tracks.values()
        ):
            return detections

        for tent in self.tentative_tracks:
            best_di = None
            best_dist = self.p.tentative_match_distance

            for di in unassigned_dets:
                if di in used_dets:
                    continue
                d = np.linalg.norm(det_centroids[di] - tent.centroid)
                if d < best_dist:
                    best_dist = d
                    best_di = di

            if best_di is not None:
                tent.centroid = det_centroids[best_di]
                tent.last_seen = self.frame_idx
                tent.seen_frames += 1
                tent.missed = 0

                matched_tentative_ids.add(id(tent))
                used_dets.add(best_di)
                frame_used_dets.add(best_di)

                detections[best_di] = replace(detections[best_di], fate="matched_tentative")

        unassigned_dets[:] = [di for di in unassigned_dets if di not in used_dets]

        self.tentative_tracks = [
            t
            for t in self.tentative_tracks
            if (id(t) in matched_tentative_ids)
            or (t.missed + 1 <= self.p.tentative_max_missed)
        ]

        for tent in list(self.tentative_tracks):
            if tent.seen_frames >= self.p.tentative_confirm_frames:
                self._promote_tentative(tent)
                self.tentative_tracks = [
                    t for t in self.tentative_tracks
                    if t is not tent
                ]

        for di in unassigned_dets:
            centroid = det_centroids[di]

            suppress = any(
                t.state.current in (TrackState.INTERACTING, TrackState.OVERLAPPING)
                and np.linalg.norm(t.predicted - centroid) < self.p.interaction_distance
                for t in self.active_tracks.values()
            )

            if suppress:
                detections[di] = replace(detections[di], fate="suppressed_interaction")
                continue

            self.tentative_tracks.append(
                TentativeTrack(
                    centroid=centroid,
                    velocity=np.zeros(2),
                    area=det_areas[di],
                    bbox=detections[di].bbox,
                    last_seen=self.frame_idx,
                    born_near_track_id=None,
                )
            )

        return detections

    def _promote_tentative(self, tentative: TentativeTrack):

        for t in self.active_tracks.values():
            if (
                t.state.current == TrackState.ACTIVE
                and not t.state.current == TrackState.OVERLAPPING
                and np.linalg.norm(t.predicted - tentative.centroid) < self.p.max_match_distance
            ):
                return  

        tid = self.next_id
        self.next_id += 1

        track = Track(
            id=tid,
            centroid=tentative.centroid.copy(),
            velocity=tentative.velocity.copy(),
            expected_area=tentative.area,
            expected_bbox=tentative.bbox, 
            last_area=tentative.area,
            interaction_start_frame=None,
            entry_centroid=None,
            entry_velocity=None,
            merged_blob_id=None,
            overlap_start_frame=None,
            predicted=tentative.centroid + tentative.velocity,
            area=tentative.area,
            age=1,
            last_seen=tentative.last_seen,
        )

        track.state.transition(
            new_state=TrackState.ACTIVE,
            frame_idx=self.frame_idx,
            reason="tentative_confirmed",
        )

        self.active_tracks[tid] = track

        self.tentative_tracks = [
            t for t in self.tentative_tracks
            if t is not tentative
        ]

    def _nearest_active_track(self, centroid: np.ndarray, max_dist: float = 50.0):
        best_id = None
        best_dist = max_dist

        for tid, t in self.active_tracks.items():
            if t.state.current != TrackState.INTERACTING:
                continue

            d = np.linalg.norm(t.centroid - centroid)
            if d < best_dist:
                best_dist = d
                best_id = tid

        return best_id

    def _find_interacting_tracks(self):
        return {
            t.id
            for t in self.active_tracks.values()
            if t.state.current == TrackState.INTERACTING
        }

    def _update_interacting_tracks(self, detections, det_centroids, det_areas):
        for t in self.active_tracks.values():
            if t.state.current != TrackState.INTERACTING:
                continue

            separated = all(
                np.linalg.norm(t.centroid - other.centroid) > self.p.interaction_distance
                for other in self.active_tracks.values()
                if other.id != t.id
            )

            if separated:
                t.overlap_partner_ids.clear()
                t.state.transition(
                    new_state=TrackState.ACTIVE,
                    frame_idx=self.frame_idx,
                    reason="interaction_resolved_separation",
                )

    def _update_overlapping_tracks(self, detections):
        overlapping = [
            t
            for t in self.active_tracks.values()
            if t.state.current == TrackState.OVERLAPPING
        ]

        if len(overlapping) < 2:
            return

        t1, t2 = overlapping[:2]

        def nearest_det(track):
            best = None
            best_d = float("inf")
            for d in detections:
                c = self._det_centroid(d)
                d0 = np.linalg.norm(c - track.centroid)
                if d0 < best_d:
                    best_d = d0
                    best = d
            return best

        d1 = nearest_det(t1)
        d2 = nearest_det(t2)

        if d1 is None or d2 is None:
            return

        if not masks_touch(d1, d2):
            t1.state.transition(
                new_state=TrackState.INTERACTING,
                frame_idx=self.frame_idx,
                reason="overlap_ended_masks_separated",
            )
            t2.state.transition(
                new_state=TrackState.INTERACTING,
                frame_idx=self.frame_idx,
                reason="overlap_ended_masks_separated",
            )
            
    def _overlap_candidates(self, tracks, detections, radius):
        cand = []
        for i, d in enumerate(detections):
            c = self._det_centroid(d)
            if any(np.linalg.norm(c - t.centroid) < radius for t in tracks):
                cand.append(i)
        return cand

    def _handle_overlap_detection(self):
        tracks = list(self.active_tracks.values())

        for i, t1 in enumerate(tracks):
            for j in range(i + 1, len(tracks)):
                t2 = tracks[j]

                if t1.state.current in (TrackState.LOST, TrackState.TERMINATED):
                    continue
                if t2.state.current in (TrackState.LOST, TrackState.TERMINATED):
                    continue

                if t1.last_detection is None or t2.last_detection is None:
                    continue

                if (
                    t1.state.current == TrackState.ACTIVE
                    and t2.state.current == TrackState.ACTIVE
                ):
                    dist = np.linalg.norm(t1.centroid - t2.centroid)
                    if dist <= self.p.interaction_distance:
                        t1.state.transition(
                            new_state=TrackState.INTERACTING,
                            frame_idx=self.frame_idx,
                            reason="entered_interaction",
                            details=f"near track {t2.id}",
                        )
                        t2.state.transition(
                            new_state=TrackState.INTERACTING,
                            frame_idx=self.frame_idx,
                            reason="entered_interaction",
                            details=f"near track {t1.id}",
                        )

                if masks_touch(t1.last_detection, t2.last_detection):
                    if t1.state.current != TrackState.OVERLAPPING:
                        t1.overlap_partner_ids = [t2.id]
                        t2.overlap_start_frame = self.frame_idx
                        t1.state.transition(
                            new_state=TrackState.OVERLAPPING,
                            frame_idx=self.frame_idx,
                            reason="mask_contact",
                        )

                    if t2.state.current != TrackState.OVERLAPPING:
                        t2.overlap_partner_ids = [t1.id]
                        t2.overlap_start_frame = self.frame_idx
                        t2.state.transition(
                            new_state=TrackState.OVERLAPPING,
                            frame_idx=self.frame_idx,
                            reason="mask_contact",
                        )

                    self.overlap_resolver.register(t1, t2, frame_idx=self.frame_idx)

    def _force_resolve_long_overlaps(self):
        """
        Prevent tracks from staying in OVERLAPPING forever.
        """
        overlapping = [
            t
            for t in self.active_tracks.values()
            if t.state.current == TrackState.OVERLAPPING
        ]

        for t in overlapping:
            if t.overlap_start_frame is None:
                continue

            if self.frame_idx - t.overlap_start_frame >= self.p.max_overlap_frames:
                for t2 in overlapping:
                    t2.state.transition(
                        new_state=TrackState.INTERACTING,
                        frame_idx=self.frame_idx,
                        reason="overlap_timeout_force_resolve",
                    )
                    t2.overlap_start_frame = None
                return

    def _resolve_overlaps(self, det_centroids, frame_used_dets):
        forced = self.overlap_resolver.resolve(
            self.active_tracks,
            det_centroids,
            frame_idx=self.frame_idx,
        )

        if not forced:
            return

        used_dets = set()
        used_tracks = set()

        for di, tid in forced.items():
            if tid not in self.active_tracks:
                continue

            if di < 0 or di >= len(det_centroids):
                continue

            if di in used_dets or tid in used_tracks:
                continue

            if di in frame_used_dets:
                continue

            t = self.active_tracks[tid]
            new_c = det_centroids[di]

            pred_error = np.linalg.norm(new_c - t.predicted)
            if pred_error > self.p.max_match_distance:
                continue

            conflict = False
            for other in self.active_tracks.values():
                if other.id == tid:
                    continue
                if other.state.current == TrackState.ACTIVE:
                    if np.linalg.norm(other.centroid - new_c) < self.p.overlap_distance:
                        conflict = True
                        break

            if conflict:
                continue

            used_dets.add(di)
            used_tracks.add(tid)
            frame_used_dets.add(di)

            t.velocity = new_c - t.centroid
            t.centroid = new_c
            t.last_seen = self.frame_idx
            t.missed = 0

            t.state.transition(
                new_state=TrackState.INTERACTING,
                frame_idx=self.frame_idx,
                reason="overlap_resolved_motion",
            )

            t.overlap_partner_ids.clear()
            t.overlap_start_frame = None

    def update(self, detections: List[BlobDetection]) -> List[Track]:
        self.frame_idx += 1

        for t in self.active_tracks.values():
            t.predict()

        det_centroids = [self._det_centroid(d) for d in detections]
        det_areas = [self._det_area(d) for d in detections]
        det_endpoints = [self._det_endpoints(d) for d in detections]

        assignments, unassigned_dets, detections = self._associate(
            detections, det_centroids, det_areas
        )
        frame_used_dets = set(assignments.values())        
        self._update_assigned_tracks(assignments, detections, det_centroids, det_areas, det_endpoints)
        self._handle_overlap_detection()
        self._update_overlapping_tracks(detections)

        self._update_interacting_tracks(detections, det_centroids, det_areas)
        detections = self._handle_tentatives(
            unassigned_dets, detections, det_centroids, det_areas, frame_used_dets
        )
        self._update_unmatched_tracks(assignments)
        self._force_resolve_long_overlaps()
        self._resolve_overlaps(det_centroids, frame_used_dets)

        # print(
        #     f"[frame {self.frame_idx}] "
        #     f"tracks_active={len(self.active_tracks)} "
        #     f"tracks_lost={len(self.lost_tracks)} "
        #     f"assignments={len(assignments)}"
        # )

        # for t in self.active_tracks.values():
        #     print(t.id, t.state.current, t.missed)

        for i, det in enumerate(detections):
            if det.fate == "created":
                detections[i] = replace(det, fate="unmatched_detection")

        self.last_detections = detections

        return list(self.active_tracks.values())
