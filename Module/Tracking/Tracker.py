from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import numpy as np

from Module.Tracking.Detection import BlobDetection
from Module.Tracking.Orientation import HeadTailState, assign_head_tail
from Module.Tracking.Skeletonize import skeleton_longest_path


@dataclass
class Track:
    id: int
    centroid: np.ndarray
    velocity: np.ndarray
    area: float
    region: int
    age: int
    missed: int
    last_seen: int
    bbox: Optional[Tuple[int, int, int, int]] = None
    endpoints: Optional[np.ndarray] = None
    skeleton_length: float = 0.0
    ht: Optional[HeadTailState] = None
    active: bool = True


@dataclass
class TentativeTrack:
    centroid: np.ndarray
    velocity: np.ndarray
    area: float
    region: int

    confirm_count: int = 1
    missed: int = 0
    last_seen: int = 0

    last_det: Optional[BlobDetection] = None


class MultiOrganismTracker:
    def __init__(self, params):
        self.p = params
        self.next_id = 1
        self.frame_idx = 0

        self.active_tracks: List[Track] = []
        self.lost_tracks: List[Track] = []  
        self.tentative_tracks: list[TentativeTrack] = []

    def _det_area(self, det) -> float:
        return float(np.count_nonzero(det.mask))

    def _det_centroid(self, det) -> np.ndarray:
        x0, y0, w, h = det.bbox
        ys, xs = np.where(det.mask > 0)
        if xs.size == 0:
            return np.array([x0 + w / 2.0, y0 + h / 2.0], dtype=float)
        return np.array([x0 + xs.mean(), y0 + ys.mean()], dtype=float)

    def _new_track(self, centroid, area, region, bbox) -> Track:
        t = Track(
            id=self.next_id,
            centroid=np.asarray(centroid, dtype=float),
            velocity=np.zeros(2, dtype=float),
            area=float(area),
            region=int(region),
            age=1,
            missed=0,
            last_seen=self.frame_idx,
            bbox = tuple(map(int,bbox)),
            endpoints=None,
            skeleton_length=0.0,
            ht=HeadTailState(),  
            active=True,
        )
        self.next_id += 1
        return t

    def _match_active(
        self, tracks: List[Track], det_centroids: List[np.ndarray]
    ) -> List[Tuple[int, int]]:
        """
        Greedy nearest neighbor with gating to match detections to active tracks.
        """
        used_t = set()
        used_d = set()
        pairs: List[Tuple[int, int]] = []

        candidates = []
        for ti, t in enumerate(tracks):
            pred = t.centroid + t.velocity
            for di, c in enumerate(det_centroids):
                d = float(np.linalg.norm(pred - c))
                if d <= self.p.max_match_distance:
                    candidates.append((d, ti, di))

        candidates.sort(key=lambda x: x[0])

        for _, ti, di in candidates:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            pairs.append((ti, di))

        return pairs

    def _match_tentative(self, det_centroids):
        """
        Greedy matching for tentative tracks.
        Returns list of (tent_idx, det_idx)
        """
        pairs = []
        used_t = set()
        used_d = set()
        candidates = []

        for ti, t in enumerate(self.tentative_tracks):
            pred = t.centroid + t.velocity
            for di, c in enumerate(det_centroids):
                d = float(np.linalg.norm(pred - c))
                if d <= self.p.tentative_match_distance:
                    candidates.append((d, ti, di))

        candidates.sort(key=lambda x: x[0])
        for _, ti, di in candidates:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            pairs.append((ti, di))

        return pairs

    def _reid_thresholds(self, track: Track) -> Tuple[int, float]:
        """
        Wall-lost tracks get bigger grace + distance (worm-specific behavior).
        """
        wall_code = getattr(self.p, "wall_region_code", 2)
        if track.region == wall_code:
            return getattr(
                self.p,
                "wall_reid_grace_frames",
                getattr(self.p, "reid_grace_frames", 200),
            ), getattr(
                self.p,
                "wall_reid_max_distance",
                getattr(self.p, "reid_max_distance", 120.0),
            )
        return getattr(self.p, "reid_grace_frames", 200), getattr(
            self.p, "reid_max_distance", 120.0
        )

    def _prune_lost(self) -> None:
        """
        Remove lost tracks beyond grace window or beyond max_missed_frames.
        """
        keep = []
        for t in self.lost_tracks:
            dt = self.frame_idx - t.last_seen
            grace, _ = self._reid_thresholds(t)
            if dt <= grace and dt <= self.p.max_missed_frames:
                keep.append(t)
        self.lost_tracks = keep

    def _best_lost_match(
        self, det_centroid: np.ndarray, det_area: float
    ) -> Optional[int]:
        """
        Pick best lost track by score:
            dist(predicted, det) + reid_area_weight * relative_area_error
        """
        best_i = None
        best_score = float("inf")

        area_w = float(getattr(self.p, "reid_area_weight", 0.0))

        for i, t in enumerate(self.lost_tracks):
            dt = self.frame_idx - t.last_seen
            grace, maxd = self._reid_thresholds(t)
            if dt > grace:
                continue

            pred = t.centroid + t.velocity * dt
            dist = float(np.linalg.norm(pred - det_centroid))
            if dist > maxd:
                continue

            area_err = abs(t.area - det_area) / max(t.area, 1.0)
            score = dist + area_w * area_err

            if score < best_score:
                best_score = score
                best_i = i

        return best_i

    def _apply_geometry_and_headtail(self, t: Track, det) -> None:
        """
        Compute skeleton longest path endpoints/length.
        Drop endpoints if below min_skeleton_length.
        If endpoints exist, update head/tail confidence.
        """
        if getattr(t, "ht", None) is None:
            t.ht = HeadTailState()

        endpoints_bbox, path_bbox, skel_len = skeleton_longest_path(det.mask)

        if endpoints_bbox is None or skel_len < self.p.min_skeleton_length:
            t.endpoints = None
            t.skeleton_length = 0.0
            return

        x0, y0, _, _ = det.bbox
        endpoints = endpoints_bbox.astype(float).copy()
        endpoints[:, 0] += x0
        endpoints[:, 1] += y0

        t.endpoints = endpoints
        t.skeleton_length = float(skel_len)

        if path_bbox is not None and len(path_bbox) > 0:
            mid = len(path_bbox) // 2
            t.centroid = np.array([path_bbox[mid, 0] + x0, path_bbox[mid, 1] + y0], dtype=float)

        t.ht = assign_head_tail(
            self.p,
            centroid=t.centroid,
            velocity=t.velocity,
            endpoints_xy=t.endpoints,
            prev_state=t.ht, # type: ignore
            shape_hint=None,
        )

    def update(self, detections):
        self.frame_idx += 1

        raw_ids = set(id(d) for d in detections)
        dets = [d for d in detections if self._det_area(d) >= self.p.min_blob_area]
        dets.sort(key=self._det_area, reverse=True)
        dets = dets[: self.p.max_detectable_blobs]

        kept_ids = set(id(d) for d in dets)
        filtered_ids = raw_ids - kept_ids

        det_centroids = [self._det_centroid(d) for d in dets]
        det_areas = [self._det_area(d) for d in dets]
        det_regions = [int(d.region) for d in dets]

        confirmed_ids: set[int] = set()

        pairs = self._match_active(self.active_tracks, det_centroids)
        matched_tracks = set(ti for ti, _ in pairs)
        matched_dets = set(di for _, di in pairs)

        for di in matched_dets:
            confirmed_ids.add(id(dets[di]))

        remaining_det_indices = [di for di in range(len(dets)) if di not in matched_dets]
        rem_centroids = [det_centroids[di] for di in remaining_det_indices]
        local_to_global = {local: global_i for local, global_i in enumerate(remaining_det_indices)}
        tent_pairs = self._match_tentative(rem_centroids)

        matched_tent = set()
        matched_local = set()

        for tent_i, local_di in tent_pairs:
            global_di = local_to_global[local_di]
            det = dets[global_di]

            t = self.tentative_tracks[tent_i]
            new_c = det_centroids[global_di]

            t.velocity = new_c - t.centroid
            t.centroid = new_c
            t.area = det_areas[global_di]
            t.region = det_regions[global_di]
            t.confirm_count += 1
            t.missed = 0
            t.last_seen = self.frame_idx
            t.last_det = det

            matched_tent.add(tent_i)
            matched_local.add(local_di)

        new_tentative = []
        for i, t in enumerate(self.tentative_tracks):
            if i in matched_tent:
                new_tentative.append(t)
            else:
                t.missed += 1
                if t.missed <= self.p.tentative_max_missed:
                    new_tentative.append(t)

        self.tentative_tracks = new_tentative

        for local_di in range(len(rem_centroids)):
            if local_di in matched_local:
                continue
            global_di = local_to_global[local_di]
            det = dets[global_di]

            self.tentative_tracks.append(
                TentativeTrack(
                    centroid=det_centroids[global_di].copy(),
                    velocity=np.zeros(2, dtype=float),
                    area=det_areas[global_di],
                    region=det_regions[global_di],
                    confirm_count=1,
                    missed=0,
                    last_seen=self.frame_idx,
                    last_det=det,
                )
            )
            

        promoted: List[Track] = []
        remaining_tent: List[TentativeTrack] = []

        for tt in self.tentative_tracks:
            if tt.confirm_count >= self.p.min_confirm_frames:
                if tt.last_det is None:
                    remaining_tent.append(tt)
                    continue

                det = tt.last_det
                new_track = self._new_track(tt.centroid, tt.area, tt.region, det.bbox)
                self._apply_geometry_and_headtail(new_track, det)

                promoted.append(new_track)
            else:
                remaining_tent.append(tt)

        self.tentative_tracks = remaining_tent
        self.active_tracks.extend(promoted)

        new_active: List[Track] = []

        for ti, di in pairs:
            t = self.active_tracks[ti]
            new_c = det_centroids[di]

            t.velocity = new_c - t.centroid
            t.centroid = new_c
            t.area = det_areas[di]
            t.region = det_regions[di]
            t.age += 1
            t.missed = 0
            t.last_seen = self.frame_idx
            t.bbox = tuple(map(int, dets[di].bbox))
            t.active = True
            self._apply_geometry_and_headtail(t, dets[di])

            new_active.append(t)

        for i, t in enumerate(self.active_tracks):
            if i in matched_tracks:
                continue
            t.missed += 1
            t.velocity[:] = 0.0
            t.active = False
            self.lost_tracks.append(t)

        self.active_tracks = new_active
        self._prune_lost()

        for di, det in enumerate(dets):
            if di in matched_dets:
                continue

            c = det_centroids[di]
            a = det_areas[di]
            r = det_regions[di]

            li = self._best_lost_match(c, a)
            if li is not None:
                confirmed_ids.add(id(det))
                t = self.lost_tracks.pop(li)
                t.active = True
                t.missed = 0
                t.velocity = c - t.centroid
                t.centroid = c
                t.area = a
                t.region = r
                t.age += 1
                t.last_seen = self.frame_idx
                t.bbox = tuple(map(int,det.bbox))
                self._apply_geometry_and_headtail(t, det)

                self.active_tracks.append(t)
            else:
                if self._suppress_birth_near_active(c, det.bbox):
                    continue
                confirmed_ids.add(id(det))
                t = self._new_track(c, a, r, det.bbox)
                self._apply_geometry_and_headtail(t, det)

                self.active_tracks.append(t)

        # 0 = filtered out (red)
        # 1 = tentative (blue)
        # 2 = kept/confirmed (green)
        if getattr(self.p, "debug", False):
            tentative_ids = kept_ids - confirmed_ids

            self._debug_det_status = {}
            for did in filtered_ids:
                self._debug_det_status[did] = 0
            for did in tentative_ids:
                self._debug_det_status[did] = 1
            for did in confirmed_ids:
                self._debug_det_status[did] = 2

            self._debug_counts = {
                "raw": len(detections),
                "kept": len(dets),
                "filtered": len(filtered_ids),
                "tentative": len(tentative_ids),
                "confirmed": len(confirmed_ids),
                "active_tracks": len(self.active_tracks),
                "lost_tracks": len(self.lost_tracks),
                "tentative_tracks": len(self.tentative_tracks),
            }

        return self.active_tracks

    def _bbox_iou(self, b1, b2) -> float:
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        ax1, ay1, ax2, ay2 = x1, y1, x1 + w1, y1 + h1
        bx1, by1, bx2, by2 = x2, y2, x2 + w2, y2 + h2
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        if inter == 0:
            return 0.0
        area1 = w1 * h1
        area2 = w2 * h2
        return inter / float(area1 + area2 - inter)


    def _suppress_birth_near_active(self, det_centroid, det_bbox) -> bool:
        """
        Suppress spawning a new ID if this detection is likely a fragment/noise near an existing active track.

        Suppression rules:
        1) If detection centroid is within birth_suppression_dist of any active track centroid -> suppress
        2) If the active track has a bbox AND IoU(track_bbox, det_bbox) >= birth_suppression_iou -> suppress

        Note: If Track objects do not store bbox, IoU suppression is skipped safely.
        """
        for t in self.active_tracks:
            # distance-based suppression
            if np.linalg.norm(t.centroid - det_centroid) <= self.p.birth_suppression_dist:
                return True

            # IoU-based suppression (ONLY if track bbox exists)
            tb = getattr(t, "bbox", None)
            if tb is not None:
                if self._bbox_iou(tb, det_bbox) >= self.p.birth_suppression_iou:
                    return True

        return False
