from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Set, Any
import numpy as np
from collections import deque, defaultdict
from scipy.optimize import linear_sum_assignment

from Module.Tracking.Orientation import HeadTailState, assign_head_tail
from Module.Tracking.Skeletonize import skeleton_longest_path
from Module.Tracking.Detection import BlobDetection


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
    frame_history: List[int] = field(default_factory=list)
    centroid_history: List[np.ndarray] = field(default_factory=list)
    appearance_signature: Optional[np.ndarray] = None  # wall re‑ID


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



class MotionMemory:
    def __init__(self, decay_rate: float = 0.95, min_motion_frames: int = 3):
        self.decay_rate = decay_rate
        self.min_motion_frames = min_motion_frames
        self.motion_map: Optional[np.ndarray] = None

    def update(self, mask_shape: Tuple[int, int], centroid: np.ndarray) -> bool:
        """Accumulate motion evidence. Returns True if enough motion."""
        if self.motion_map is None:
            self.motion_map = np.zeros(mask_shape, dtype=np.float32)

        self.motion_map *= self.decay_rate
        y, x = int(centroid[1]), int(centroid[0])
        if 0 <= y < self.motion_map.shape[0] and 0 <= x < self.motion_map.shape[1]:
            self.motion_map[y, x] += 1.0

        return np.any(self.motion_map > self.min_motion_frames) # type: ignore



class MultiOrganismTracker:
    def __init__(self, params: Any):
        self.p = params
        self.next_id = 1
        self.frame_idx = 0

        self.active_tracks: List[Track] = []
        self.lost_tracks: List[Track] = []
        self.tentative_tracks: List[TentativeTrack] = []

        self.motion_memories: Dict[int, MotionMemory] = {}

        self._debug_det_status: Dict[int, int] = {}
        self._debug_counts: Dict[str, int] = {}

    def _det_area(self, det: BlobDetection) -> float:
        return float(np.count_nonzero(det.mask))

    def _det_centroid(self, det: BlobDetection) -> np.ndarray:
        x0, y0, w, h = det.bbox
        ys, xs = np.where(det.mask > 0)
        if xs.size == 0:
            return np.array([x0 + w / 2.0, y0 + h / 2.0], dtype=float)
        return np.array([x0 + xs.mean(), y0 + ys.mean()], dtype=float)

    def _compute_appearance_signature(self, det: BlobDetection) -> Optional[np.ndarray]:
        """
        Mask-only appearance signature.
        No skeleton, no full-frame image required.
        """
        mask = det.mask
        if mask is None:
            return None

        mask = mask.astype(bool)
        area = float(mask.sum())
        if area < 10:
            return None

        ys, xs = np.where(mask)

        # Centroid (mask-local)
        cx = xs.mean()
        cy = ys.mean()

        # Second central moments
        dx = xs - cx
        dy = ys - cy

        mu_xx = np.mean(dx * dx)
        mu_yy = np.mean(dy * dy)
        mu_xy = np.mean(dx * dy)

        # Eigenvalues of inertia tensor (shape elongation)
        trace = mu_xx + mu_yy
        det = mu_xx * mu_yy - mu_xy * mu_xy
        temp = max(trace * trace / 4 - det, 0.0)
        l1 = trace / 2 + np.sqrt(temp)
        l2 = trace / 2 - np.sqrt(temp)

        elongation = l1 / max(l2, 1e-6)

        # Bounding-box–normalized centroid
        h, w = mask.shape
        cx_n = cx / max(w, 1)
        cy_n = cy / max(h, 1)

        # Compactness (perimeter^2 / area)
        # simple morphological perimeter
        from scipy.ndimage import binary_erosion
        eroded = binary_erosion(mask)
        perimeter = float(np.sum(mask ^ eroded))
        compactness = (perimeter * perimeter) / max(area, 1.0)

        return np.array(
            [
                area,
                elongation,
                compactness,
                cx_n,
                cy_n,
            ],
            dtype=np.float32,
        )


    def _new_track(
        self,
        centroid: np.ndarray,
        area: float,
        region: int,
        bbox: Tuple[int, int, int, int],
        det: Optional[BlobDetection] = None,
    ) -> Track:
        t = Track(
            id=self.next_id,
            centroid=np.asarray(centroid, dtype=float),
            velocity=np.zeros(2, dtype=float),
            area=float(area),
            region=int(region),
            age=1,
            missed=0,
            last_seen=self.frame_idx,
            bbox=tuple(map(int, bbox)), # type: ignore
            endpoints=None,
            skeleton_length=0.0,
            ht=HeadTailState(),
            active=True,
        )
        if det is not None:
            self._apply_geometry_and_headtail(t, det)
            t.appearance_signature = self._compute_appearance_signature(det)
        self.next_id += 1
        return t

    def _apply_geometry_and_headtail(self, track: Track, det: BlobDetection) -> None:
        """Compute skeleton, endpoints, length, and update head/tail."""
        if track.ht is None:
            track.ht = HeadTailState()

        endpoints_bbox, path_bbox, skel_len = skeleton_longest_path(det.mask)

        if endpoints_bbox is None or skel_len < getattr(
            self.p, "min_skeleton_length", 10
        ):
            track.endpoints = None
            track.skeleton_length = 0.0
            return

        x0, y0, _, _ = det.bbox
        endpoints = endpoints_bbox.astype(float).copy()
        endpoints[:, 0] += x0
        endpoints[:, 1] += y0

        track.endpoints = endpoints
        track.skeleton_length = float(skel_len)

        # Update centroid from skeleton mid‑point
        if path_bbox is not None and len(path_bbox) > 0:
            mid = len(path_bbox) // 2
            track.centroid = np.array(
                [path_bbox[mid, 0] + x0, path_bbox[mid, 1] + y0], dtype=float
            )

        track.ht = assign_head_tail(
            self.p,
            centroid=track.centroid,
            velocity=track.velocity,
            endpoints_xy=track.endpoints,
            prev_state=track.ht,
            shape_hint=None,
        )


    def _update_velocity_smart(
        self, track: Track, new_centroid: np.ndarray, dt_frames: int = 1
    ) -> None:
        raw_vel = (new_centroid - track.centroid) / dt_frames
        speed = float(np.linalg.norm(raw_vel))
        # higher learning rate when moving
        alpha = 0.7 if speed > getattr(self.p, "moving_speed_threshold", 3.0) else 0.3
        track.velocity = alpha * raw_vel + (1 - alpha) * track.velocity
        # clamp
        max_speed = getattr(self.p, "max_expected_speed", 25.0)
        if np.linalg.norm(track.velocity) > max_speed:
            track.velocity = track.velocity / np.linalg.norm(track.velocity) * max_speed


    def _match_active(
        self,
        tracks: List[Track],
        detections: List[BlobDetection],
        det_centroids: List[np.ndarray],
    ) -> List[Tuple[int, int]]:
        """Weighted matching: distance (0.7) + area (0.2) + orientation (0.1)."""
        candidates = []
        for ti, t in enumerate(tracks):
            pred = t.centroid + t.velocity
            for di, (c, det) in enumerate(zip(det_centroids, detections)):
                # distance cost
                dist = float(np.linalg.norm(pred - c))
                if dist > self.p.max_match_distance:
                    continue
                dist_cost = dist / self.p.max_match_distance

                # area cost
                area_err = abs(t.area - self._det_area(det)) / max(t.area, 1.0)
                area_cost = min(area_err, 0.5)

                # orientation cost (if endpoints exist)
                orient_cost = 0.0
                if (
                    t.endpoints is not None
                    and hasattr(det, "endpoints")
                    and det.endpoints is not None # type: ignore
                ):
                    t_dir = t.endpoints[1] - t.endpoints[0]
                    d_dir = det.endpoints[1] - det.endpoints[0] # type: ignore
                    if np.linalg.norm(t_dir) > 0 and np.linalg.norm(d_dir) > 0:
                        cos_sim = abs(
                            np.dot(t_dir, d_dir)
                            / (np.linalg.norm(t_dir) * np.linalg.norm(d_dir))
                        )
                        orient_cost = 1.0 - cos_sim

                total_cost = 0.7 * dist_cost + 0.2 * area_cost + 0.1 * orient_cost
                if total_cost < 1.0:
                    candidates.append((total_cost, ti, di))

        candidates.sort(key=lambda x: x[0])
        used_t, used_d = set(), set()
        pairs = []
        for _, ti, di in candidates:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            pairs.append((ti, di))
        return pairs

    def _match_tentative(
        self, det_centroids: List[np.ndarray]
    ) -> List[Tuple[int, int]]:
        """Simplified greedy matching for tentative tracks."""
        candidates = []
        for ti, t in enumerate(self.tentative_tracks):
            pred = t.centroid + t.velocity
            for di, c in enumerate(det_centroids):
                d = float(np.linalg.norm(pred - c))
                if d <= self.p.tentative_match_distance:
                    candidates.append((d, ti, di))
        candidates.sort(key=lambda x: x[0])
        used_t, used_d = set(), set()
        pairs = []
        for _, ti, di in candidates:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            pairs.append((ti, di))
        return pairs


    def _reid_thresholds(self, track: Track) -> Tuple[int, float]:
        wall_code = getattr(self.p, "wall_region_code", 2)
        if track.region == wall_code:
            return (
                getattr(self.p, "wall_reid_grace_frames", 400),
                getattr(self.p, "wall_reid_max_distance", 80.0),
            )
        return (
            getattr(self.p, "reid_grace_frames", 200),
            getattr(self.p, "reid_max_distance", 120.0),
        )

    def _best_lost_match(
        self,
        det_centroid: np.ndarray,
        det_area: float,
        det: Optional[BlobDetection] = None,
    ) -> Optional[int]:
        """Pick best lost track using distance + area + appearance."""
        best_i = None
        best_score = float("inf")
        area_w = getattr(self.p, "reid_area_weight", 0.2)
        appear_w = getattr(self.p, "reid_appearance_weight", 0.3)

        det_sig = self._compute_appearance_signature(det) if det is not None else None

        for i, t in enumerate(self.lost_tracks):
            dt = self.frame_idx - t.last_seen
            grace, maxd = self._reid_thresholds(t)
            if dt > grace:
                continue

            pred = t.centroid + t.velocity * dt
            dist = float(np.linalg.norm(pred - det_centroid))
            if dist > maxd:
                continue

            dist_score = dist / maxd
            area_err = abs(t.area - det_area) / max(t.area, 1.0)

            appear_score = 0.0
            if det_sig is not None and t.appearance_signature is not None:
                appear_score = np.linalg.norm(det_sig - t.appearance_signature) / 255.0

            total_score = dist_score + area_w * area_err + appear_w * appear_score
            if total_score < best_score:
                best_score = total_score
                best_i = i
        return best_i


    def _bbox_iou(
        self, b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]
    ) -> float:
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

    def _suppress_birth_near_active(
        self, det_centroid: np.ndarray, det_bbox: Tuple[int, int, int, int]
    ) -> bool:
        for t in self.active_tracks:
            if (
                np.linalg.norm(t.centroid - det_centroid)
                <= self.p.birth_suppression_dist
            ):
                return True
            tb = getattr(t, "bbox", None)
            if (
                tb is not None
                and self._bbox_iou(tb, det_bbox) >= self.p.birth_suppression_iou
            ):
                return True
        return False

    def update(self, detections: List[BlobDetection]) -> List[Track]:
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

        confirmed_ids: Set[int] = set()

        pairs = self._match_active(self.active_tracks, dets, det_centroids)
        matched_tracks = {ti for ti, _ in pairs}
        matched_dets = {di for _, di in pairs}
        for di in matched_dets:
            confirmed_ids.add(id(dets[di]))

        new_active: List[Track] = []
        for ti, di in pairs:
            t = self.active_tracks[ti]
            new_c = det_centroids[di]
            self._update_velocity_smart(t, new_c)
            t.centroid = new_c
            t.area = det_areas[di]
            t.region = det_regions[di]
            t.age += 1
            t.missed = 0
            t.last_seen = self.frame_idx
            t.bbox = tuple(map(int, dets[di].bbox)) # type: ignore
            self._apply_geometry_and_headtail(t, dets[di])
            t.appearance_signature = self._compute_appearance_signature(dets[di])
            # update motion memory
            if t.id not in self.motion_memories:
                self.motion_memories[t.id] = MotionMemory()
            self.motion_memories[t.id].update(dets[di].mask.shape, t.centroid)
            new_active.append(t)

        for i, t in enumerate(self.active_tracks):
            if i in matched_tracks:
                continue
            t.missed += 1
            t.velocity[:] = 0.0
            t.active = False
            self.lost_tracks.append(t)

        self.active_tracks = new_active

        # Prepare remaining detections
        remaining_det_indices = [
            di for di in range(len(dets)) if di not in matched_dets
        ]
        rem_centroids = [det_centroids[di] for di in remaining_det_indices]
        local_to_global = {
            local: global_i for local, global_i in enumerate(remaining_det_indices)
        }

        tent_pairs = self._match_tentative(rem_centroids)
        matched_tent = set()
        matched_local = set()
        for tent_i, local_di in tent_pairs:
            global_di = local_to_global[local_di]
            det = dets[global_di]
            tt = self.tentative_tracks[tent_i]
            new_c = det_centroids[global_di]
            tt.velocity = new_c - tt.centroid
            tt.centroid = new_c
            tt.area = det_areas[global_di]
            tt.region = det_regions[global_di]
            tt.confirm_count += 1
            tt.missed = 0
            tt.last_seen = self.frame_idx
            tt.last_det = det
            matched_tent.add(tent_i)
            matched_local.add(local_di)

        new_tentative = []
        for i, tt in enumerate(self.tentative_tracks):
            if i in matched_tent:
                new_tentative.append(tt)
            else:
                tt.missed += 1
                if tt.missed <= self.p.tentative_max_missed:
                    new_tentative.append(tt)
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

        promoted = []
        remaining_tent = []
        for tt in self.tentative_tracks:
            if (
                tt.confirm_count >= self.p.min_confirm_frames
                and tt.last_det is not None
            ):
                new_track = self._new_track(
                    tt.centroid, tt.area, tt.region, tt.last_det.bbox, tt.last_det
                )
                promoted.append(new_track)
            else:
                remaining_tent.append(tt)
        self.tentative_tracks = remaining_tent
        self.active_tracks.extend(promoted)

        for di, det in enumerate(dets):
            if di in matched_dets:
                continue
            c = det_centroids[di]
            a = det_areas[di]
            r = det_regions[di]
            li = self._best_lost_match(c, a, det)
            if li is not None:
                confirmed_ids.add(id(det))
                t = self.lost_tracks.pop(li)
                t.active = True
                t.missed = 0
                self._update_velocity_smart(t, c)
                t.centroid = c
                t.area = a
                t.region = r
                t.age += 1
                t.last_seen = self.frame_idx
                t.bbox = tuple(map(int, det.bbox)) # type: ignore
                self._apply_geometry_and_headtail(t, det)
                t.appearance_signature = self._compute_appearance_signature(det)
                self.active_tracks.append(t)
            else:
                # Birth new track only if not suppressed
                if self._suppress_birth_near_active(c, det.bbox):
                    continue
                confirmed_ids.add(id(det))
                new_track = self._new_track(c, a, r, det.bbox, det)
                self.active_tracks.append(new_track)

        self._prune_lost()

        if getattr(self.p, "require_motion", True):
            self.active_tracks = [
                t
                for t in self.active_tracks
                if t.id in self.motion_memories
                and self.motion_memories[t.id].update(dets[di].mask.shape, t.centroid)
            ]


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

    def _prune_lost(self) -> None:
        keep = []
        for t in self.lost_tracks:
            dt = self.frame_idx - t.last_seen
            grace, _ = self._reid_thresholds(t)
            if dt <= grace and dt <= self.p.max_missed_frames:
                keep.append(t)
        self.lost_tracks = keep



@dataclass
class TrackSegment:
    """A continuous segment of a worm's trajectory."""

    id: int
    frames: List[int]
    centroids: List[np.ndarray]
    areas: List[float]
    regions: List[int]
    velocities: List[np.ndarray]
    endpoints_list: List[Optional[np.ndarray]]
    skeleton_lengths: List[float]

    @property
    def start_frame(self) -> int:
        return self.frames[0]

    @property
    def end_frame(self) -> int:
        return self.frames[-1]

    @property
    def duration(self) -> int:
        return len(self.frames)

    @property
    def mean_centroid(self) -> np.ndarray:
        return np.mean(self.centroids, axis=0)

    @property
    def mean_area(self) -> float:
        return np.mean(self.areas) # type: ignore

    def get_appearance_signature(self) -> Optional[np.ndarray]:
        if len(self.areas) < 5:
            return None
        area_std = np.std(self.areas) / max(self.mean_area, 1)
        area_trend = (self.areas[-1] - self.areas[0]) / max(self.mean_area, 1)
        length_ratios = []
        for sk_len in self.skeleton_lengths:
            if sk_len > 0 and self.mean_area > 0:
                width_est = self.mean_area / max(sk_len, 1)
                length_ratios.append(sk_len / max(width_est, 1))
        mean_lw = np.mean(length_ratios) if length_ratios else 5.0
        return np.array([self.mean_area, area_std, area_trend, mean_lw])


class TrackMerger:
    def __init__(self, params: Any):
        self.p = params

    def segment_tracks_by_gap(self, raw_tracks: List[Track]) -> List[TrackSegment]:
        segments = []
        gap_thresh = getattr(self.p, "max_gap_frames", 30)
        for track in raw_tracks:
            frames = getattr(track, "frame_history", [])
            if not frames:
                # No history – treat as single segment
                seg = TrackSegment(
                    id=track.id,
                    frames=(
                        [self.p.some_frame] if hasattr(self.p, "some_frame") else [0]
                    ),
                    centroids=[track.centroid],
                    areas=[track.area],
                    regions=[track.region],
                    velocities=[track.velocity],
                    endpoints_list=[track.endpoints],
                    skeleton_lengths=[track.skeleton_length],
                )
                segments.append(seg)
                continue

            # Split on large gaps
            curr = {
                "frames": [],
                "centroids": [],
                "areas": [],
                "regions": [],
                "velocities": [],
                "endpoints": [],
                "skel_lengths": [],
            }
            for i, f in enumerate(frames):
                if i > 0 and f - frames[i - 1] > gap_thresh:
                    if len(curr["frames"]) >= getattr(self.p, "min_segment_length", 5):
                        segments.append(
                            TrackSegment(
                                id=track.id,
                                frames=curr["frames"],
                                centroids=curr["centroids"],
                                areas=curr["areas"],
                                regions=curr["regions"],
                                velocities=curr["velocities"],
                                endpoints_list=curr["endpoints"],
                                skeleton_lengths=curr["skel_lengths"],
                            )
                        )
                    curr = {
                        "frames": [],
                        "centroids": [],
                        "areas": [],
                        "regions": [],
                        "velocities": [],
                        "endpoints": [],
                        "skel_lengths": [],
                    }
                curr["frames"].append(f)
                curr["centroids"].append(
                    track.centroid_history[i]
                    if hasattr(track, "centroid_history")
                    else track.centroid
                )
                curr["areas"].append(track.area)
                curr["regions"].append(track.region)
                curr["velocities"].append(track.velocity)
                curr["endpoints"].append(track.endpoints)
                curr["skel_lengths"].append(track.skeleton_length)
            if len(curr["frames"]) >= getattr(self.p, "min_segment_length", 5):
                segments.append(TrackSegment(**curr, id=track.id))
        return segments

    def compute_segment_similarity(
        self, seg1: TrackSegment, seg2: TrackSegment
    ) -> float:
        if seg2.start_frame <= seg1.end_frame:
            return float("inf")
        time_gap = seg2.start_frame - seg1.end_frame
        # Extrapolate from seg1 end
        if len(seg1.centroids) >= 2 and len(seg1.velocities) > 0:
            v_last = seg1.velocities[-1]
            predicted = seg1.centroids[-1] + v_last * time_gap
            dist = np.linalg.norm(predicted - seg2.centroids[0])
        else:
            dist = np.linalg.norm(seg1.centroids[-1] - seg2.centroids[0])
        max_move = getattr(self.p, "max_reid_movement", 100.0)
        dist_score = min(dist / max_move, 1.0)

        area_diff = abs(seg1.mean_area - seg2.mean_area) / max(seg1.mean_area, 1.0)
        area_score = min(area_diff, 0.5)

        app1 = seg1.get_appearance_signature()
        app2 = seg2.get_appearance_signature()
        app_score = 0.0
        if app1 is not None and app2 is not None:
            app_score = min(np.linalg.norm(app1 - app2) / 10.0, 0.5)

        region_penalty = 0.0
        if seg1.regions and seg2.regions and seg1.regions[-1] != seg2.regions[0]:
            region_penalty = 0.2

        return (
            0.5 * dist_score + 0.2 * area_score + 0.2 * app_score + 0.1 * region_penalty
        ) # type: ignore

    def greedy_link_segments(
        self, segments: List[TrackSegment], cost_matrix: np.ndarray
    ) -> List[List[int]]:
        n = len(segments)
        used_target = set()
        used_source = set()
        chains = defaultdict(list)
        sorted_idx = sorted(range(n), key=lambda i: segments[i].start_frame)
        for i in sorted_idx:
            if i in used_source:
                continue
            chain = [i]
            used_source.add(i)
            cur = i
            while True:
                best_j = None
                best_cost = float("inf")
                for j in range(n):
                    if j in used_target or j == cur:
                        continue
                    if segments[j].start_frame <= segments[cur].end_frame:
                        continue
                    cost = cost_matrix[cur, j]
                    if cost < best_cost and cost < self.p.merge_threshold:
                        best_cost = cost
                        best_j = j
                if best_j is None:
                    break
                chain.append(best_j)
                used_target.add(best_j)
                used_source.add(best_j)
                cur = best_j
            chains[len(chains)] = chain
        # Isolated segments
        for i in range(n):
            if i not in used_source and i not in used_target:
                chains[len(chains)] = [i]
        return list(chains.values())

    def merge_segments(
        self, segments: List[TrackSegment], chains: List[List[int]]
    ) -> List[TrackSegment]:
        merged = []
        for chain in chains:
            if len(chain) == 1:
                merged.append(segments[chain[0]])
                continue
            chain_segs = [segments[i] for i in chain]
            chain_segs.sort(key=lambda s: s.start_frame)
            all_frames, all_centroids, all_areas, all_regions = [], [], [], []
            all_velocities, all_endpoints, all_skel = [], [], []
            for seg in chain_segs:
                all_frames.extend(seg.frames)
                all_centroids.extend(seg.centroids)
                all_areas.extend(seg.areas)
                all_regions.extend(seg.regions)
                all_velocities.extend(seg.velocities)
                all_endpoints.extend(seg.endpoints_list)
                all_skel.extend(seg.skeleton_lengths)
            merged.append(
                TrackSegment(
                    id=chain_segs[0].id,
                    frames=all_frames,
                    centroids=all_centroids,
                    areas=all_areas,
                    regions=all_regions,
                    velocities=all_velocities,
                    endpoints_list=all_endpoints,
                    skeleton_lengths=all_skel,
                )
            )
        return merged

    def interpolate_gaps(self, track: TrackSegment) -> TrackSegment:
        if len(track.frames) < 2:
            return track
        new_frames = []
        new_centroids = []
        for i in range(len(track.frames) - 1):
            new_frames.append(track.frames[i])
            new_centroids.append(track.centroids[i])
            gap = track.frames[i + 1] - track.frames[i]
            if gap > 1:
                for t in range(1, gap):
                    alpha = t / gap
                    interp_frame = track.frames[i] + t
                    interp_cent = (1 - alpha) * track.centroids[
                        i
                    ] + alpha * track.centroids[i + 1]
                    new_frames.append(interp_frame)
                    new_centroids.append(interp_cent)
        new_frames.append(track.frames[-1])
        new_centroids.append(track.centroids[-1])
        return TrackSegment(
            id=track.id,
            frames=new_frames,
            centroids=new_centroids,
            areas=track.areas,
            regions=track.regions,
            velocities=track.velocities,
            endpoints_list=track.endpoints_list,
            skeleton_lengths=track.skeleton_lengths,
        )

    def remove_short_tracks(
        self, segments: List[TrackSegment], min_frames: int = 10
    ) -> List[TrackSegment]:
        return [s for s in segments if s.duration >= min_frames]

    def process(
        self, raw_tracks: List[Track], frame_history: Dict[int, List[int]]
    ) -> List[TrackSegment]:
        print(f"Post‑processing: {len(raw_tracks)} raw tracks")
        # Attach history
        for t in raw_tracks:
            t.frame_history = frame_history.get(t.id, [])
        segments = self.segment_tracks_by_gap(raw_tracks)
        print(f"  → Split into {len(segments)} segments")
        segments = self.remove_short_tracks(
            segments, getattr(self.p, "min_segment_frames", 5)
        )
        print(f"  → After removing short: {len(segments)} segments")
        n = len(segments)
        cost_matrix = np.full((n, n), float("inf"))
        for i in range(n):
            for j in range(n):
                if i != j and segments[j].start_frame > segments[i].end_frame:
                    cost_matrix[i, j] = self.compute_segment_similarity(
                        segments[i], segments[j]
                    )
        chains = self.greedy_link_segments(segments, cost_matrix)
        print(f"  → Linked into {len(chains)} chains")
        merged = self.merge_segments(segments, chains)
        print(f"  → After merging: {len(merged)} tracks")
        interpolated = [self.interpolate_gaps(t) for t in merged]
        final = self.remove_short_tracks(
            interpolated, getattr(self.p, "min_track_frames", 20)
        )
        print(f"  → Final tracks: {len(final)}")
        return final



def apply_post_processing(
    tracker_output: List[Dict[str, Any]], params: Any
) -> List[TrackSegment]:
    """Expects tracker_output = list of dicts with keys 'frame', 'tracks'."""
    all_tracks_by_id = defaultdict(
        lambda: {"track": None, "frames": [], "centroids": []}
    )
    for frame_data in tracker_output:
        for track in frame_data["tracks"]:
            tid = track.id
            if all_tracks_by_id[tid]["track"] is None:
                all_tracks_by_id[tid]["track"] = track
            all_tracks_by_id[tid]["frames"].append(frame_data["frame"])
            all_tracks_by_id[tid]["centroids"].append(track.centroid.copy())
    raw_tracks = []
    frame_history = {}
    for tid, data in all_tracks_by_id.items():
        track = data["track"]
        track.frame_history = data["frames"]
        track.centroid_history = data["centroids"]
        raw_tracks.append(track)
        frame_history[tid] = data["frames"]
    merger = TrackMerger(params)
    return merger.process(raw_tracks, frame_history)
