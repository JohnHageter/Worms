from __future__ import annotations
from dataclasses import dataclass, asdict
import json


@dataclass
class TrackingParameters:
    # ------------------------------------------------------------------
    # Detection & Filtering
    # ------------------------------------------------------------------
    max_detectable_blobs: int = 50
    """Maximum number of blobs (worms) to keep per frame.
      Sorted by area, only the largest N are passed to the tracker.
      For a crowded Petri dish (e.g., 50‑100 worms), increase to 100‑150."""

    min_blob_area: int = 20
    """Minimum area (pixels²) of a valid blob.
      Discards noise, debris, or very small fragments.
      At 2 fps, worms may be motion‑blurred → lower to 15 if small worms disappear."""

    # ------------------------------------------------------------------
    # Skeleton & Geometry
    # ------------------------------------------------------------------
    min_skeleton_length: float = 50.0
    """Minimum skeleton path length (pixels) to trust endpoints for head/tail.
      Below this value, the worm is considered too small/curled and uses centroid‑only.
      For tiny larvae, reduce to 20‑30; for large adults, keep 50+."""

    # ------------------------------------------------------------------
    # Track Lifecycle
    # ------------------------------------------------------------------
    max_missed_frames: int = 30
    """Frames a track can be missing before being permanently deleted.
      At 2 fps, 30 frames = 15 seconds. For wall‑crawlers that disappear longer,
      rely on `reid_grace_frames` and `wall_reid_grace_frames` instead."""

    # ------------------------------------------------------------------
    # Association (Matching detections to tracks)
    # ------------------------------------------------------------------
    max_match_distance: float = 60.0
    """Maximum Euclidean distance (pixels) to consider a detection as a match
      to an existing track's predicted position. At 2 fps, worms can move
      20‑40 px between frames → increase to 80‑100 for fast species."""

    # ------------------------------------------------------------------
    # Head/Tail Assignment (confidence‑based)
    # ------------------------------------------------------------------
    min_speed_for_headtail: float = 3.0
    """Minimum speed (px/frame) required to use motion direction as evidence
      for head/tail assignment. Below this speed, only shape cues are used."""

    headtail_flip_threshold: float = -3.0
    """If head/tail confidence falls below this value, flip the identity.
      Negative values make flipping harder; adjust to make assignment stable."""

    headtail_gain_motion: float = 0.6
    """Confidence increase when motion direction agrees with current head/tail.
      Higher = more trust in motion cues (good for crawling worms)."""

    headtail_penalty_motion: float = 0.8
    """Confidence decrease when motion direction disagrees.
      Should be > gain to quickly correct wrong assignments."""

    headtail_gain_shape: float = 0.15
    """Weak confidence increase when shape (endpoint curvature) agrees.
      Useful when worm is stationary or turning."""

    headtail_penalty_shape: float = 0.15
    """Weak confidence decrease when shape disagrees.
      Keeps symmetric worms from oscillating."""

    headtail_gain_persist: float = 0.1
    """Small confidence boost when endpoints are consistent with the previous
      frame's head/tail assignment. Helps maintain identity during slow movement."""

    # ------------------------------------------------------------------
    # Re‑identification (lost tracks)
    # ------------------------------------------------------------------
    reid_grace_frames: int = 200
    """How many frames a lost track is kept in memory for possible re‑linking.
      At 2 fps, 200 frames = 100 seconds. Increase to 500‑1000 for week‑long
      recordings where worms may hide under debris or crawl on walls."""

    reid_max_distance: float = 120.0
    """Maximum distance (pixels) from predicted position to re‑link a lost track.
      For 2 fps, worms can travel 40‑80 px during absence; increase if they
      reappear far away (e.g., 150‑200)."""

    # ------------------------------------------------------------------
    # Wall Excursion Handling (worms on Petri dish wall)
    # ------------------------------------------------------------------
    wall_region_code: int = 2
    """Region code indicating the wall area (used by region_mask)."""

    wall_reid_grace_frames: int = 600
    """Longer grace period specifically for tracks lost on the wall.
      600 frames = 300 seconds (5 minutes). Worms often crawl along the wall
      and may disappear from view for several minutes."""

    wall_reid_max_distance: float = 200.0
    """Larger search radius for re‑linking wall tracks, because worms can
      travel further along the wall without being detected."""

    reid_area_weight: float = 50.0
    """Weight multiplier for area difference when scoring lost track candidates.
      Higher values penalize area mismatches more (helps avoid linking
      a large worm to a small debris fragment)."""

    # ------------------------------------------------------------------
    # Birth Suppression (avoid creating new IDs near existing tracks)
    # ------------------------------------------------------------------
    birth_suppression_dist: float = 80.0
    """If a new detection's centroid is within this distance (pixels) of an
      existing active track's centroid, do NOT spawn a new track.
      Prevents ID fragmentation when worms touch or overlap."""

    birth_suppression_iou: float = 0.2
    """If the Intersection over Union (IoU) between a new detection's bounding
      box and an existing track's bbox exceeds this value, suppress birth.
      Useful when two worms are partially overlapping."""

    # ------------------------------------------------------------------
    # Tentative Tracks (short‑term memory before confirmation)
    # ------------------------------------------------------------------
    min_confirm_frames: int = 3
    """Number of consecutive matches required to promote a tentative track
      to a confirmed active track. At 2 fps, 3 frames = 1.5 seconds."""

    tentative_max_missed: int = 1
    """Number of frames a tentative track can be missed before being discarded.
      Tentative tracks are very short‑lived; set to 1‑2."""

    tentative_match_distance: float = 80.0
    """Gating distance (pixels) for matching detections to tentative tracks.
      Usually larger than `max_match_distance` because tentative tracks have
      less reliable velocity prediction."""

    require_motion: bool = True  # suppress ghost tracks
    # ------------------------------------------------------------------
    # Debug & Visualization
    # ------------------------------------------------------------------
    debug: bool = False
    """If True, the tracker will store internal debug info (detection status,
      counts) which can be drawn on the output frames. May slow performance."""

    def to_json(self) -> str:
        """Export parameters to a pretty JSON string."""
        return json.dumps(asdict(self), indent=2, sort_keys=True)
