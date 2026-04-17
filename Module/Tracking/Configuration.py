from __future__ import annotations
from dataclasses import dataclass, asdict
import json


@dataclass
class TrackingParameters:
    # Detection/Filtering
    max_detectable_blobs: int = 50  # keep largest N by area
    min_blob_area: int = 20  # drop blobs smaller than this (pixels)

    # Skeleton gating
    min_skeleton_length: float = (
        50.0  # below => centroid-only (pixels along skeleton path)
    )

    # Track lifecycle
    max_missed_frames: int = 30  # kill track after this many missed frames

    # Association
    max_match_distance: float = (
        60.0  # gating distance (pixels) for matching detections to tracks
    )

    # Head/Tail confidence
    min_speed_for_headtail: float = (
        3.0  # px/frame needed to use motion direction evidence
    )
    headtail_flip_threshold: float = -3.0  # flip identity if confidence goes below this
    headtail_gain_motion: float = 0.6  # confidence gain when motion agrees
    headtail_penalty_motion: float = 0.8  # confidence penalty when motion disagrees
    headtail_gain_shape: float = 0.15  # weak gain when shape agrees
    headtail_penalty_shape: float = 0.15  # weak penalty when shape disagrees
    headtail_gain_persist: float = (
        0.1  # gain when endpoints consistent with previous head/tail
    )

    # Reappearance handling
    reid_grace_frames: int = (
        200  # keep "inactive" tracks this long for possible re-linking
    )
    reid_max_distance: float = 120.0  # distance gating to reconnect returning objects

    # Worm wall excursion handling
    wall_region_code: int = 2  # region code used for wall
    wall_reid_grace_frames: int = 600  # how long to keep a "wall lost" track
    wall_reid_max_distance: float = 200.0
    reid_area_weight: float = 50.0

    birth_suppression_dist: float = 80.0
    birth_suppression_iou: float = 0.2

    min_confirm_frames: int = 3           # require this many consecutive matches
    tentative_max_missed: int = 1         # tentative dies quickly if it disappears
    tentative_match_distance: float = 80.0  # gating distance for tentative matching
    
    debug: bool = False


    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)
