from __future__ import annotations
from dataclasses import dataclass, asdict
import json


@dataclass
class TrackingParameters:

    max_detectable_blobs: int = 50
    """
    Maximum number of blobs that can be detected in a single image. exceed expected number
    """
    
    min_blob_area: int = 20
    """
    Minimum area in pixels for a blob to be tracked.
    """
    
    min_length_for_skeleton: int = 20
    """
    Minimum length for skeleton to be applied to blob.
    """
    
    
    min_speed_for_headtail: int = 3
    """
    Minimum speed (px/frame) to establish head/tail assignment.
    """
    
    headtail_gain_motion: float = 0.2
    headtail_penalty_motion: float = 0.5
    headtail_gain_persist: float = 0.1
    headtail_gain_shape: float = 0.1
    headtail_penalty_shape: float = 0.1
    headtail_flip_threshold: float = 0.4
    """
    Floating confidence adjustments for head/tail assignment retention.
    """
    


    max_match_distance: float = 80.0
    """
    Maximum distance (pixels) between a detection and a track's *predicted*
    position to consider a match.
    """

    motion_angle_weight: float = 30.0
    """
    Weight applied to angular deviation from predicted motion direction.
    Higher values strongly discourage sideways/backwards assignments.
    """

    max_missed_frames: int = 30
    """
    Number of consecutive frames a confirmed track may be unmatched
    before transitioning to TERMINATED.
    """

    tentative_confirm_frames: int = 3
    """
    Number of consecutive matches required to promote a TentativeTrack
    to a confirmed Track.
    """

    tentative_max_missed: int = 1
    """
    Number of frames a TentativeTrack may be missed before being discarded.
    """

    tentative_match_distance: float = 80.0
    """
    Gating distance used when associating detections to TentativeTracks.
    Generally larger than max_match_distance because velocity is unreliable.
    """

    overlap_distance: float = 40.0
    """
    Predicted inter-track distance below which two tracks are considered
    to be entering an overlap.
    """

    min_speed_for_overlap: float = 1.5
    """
    Minimum relative speed required to consider an approaching pair
    as a true overlap (filters stationary clutter).
    """

    max_overlap_frames: int = 30
    """
    Maximum duration (frames) an overlap may persist before forced resolution.
    """
    
    interaction_distance: float = overlap_distance * 1.5
    """
    Distance between blobs to consider interaction interaction state. Predecessor to Overlapping state
    """
    

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)
