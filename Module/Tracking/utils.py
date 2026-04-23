import numpy as np
from typing import List, Optional, Tuple, Any
from dataclasses import dataclass, field
from typing import Optional

from Module.Tracking.Detection import BlobDetection
from Module.Tracking.Orientation import HeadTailState
from Module.Tracking.StateManager import StateManager, TrackState


@dataclass
class Track:
    id: int

    centroid: np.ndarray
    velocity: np.ndarray
    predicted: np.ndarray
    area: float
    
    expected_area: float
    expected_bbox: tuple
    last_area: float
    
    interaction_start_frame: int | None
    overlap_start_frame: int | None
    entry_centroid: np.ndarray | None
    entry_velocity: np.ndarray | None
    merged_blob_id: int | None

    age: int
    last_seen: int
    missed: int = 0

    bbox: Optional[Tuple[int, int, int, int]] = None
    endpoints: Optional[np.ndarray] = None
    skeleton: Optional[np.ndarray] = None
    skeleton_length: float = 0.0
    ht_state: HeadTailState = HeadTailState()

    centroid_history: List[np.ndarray] = field(default_factory=list)
    velocity_history: List[np.ndarray] = field(default_factory=list)
    last_detection: Optional[BlobDetection] = None
    overlap_partner_ids: List[int] = field(default_factory=list)

    def __post_init__(self):
        self.state = StateManager(
            initial_state=TrackState.ACTIVE,
            frame_idx=self.last_seen,
        )

        self.centroid_history.append(self.centroid.copy())
        self.velocity_history.append(self.velocity.copy())
        
    def predict(self):
        if self.state.current == TrackState.ACTIVE:
            self.predicted = self.centroid + self.velocity
        else:
            self.predicted = self.centroid.copy()


@dataclass
class TentativeTrack:
    centroid: np.ndarray
    velocity: np.ndarray
    area: float
    bbox: Optional[Tuple[int, int, int, int]]

    seen_frames: int = 1
    missed: int = 0
    last_seen: int = 0

    born_near_track_id: Optional[int] = None
    born_during_overlap: bool = False

    def __post_init__(self):
        self.state = StateManager(
            initial_state=TrackState.TENTATIVE,
            frame_idx=self.last_seen,
        )

def masks_touch(det1, det2) -> bool:
    x1, y1, w1, h1 = det1.bbox
    x2, y2, w2, h2 = det2.bbox

    if x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1:
        return False

    ix1, iy1 = max(x1, x2), max(y1, y2)
    ix2, iy2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)

    if ix1 >= ix2 or iy1 >= iy2:
        return False

    m1 = det1.mask[iy1 - y1 : iy2 - y1, ix1 - x1 : ix2 - x1]
    m2 = det2.mask[iy1 - y2 : iy2 - y2, ix1 - x2 : ix2 - x2]

    return np.any(m1 & m2) # type: ignore

def id_to_color(idx: int) -> tuple[int, int, int]:
    """
    Deterministic vivid color from an integer ID.
    Returns BGR for OpenCV.
    """
    rng = np.random.default_rng(idx)
    color = rng.integers(80, 255, size=3)
    return int(color[0]), int(color[1]), int(color[2])
