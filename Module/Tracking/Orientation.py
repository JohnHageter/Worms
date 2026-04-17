from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from Module.Tracking.Configuration import TrackingParameters

@dataclass
class HeadTailState:
    head: Optional[np.ndarray] = None  # (2,)
    tail: Optional[np.ndarray] = None  # (2,)
    confidence: float = 0.0
    stable_frames: int = 0  # number of consecutive confident assignments

def assign_head_tail(
    params: TrackingParameters,
    centroid: np.ndarray,
    velocity: np.ndarray,
    endpoints_xy: np.ndarray,  # shape (2,2) in full-frame coords
    prev_state: HeadTailState,
    shape_hint: Optional[Tuple[float, float]] = None,  # (score_end0, score_end1) optional
) -> HeadTailState:
    """
    Returns updated head/tail state given endpoints and track motion.
    shape_hint optional: higher score => more likely head (species analyzer can provide).
    """

    p0 = endpoints_xy[0].astype(float)
    p1 = endpoints_xy[1].astype(float)

    # initialize if empty
    if prev_state.head is None or prev_state.tail is None:
        st = HeadTailState(head=p0.copy(), tail=p1.copy(), confidence=0.0, stable_frames=0)
        return st

    motion = velocity.astype(float)
    speed = float(np.linalg.norm(motion))

    # Candidate assignment by motion
    motion_head = None
    motion_tail = None
    motion_update = 0.0

    if speed >= params.min_speed_for_headtail:
        d0 = float(np.dot(p0 - centroid, motion))
        d1 = float(np.dot(p1 - centroid, motion))
        if d0 >= d1:
            motion_head, motion_tail = p0, p1
        else:
            motion_head, motion_tail = p1, p0

        # compare to previous head position (identity consistency)
        if np.linalg.norm(motion_head - prev_state.head) <= np.linalg.norm(motion_tail - prev_state.head):
            motion_update = params.headtail_gain_motion
        else:
            motion_update = -params.headtail_penalty_motion

    # Candidate assignment by persistence (distance to previous head/tail)
    # Choose ordering that best matches previous (min total distance)
    d_same = np.linalg.norm(p0 - prev_state.head) + np.linalg.norm(p1 - prev_state.tail)
    d_swap = np.linalg.norm(p1 - prev_state.head) + np.linalg.norm(p0 - prev_state.tail)
    if d_same <= d_swap:
        persist_head, persist_tail = p0, p1
        persist_update = params.headtail_gain_persist
    else:
        persist_head, persist_tail = p1, p0
        persist_update = -params.headtail_gain_persist

    # Optional shape hint
    shape_update = 0.0
    shape_head, shape_tail = persist_head, persist_tail
    if shape_hint is not None:
        s0, s1 = shape_hint
        if s0 >= s1:
            shape_head, shape_tail = p0, p1
        else:
            shape_head, shape_tail = p1, p0

        # reinforce if shape agrees with persistence head
        if np.allclose(shape_head, persist_head, atol=1e-3):
            shape_update = params.headtail_gain_shape
        else:
            shape_update = -params.headtail_penalty_shape

    # Combine evidence: persistence is always available; motion contributes when speed high
    new_conf = prev_state.confidence + persist_update + motion_update + shape_update

    # Choose final assignment:
    # - if motion is available and confidence is positive, trust motion ordering
    # - otherwise trust persistence ordering
    if motion_head is not None and new_conf > 0:
        head, tail = motion_head, motion_tail
    else:
        head, tail = persist_head, persist_tail

    # Flip only if sustained contradiction
    if new_conf < params.headtail_flip_threshold:
        head, tail = tail, head
        new_conf = 1.0  # reset after flip
        stable = 0
    else:
        stable = prev_state.stable_frames + (1 if new_conf > 0 else 0)

    return HeadTailState(head=head.copy(), tail=tail.copy(), confidence=float(new_conf), stable_frames=int(stable)) # type: ignore
