from dataclasses import dataclass
from typing import Optional
from enum import Enum, auto


class TrackState(Enum):
    TENTATIVE = auto()
    ACTIVE = auto()
    INTERACTING = auto()
    OVERLAPPING = auto()
    LOST = auto()
    TERMINATED = auto()


@dataclass
class StateTransition:
    frame_idx: int
    previous_state: TrackState
    new_state: TrackState
    reason: str
    details: Optional[str] = None


class StateManager:
    def __init__(self, initial_state: TrackState, frame_idx: int):
        self.current = initial_state
        self.history: list[StateTransition] = [
            StateTransition(
                frame_idx=frame_idx,
                previous_state=initial_state,
                new_state=initial_state,
                reason="initialized",
            )
        ]

    def transition(
        self,
        *,
        new_state: TrackState,
        frame_idx: int,
        reason: str,
        details: Optional[str] = None,
    ):
        if new_state == self.current:
            return

        transition = StateTransition(
            frame_idx=frame_idx,
            previous_state=self.current,
            new_state=new_state,
            reason=reason,
            details=details,
        )
        self.history.append(transition)
        self.current = new_state
