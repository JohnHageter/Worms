from enum import Enum, auto


class CameraState(Enum):
    CLOSED = auto()
    OPEN = auto()
    STREAMING = auto()
