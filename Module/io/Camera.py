from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

from Module.io.CameraStates import CameraState
from Module.io.Exceptions import CaptureError, InvalidStateError

class Camera(ABC):
    """Basic camera object"""
    def __init__(self):
        self._state=CameraState.CLOSED

    @property
    def state(self) -> CameraState:
        return self._state
    
    def _require_state(self, *allowed: CameraState, operation: str = "") -> None:
        if self._state not in allowed:
            raise InvalidStateError(f"Operation not allowd in state {self._state.name}")

    def open(self) -> None:
        self._require_state(CameraState.CLOSED)
        self._do_open()
        self._state = CameraState.OPEN
        
    def close(self) -> None:
        self._require_state(CameraState.OPEN)
        self._do_close()
        self._state = CameraState.CLOSED
        pass

    def set_parameter(self, key="none", value=None) -> None:
        self._require_state(CameraState.OPEN)
        self._do_set_parameter()

    def set_framerate(self, fps: float) -> None:
        self._require_state(CameraState.OPEN)
        self._do_set_framerate()


    def set_watch_window(self, x_start, width, y_start, height) -> None:
        self._require_state(CameraState.OPEN)
        self._do_set_watch_window(x_start, width, y_start, height)

    def grab_frame(self) -> Optional[np.ndarray]:
        self._require_state(CameraState.STREAMING)
        self._do_grab_frame()


    def start_stream(self, fps: float) -> None:
        self._require_state(CameraState.OPEN)
        self._do_start_stream()
        self._state = CameraState.STREAMING

    def stop_stream(self) -> None:
        self._require_state(CameraState.STREAMING)
        self._do_stop_stream()
        self._state = CameraState.OPEN

    def grab_last_frame(self) -> Optional[np.ndarray]:
        try:
            self._require_state(CameraState.OPEN, CameraState.STREAMING)
            self._do_grab_last_frame()
        except CaptureError as c:


    @abstractmethod
    def _do_open(self) -> None:
        pass

    @abstractmethod
    def _do_close(self) -> None:
        pass