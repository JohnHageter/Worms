from abc import ABC, abstractmethod
from typing import Optional, Any
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
    
    def _require_state(self, *allowed: CameraState) -> None:
        if self._state not in allowed:
            raise InvalidStateError(f"Operation not allowed in state {self._state.name}")

    def open(self) -> None:
        self._require_state(CameraState.CLOSED)
        self._do_open()
        self._state = CameraState.OPEN
        
    def close(self) -> None:
        self._require_state(CameraState.OPEN)
        self._do_close()
        self._state = CameraState.CLOSED
        pass

    def set_parameter(self, key, value) -> None:
        self._require_state(CameraState.OPEN)
        self._do_set_parameter(key, value)

    def set_framerate(self, fps: float) -> None:
        self._require_state(CameraState.OPEN)
        self._do_set_framerate(fps)


    def set_watch_window(self, x_start, width, y_start, height) -> None:
        self._require_state(CameraState.OPEN)
        self._do_set_watch_window(x_start, width, y_start, height)

    def grab_frame(self) -> Optional[np.ndarray]:
        self._require_state(CameraState.STREAMING)
        return self._do_grab_frame()


    def start_stream(self, max_fps: float = 30) -> None:
        self._require_state(CameraState.OPEN)
        self._do_start_stream(max_fps)
        self._state = CameraState.STREAMING

    def stop_stream(self) -> None:
        self._require_state(CameraState.STREAMING)
        self._do_stop_stream()
        self._state = CameraState.OPEN

    def grab_last_frame(self) -> Optional[np.ndarray]:
        try:
            self._require_state(CameraState.OPEN, CameraState.STREAMING)
            return self._do_grab_last_frame()
        except CaptureError as c:
            print(str(c))
            return None


    @abstractmethod
    def _do_open(self) -> None:
        pass

    @abstractmethod
    def _do_close(self) -> None:
        pass
    
    @abstractmethod
    def _do_set_parameter(self, key: str = "none", value: Any = None) -> None:
        pass
    
    @abstractmethod
    def _do_set_framerate(self, fps: float) -> None:
        pass
    
    @abstractmethod
    def _do_grab_frame(self) -> np.ndarray:
        pass
    
    @abstractmethod
    def _do_set_watch_window(self, x_start, width, y_start, height) -> None:
        pass
    
    @abstractmethod
    def _do_start_stream(self, max_fps: float = 30) -> None:
        pass
    
    @abstractmethod
    def _do_stop_stream(self) -> None:
        pass
    
    @abstractmethod
    def _do_grab_last_frame(self) -> np.ndarray:
        pass