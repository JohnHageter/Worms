from abc import ABC, abstractmethod
from typing import Optional, Any, Tuple
import numpy as np
import time


class Camera(ABC):

    def __init__(self):
        self.cam = None
        self.watch_window = None #x_off, width, y_off, height
        self._is_open: bool = False

    def open(self) -> bool:
        """
        Open the camera device.
        """
        if self._is_open:
            return True

        try:
            self._open()
            self._is_open = True
            return True
        except Exception as e:
            self._is_open = False
            raise CameraError("Failed to open camera.") from e

    def close(self) -> None:
        """
        Close and release the camera.
        """
        if not self._is_open:
            return

        try:
            self._close()
        finally:
            self._is_open = False
            self.cam = None

    def watch(self, x_off: int, width: int, y_off: int, height: int):
        if width <= 0 or height <= 0:
            return False

        if x_off < 0 or y_off < 0:
            return False

        try:
            self._watch(x_off, width, y_off, height)
        except Exception:
            return False

    def set(self, parameter: str, value: Any) -> bool:
        """
        Set a camera parameter (exposure, gain, fps, etc).
        """
        if not self._is_open:
            raise InvalidStateError("Camera must be opened before setting parameters.")

        try:
            return bool(self._set(parameter, value))
        except Exception as e:
            raise CameraError(f"Failed to set parameter '{parameter}'") from e

    def read(self) -> Tuple[bool, np.ndarray, float]:
        """
        Read a frame from the camera.

        Returns:
            success: bool
            frame: ndarray or None
            timestamp: perf_counter timestamp
        """
        if not self._is_open:
            raise InvalidStateError("Camera must be opened before reading.")

        timestamp = time.perf_counter()

        try:
            success, frame = self._read_frame()
            return success, frame, timestamp
        except Exception as e:
            raise CaptureError("Failed to read frame.") from e

    @abstractmethod
    def _open(self) -> None:
        """
        Backend-specific open.
        Must raise on fatal failure.
        """
        ...

    @abstractmethod
    def _close(self) -> None:
        """
        Backend-specific close.
        """
        ...

    @abstractmethod
    def _set(self, parameter: str, value: Any) -> bool:
        """
        Backend-specific parameter setter.
        Return False if unsupported or rejected.
        """
        ...

    @abstractmethod
    def _watch(self, x_off: int, width: int, y_off: int, height: int) -> bool:
        """
        Set an ROI within the camera.
        """
        ...

    @abstractmethod
    def _read_frame(self) -> Tuple[bool, np.ndarray]:
        """
        Backend-specific frame grab.
        Must NOT raise on normal capture failure.
        """
        ...


class CameraError(Exception):
    """Base class for all camera-related errors."""


class InvalidStateError(CameraError):
    """Invalid operation for the current camera state."""


class CaptureError(CameraError):
    """Frame capture failed."""
