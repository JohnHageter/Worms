# Written by ChatGPT
import time
import numpy as np
from typing import Any, Optional

from Module.io.Camera import Camera
from Module.io.CameraStates import CameraState
from Module.io.Exceptions import CaptureError

class TestCamera(Camera):
    """
    Synthetic camera for UI testing.
    Produces moving gradient frames and supports a watch window.
    """

    def __init__(self, width: int = 640, height: int = 480):
        super().__init__()

        self.width = width
        self.height = height

        self.fps = 30.0
        self.streaming = False

        self._last_frame: Optional[np.ndarray] = None
        self._last_frame_time = 0.0

        self.watch_x = 0
        self.watch_y = 0
        self.watch_w = width
        self.watch_h = height

        self._frame_counter = 0

    def _generate_frame(self) -> np.ndarray:
        """
        Generate a synthetic frame with motion.
        """
        # advance counter safely in uint8 space
        self._frame_counter = (self._frame_counter + 1) & 0xFF
        c = np.uint8(self._frame_counter)

        x = np.arange(self.width, dtype=np.uint8)
        y = np.arange(self.height, dtype=np.uint8)
        xv, yv = np.meshgrid(x, y)

        frame = np.empty((self.height, self.width, 3), dtype=np.uint8)

        frame[..., 0] = xv + c
        frame[..., 1] = yv + (c << 1)      # shift, not multiply
        frame[..., 2] = np.uint8(128)

        return frame

    def _apply_watch_window(self, frame: np.ndarray) -> np.ndarray:
        x0 = self.watch_x
        y0 = self.watch_y
        x1 = min(x0 + self.watch_w, frame.shape[1])
        y1 = min(y0 + self.watch_h, frame.shape[0])

        return frame[y0:y1, x0:x1]

    def _do_open(self) -> None:
        self._last_frame = None
        self._frame_counter = 0

    def _do_close(self) -> None:
        self.streaming = False
        self._last_frame = None

    def _do_set_parameter(self, key: str, value: Any) -> None:
        # Accept anything for testing
        print(f"[TestCamera] Set parameter {key} = {value}")

    def _do_set_framerate(self, fps: float) -> None:
        if fps <= 0:
            raise CaptureError("FPS must be > 0")
        self.fps = fps

    def _do_set_watch_window(self, x_start, width, y_start, height) -> None:
        self.watch_x = int(x_start)
        self.watch_y = int(y_start)
        self.watch_w = int(width)
        self.watch_h = int(height)

    def _do_start_stream(self, max_fps: float = 30) -> None:
        self.fps = min(self.fps, max_fps)
        self.streaming = True
        self._last_frame_time = 0.0

    def _do_stop_stream(self) -> None:
        self.streaming = False

    def _do_grab_frame(self) -> np.ndarray:
        if not self.streaming:
            raise CaptureError("Stream not running")

        now = time.time()
        min_interval = 1.0 / self.fps

        if now - self._last_frame_time < min_interval:
            time.sleep(min_interval - (now - self._last_frame_time))

        frame = self._generate_frame()
        frame = self._apply_watch_window(frame)

        self._last_frame = frame
        self._last_frame_time = time.time()

        return frame

    def _do_grab_last_frame(self) -> Optional[np.ndarray]:
        if self._last_frame is None:
            raise CaptureError("No frame available")
        return self._last_frame
