import cv2
import threading
import time
from typing import Optional
import numpy as np
from Module.io.Camera import Camera
from Module.io.CameraStates import CameraState
from Module.io.Exceptions import CameraError, CaptureError


class OpenCVCamera(Camera):
    """OpenCV camera implementation of the Camera ABC. 
    Intended as a fallback for camera types that require specific SDKs."""

    def __init__(self, device_index=0):
        super().__init__()
        self.device_index = device_index
        self.cap: Optional[cv2.VideoCapture] = None
        self.fps = 30.0
        self.exposure: Optional[float] = None
        self.gain: Optional[float] = None

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._latest_frame = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._watch_window = None  # (x_start, y_start, width, height)
        self._parameters = {}  # user-defined camera params


    def open(self) -> None:
        self._require_state(CameraState.CLOSED)

        self.cap = cv2.VideoCapture(self.device_index)
        if not self.cap.isOpened():
            raise CameraError(f"Failed to open camera {self.device_index}.")

        self._state = CameraState.OPEN

    def close(self) -> None:
        if self._state == CameraState.STREAMING:
            self.stop_stream()

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self._state = CameraState.CLOSED

    def set_parameter(self, parameter="none", value=None) -> None:
        """
        Set camera parameters

        parameter="brightness","contrast","exposure","gain"
        value: value to set parameter to

        """
        self._require_state(CameraState.OPEN)

        self._parameters[parameter] = value
        if self.cap is not None:
            prop_map = {
                "brightness": cv2.CAP_PROP_BRIGHTNESS,
                "contrast": cv2.CAP_PROP_CONTRAST,
                "exposure": cv2.CAP_PROP_EXPOSURE,
                "gain": cv2.CAP_PROP_GAIN,
            }
            if parameter in prop_map and value is not None:
                self.cap.set(prop_map[parameter], value)

    def set_framerate(self, framerate: float) -> None:
        self._require_state(CameraState.OPEN)
        if self.cap is None:
            raise RuntimeError("Camera not opened.")

        cam_framerate = self.cap.get(cv2.CAP_PROP_FPS)

        if framerate <= 0:
            framerate = 1
            print(
                f"Warning: Requested framerate {framerate} is less than or equal to 0. "
                f"Setting camera framerate to 1"
            )

        if cam_framerate > 0 and framerate > cam_framerate:
            print(
                f"Warning: Requested framerate {framerate} FPS exceeds "
                f"camera capability ({cam_framerate:.1f} FPS). "
                f"Clamping to {cam_framerate:.1f} FPS."
            )
            self.fps = cam_framerate
        else:
            self.fps = framerate

    def set_watch_window(self, x_start, width, y_start, height) -> None:
        """Restrict frames to a ROI (Region of Interest)"""
        self._require_state(CameraState.OPEN)
        self._watch_window = (x_start, y_start, width, height)

    def grab_frame(self) -> np.ndarray:
        """Grab a single frame (blocking)"""
        self._require_state(CameraState.STREAMING)

        if self.cap is None:
            raise CaptureError("Camera not open.")

        ret, frame = self.cap.read()
        if not ret:
            raise CaptureError("Failed to grab frame.")

        if self._watch_window:
            x, y, w, h = self._watch_window
            frame = frame[y : y + h, x : x + w]

        with self._lock:
            self._latest_frame = frame.copy()

        return frame

    def start_stream(self, fps: float = 30) -> None:
        """Start a background thread to capture frames at fixed FPS"""
        self._require_state(CameraState.OPEN)
        if self._running:
            return  

        self.set_framerate(fps)
        self._running = True
        self._stop_event.clear()
        interval = 1.0 / self.fps
        self._state = CameraState.STREAMING

        def _loop():
            next_time = time.perf_counter()
            try:
                while self._running:
                    self.grab_frame()

                    next_time += interval
                    sleep_time = next_time - time.perf_counter()

                    if sleep_time > 0:
                        if self._stop_event.wait(timeout=sleep_time):
                            break
            except Exception as e:
                raise CaptureError("Stream interrupted.")


        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop_stream(self) -> None:
        """Stop streaming thread"""
        self._require_state(CameraState.STREAMING)

        if not self._thread:
            raise CameraError(
                "Streaming thread missing while stopping camera."
            )

        self._running = False
        self._stop_event.set()
        self._thread.join()
        self._thread = None
        self._state = CameraState.OPEN

    def grab_last_frame(self) -> Optional[np.ndarray]:
        """Return the latest frame grabbed by the streaming thread"""
        self._require_state(CameraState.STREAMING)
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()


