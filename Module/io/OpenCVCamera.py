import cv2
from typing import Optional
import numpy as np
from Module.io.Camera import Camera, CameraError
import platform


class OpenCVCamera(Camera):
    def __init__(self, device_index=0):
        super().__init__()
        self.device_index = device_index
        self.cap: Optional[cv2.VideoCapture] = None

    def _open(self):
        system = platform.system()

        if system == "Windows":
            backend = cv2.CAP_DSHOW
        elif system == "Linux":
            backend = cv2.CAP_V4L2
        elif system == "Darwin": #MacOS
            backend = cv2.CAP_AVFOUNDATION
        else:
            backend = cv2.CAP_ANY  

        self.cap = cv2.VideoCapture(self.device_index, backend)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Failed to open camera {self.device_index} with backend {backend}"
        )

    def _close(self):
        if self.cap:
            self.cap.release()
            self.cap = None

    def _set(self, parameter: str, value) -> bool:
        if not self.cap:
            return False

        try:
            if parameter == "exposure":
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) #attempt to turn off auto exposure. prob won't work for most cameras
                return bool(self.cap.set(cv2.CAP_PROP_EXPOSURE, value))

            elif parameter == "gain":
                return bool(self.cap.set(cv2.CAP_PROP_GAIN, value))

            elif parameter == "fps":
                return bool(self.cap.set(cv2.CAP_PROP_FPS, value))

            else:
                return False

        except Exception:
            return False

    def _watch(self, x_off, width, y_off, height) -> bool:
        self.watch_window = x_off, width, y_off, height
        return True

    def _reset_watch_window(self):
        self.watch_window = None

    def _read_frame(self) -> tuple[bool, Optional[np.ndarray]]:
        if not self.cap:
            raise CameraError("Cannot read frame while camera is closed.")

        latest_frame = None
        while True:
            ret, frame = self.cap.read()  
            if not ret or frame is None:
                break
            latest_frame = frame  

            # If driver is fast and more frames exist, loop again
            # For most backends, read() will block until next frame
            # So effectively this will usually grab just one

            # Break if there’s nothing immediately available
            # But OpenCV doesn't provide a non-blocking check
            break

        if latest_frame is None:
            return False, None

        frame = latest_frame

        if self.watch_window is not None:
            x_off, width, y_off, height = self.watch_window
            x0 = max(0, x_off)
            y0 = max(0, y_off)
            x1 = min(x0 + width, frame.shape[1])
            y1 = min(y0 + height, frame.shape[0])
            if x1 > x0 and y1 > y0:
                frame = frame[y0:y1, x0:x1]

        return True, frame
