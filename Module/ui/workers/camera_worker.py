from typing import Optional
from PySide6.QtCore import QObject, Signal, Slot, QTimer
from Module.io.Camera import Camera
from Module.io.IDSCamera import IDSCamera
from Module.io.OpenCVCamera import OpenCVCamera
from Module.io.TestCamera import TestCamera
from Module.ui.utils.logger import logger


class CameraWorker(QObject):
    frame_ready = Signal(object)
    camera_opened = Signal()
    camera_closed = Signal()
    live_started = Signal()
    live_stopped = Signal()
    framerate_reverted = Signal(float)

    def __init__(self):
        super().__init__()
        self.camera: Optional[Camera] = None
        self.streaming = False
        self.requested_fps: float = 10.0
        self.timer: Optional[QTimer] = None

    @Slot(str)
    def open_camera(self, camera_type: str):
        try:
            if camera_type == "OpenCV":
                self.camera = OpenCVCamera()
            elif camera_type == "IDS":
                self.camera = IDSCamera()
            elif camera_type == "Test":
                self.camera = TestCamera()
            else:
                logger.log_signal.emit(f"Unknown camera type: {camera_type}", "ERROR")
                return

            self.camera._open()  # open camera, does not start stream
            self.camera_opened.emit()
            logger.log_signal.emit(f"{camera_type} camera opened", "INFO")

        except Exception as e:
            logger.log_signal.emit(f"Failed to open camera: {e}", "ERROR")
            self.camera = None

    @Slot()
    def close_camera(self):
        if self.camera:
            self.stop_stream()
            self.camera._close()
            self.camera = None
            self.camera_closed.emit()
            logger.log_signal.emit("Camera closed", "INFO")

    @Slot()
    def snap_image(self):
        if not self.camera:
            logger.log_signal.emit("Cannot snap: camera not open", "ERROR")
            return

        temp_stream = False
        if not self.streaming:
            self.camera._start_stream(max_fps=self.requested_fps)
            temp_stream = True

        try:
            frame = self.camera._grab_frame(timeout_s=1.0 / self.requested_fps)
            self.frame_ready.emit(frame)
            logger.log_signal.emit("Snap taken", "INFO")
        except Exception as e:
            logger.log_signal.emit(f"Snap failed: {e}", "ERROR")
        finally:
            if temp_stream:
                self.camera._stop_stream()

    @Slot(float)
    def start_stream(self, fps: float):
        if not self.camera:
            logger.log_signal.emit("Cannot start stream: camera not open", "ERROR")
            return

        self.requested_fps = fps
        self.camera._start_stream(max_fps=fps)
        self.streaming = True
        self.live_started.emit()
        logger.log_signal.emit(f"Live stream started at {fps} FPS", "INFO")

    @Slot()
    def stop_stream(self):
        if self.camera and self.streaming:
            self.camera._stop_stream()
            self.streaming = False
            self.live_stopped.emit()
            logger.log_signal.emit("Live stream stopped", "INFO")

    @Slot(str, object)
    def set_parameter(self, name, value):
        """Set camera parameter safely. Stop/start stream if needed."""
        if not self.camera:
            return

        was_streaming = self.streaming
        if was_streaming:
            self.camera._stop_stream()

        try:
            setattr(self.camera, name, value)
            logger.log_signal.emit(f"{name} set to {value}", "INFO")
        except Exception as e:
            logger.log_signal.emit(f"Failed to set {name}: {e}", "ERROR")
        finally:
            if was_streaming:
                self.camera._start_stream(max_fps=self.requested_fps)

    @Slot()
    def grab_preview_frame(self):
        """Pull the latest frame from a running stream for the UI."""
        if self.camera and self.streaming:
            frame = self.camera._grab_last_frame()
            if frame is not None:
                self.frame_ready.emit(frame)
            else:
                logger.log_signal.emit("No frame received", "WARNING")
