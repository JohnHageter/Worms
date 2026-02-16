from typing import Optional
from PySide6.QtCore import QObject, Signal, Slot, QTimer, QRect, Qt
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
    frames_dropped = Signal(int)
    watch_window_set = Signal()
    watch_window_reset = Signal()

    def __init__(self):
        super().__init__()
        self.camera: Optional[Camera] = None
        self.streaming = False
        self.requested_fps = 10.0
        self.timer: Optional[QTimer] = None
        self.last_frame = None
        self._running = False  # controls background read loop

        self._last_ts = 0.0
        self._camera_max_fps = 30.0

    @Slot()
    def init_timer(self):
        self.timer = QTimer()
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.timeout.connect(self._read_frame)

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

            self.camera.open()
            self.camera_opened.emit()
            logger.log_signal.emit(f"{camera_type} camera opened", "INFO")
        except Exception as e:
            logger.log_signal.emit(f"Failed to open camera: {e}", "ERROR")
            self.camera = None

    @Slot()
    def close_camera(self):
        self.stop_stream()
        self._running = False
        if self.camera:
            self.camera.close()
            self.camera = None
            self.camera_closed.emit()
            logger.log_signal.emit("Camera closed", "INFO")

    @Slot()
    def snap_image(self):
        if self.camera:
            ret, frame, _ = self.camera.read()
            if ret and frame is not None:
                self.last_frame = frame
                self.frame_ready.emit(frame.copy())
                logger.log_signal.emit("Snap taken (direct read)", "INFO")
            else:
                logger.log_signal.emit("Snap failed: no frame", "WARNING")
        else:
            logger.log_signal.emit("Snap failed: camera not open", "ERROR")

    @Slot(float)
    def update_fps(self, fps: float):
        self.software_fps = fps
        if self.timer:
            self.timer.setInterval(int(1000 / fps))
        logger.log_signal.emit(f"Software FPS updated to {fps}", "INFO")

    @Slot(float)
    def start_stream(self, fps: float):
        if not self.camera:
            logger.log_signal.emit("Cannot start stream: camera not open", "ERROR")
            return

        self.software_fps = fps
        self.streaming = True

        if self.timer:
            self.timer.start(int(1000 / fps))

        self.live_started.emit()
        logger.log_signal.emit(f"Live stream started at {fps} FPS", "INFO")

    @Slot()
    def stop_stream(self):
        self.streaming = False
        if self.timer:
            self.timer.stop()
        self.live_stopped.emit()
        logger.log_signal.emit("Live stream stopped", "INFO")

    @Slot(str, object)
    def set_parameter(self, name, value):
        if not self.camera:
            return
        try:
            self.camera.set(name, value)
            logger.log_signal.emit(f"{name} set to {value}", "INFO")
        except Exception as e:
            logger.log_signal.emit(f"Failed to set {name}: {e}", "ERROR")

    def _read_frame(self):
        if not self.camera or not self.streaming:
            return

        try:
            ret, frame, _ = self.camera.read()
            if not ret or frame is None:
                return

            self.frame_ready.emit(frame.copy())
        except Exception as e:
            logger.log_signal.emit(f"Camera read error: {e}", "ERROR")

    @Slot()
    def set_watch_window(self, rect: QRect):
        if not self.camera:
            return
        
        if rect.isNull():
            self.camera.watch_window = None
        else:
            self.camera.watch(rect.x(),
                rect.width(),
                rect.y(),
                rect.height(),)
            self.watch_window_set.emit()

    @Slot()
    def reset_watch_window(self):
        if not self.camera:
            return
        
        self.camera.watch(-1,-1,-1,-1)
        logger.log_signal.emit("Reset Watch window", "INFO")
        self.watch_window_reset.emit()