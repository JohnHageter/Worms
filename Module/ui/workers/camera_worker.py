from PySide6.QtCore import QObject, Signal, Slot, QTimer
from Module.io import __all__ as camera_names
from Module.ui.utils.logger import logger
import importlib


camera_classes = {}
for name in camera_names:
    module = importlib.import_module(f"Module.io.{name}")
    cls = getattr(module, name)
    camera_classes[name] = cls


class CameraWorker(QObject):
    frame_ready = Signal(object)
    camera_opened = Signal()
    camera_closed = Signal()

    def __init__(self):
        super().__init__()
        self.camera = None
        self.timer = None
        self.streaming = False

    @Slot(str)
    def open_camera(self, camera_type: str):
        try:
            key = f"{camera_type}Camera"
            cls = camera_classes.get(key)

            if cls is None:
                logger.log_signal.emit(f"Unknown camera type: {camera_type}", "ERROR")
                return

            self.camera = cls()
            # logger.log_signal.emit("Camera starting to open", "INFO")
            self.camera.open()
            self.camera.start_stream()
            self.camera_opened.emit()
            logger.log_signal.emit(f"{camera_type} opened", "INFO")

        except Exception as e:
            logger.log_signal.emit(f"Failed to open camera: {e}", "ERROR")
            self.camera = None

    @Slot()
    def close_camera(self):
        self.stop_stream()

        if self.camera:
            try:
                self.camera.close()
                logger.log_signal.emit("Camera closed", "INFO")
            except Exception as e:
                logger.log_signal.emit(f"Error closing camera: {e}", "ERROR")

        self.camera = None
        self.camera_closed.emit()

    @Slot(int)
    def start_stream(self):
        if not self.camera:
            logger.log_signal.emit("Cannot start stream: camera not open", "ERROR")
            return

        if self.timer is None:
            self.timer = QTimer()
            self.timer.timeout.connect(self._grab_frame)

        # The interval can be derived from the camera's frame_rate property if it exists
        interval_ms = (
            int(1000 / self.camera.frame_rate)
            if hasattr(self.camera, "frame_rate")
            else 33
        )
        self.timer.start(interval_ms)
        logger.log_signal.emit("Live stream started", "INFO")

    @Slot(object)
    def snap_image(self, save_path=None):
        """Grab one frame, start streaming if needed."""
        if not self.camera:
            logger.log_signal.emit("Cannot snap: camera not open", "ERROR")
            return

        started_for_snap = False
        if not self.streaming:
            self.start_stream()
            started_for_snap = True

        frame = self.camera._do_grab_frame()  # grab one frame
        if save_path:
            self.camera.save_frame(frame, save_path)
        self.frame_ready.emit(frame)
        logger.log_signal.emit("Snap taken", "INFO")

        # Stop temporary streaming if we started it just for Snap
        if started_for_snap:
            self.stop_stream()

    @Slot()
    def stop_stream(self):
        if self.timer and self.timer.isActive():
            self.timer.stop()
            logger.log_signal.emit("Live stream stopped", "INFO")

    def _grab_frame(self):
        if self.camera:
            frame = self.camera.grab_frame()
            self.frame_ready.emit(frame)

    @Slot(str, object)
    def update_setting(self, name, value):
        if not self.camera:
            return

        try:
            setattr(self.camera, name, value)
            logger.log_signal.emit(f"{name} set to {value}", "INFO")
        except Exception as e:
            logger.log_signal.emit(f"Failed to set {name}: {e}", "ERROR")
