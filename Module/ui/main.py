from PySide6.QtWidgets import QMainWindow, QSizePolicy
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt, QThread, QTimer
from Module.ui.docking.camera_preview_dock import CameraPreviewDock
from Module.ui.docking.camera_config_dock import CameraConfigDock
from Module.ui.docking.acquisition_settings_dock import AcquisitionSettingsDock
from Module.ui.docking.console_dock import ConsoleDock
from Module.ui.utils.stylesheet import DARK_STYLESHEET
from Module.ui.workers.camera_worker import CameraWorker

from Module.ui.utils.common import cv_image_to_qpixmap


class TimelapseApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Planaria Tracker")
        self.resize(1400, 800)
        self.setFont(QFont("Helvetica", 10))
        self.setStyleSheet(DARK_STYLESHEET)

        # ---------- Docks ----------
        self.console_dock = ConsoleDock(self)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.console_dock)
        self.console_dock.widget().setMinimumHeight(150)
        self.console_dock.widget().setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )

        self.camera_dock = CameraConfigDock(self)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.camera_dock)

        self.preview_dock = CameraPreviewDock(self)
        self.setCentralWidget(self.preview_dock)

        self.acquisition_dock = AcquisitionSettingsDock(self)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.acquisition_dock)
        self.tabifyDockWidget(self.camera_dock, self.acquisition_dock)
        self.camera_dock.raise_()

        # Camera Worker
        self.camera_thread = QThread()
        self.camera_worker = CameraWorker()
        self.camera_worker.moveToThread(self.camera_thread)
        self.camera_thread.start()
        QTimer.singleShot(0, self.camera_worker.init_timer)

        # Worker signaling
        self.camera_worker.frame_ready.connect(
            lambda frame: self.preview_dock.update_frame(cv_image_to_qpixmap(frame))
        )
        self.camera_worker.camera_opened.connect(self.camera_dock.on_camera_open)
        self.camera_worker.camera_closed.connect(self.camera_dock.on_camera_close)
        self.camera_worker.live_started.connect(self.camera_dock.on_live_start)
        self.camera_worker.live_stopped.connect(self.camera_dock.on_live_stopped)

        # Dock signaling
        self.camera_dock.request_open.connect(self.camera_worker.open_camera)
        self.camera_dock.request_close.connect(self.camera_worker.close_camera)
        self.camera_dock.request_snap.connect(self.camera_worker.snap_image)
        self.camera_dock.request_live.connect(
            lambda: self.camera_worker.start_stream(
                fps=self.camera_dock.frame_rate.value()
            )
        )
        self.camera_dock.request_live_stop.connect(self.camera_worker.stop_stream)

        self.camera_dock.request_exposure_change.connect(
            lambda v: self.camera_worker.set_parameter("exposure", v)
        )
        self.camera_dock.request_gain_change.connect(
            lambda v: self.camera_worker.set_parameter("gain", v)
        )
        self.camera_dock.request_frame_rate_change.connect(
            lambda v: self.camera_worker.update_fps(v)
        )

        # Watch window singaling
        self.camera_dock.request_watch_set.connect(self._handle_watch_set)
        self.preview_dock.image_label.watch_window_drawn.connect(self.camera_worker.set_watch_window)
        self.camera_worker.camera_watch_window_set.connect(self.camera_dock.on_watch_window_set)
        
        self.camera_dock.request_watch_reset.connect(self.camera_worker.reset_watch_window)
        self.camera_worker.camera_watch_window_reset.connect(self.camera_dock.on_watch_window_reset)
        
        self.preview_dock.image_label.watch_window_draw_state.connect(
            self.camera_dock.update_watch_draw_state
        )
        
        self.preview_dock.image_label.watch_window_drawn.connect(
            self.camera_dock.update_watch_dimensions
        )
        
        self.camera_worker.camera_watch_window_updated.connect(
            self.camera_dock.update_watch_dimensions
        )
        
        
    def _handle_watch_set(self):
        aspect_text = self.camera_dock.watch_aspect_ratio.currentText()
        self.preview_dock.start_watch_drawing(aspect_text)
