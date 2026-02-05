from PySide6.QtWidgets import QMainWindow, QSizePolicy
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt, QThread, QTimer

from Module.ui.docking.acquisition_config import AcquisitionSettingsDock
from Module.ui.docking.camera_config import CameraConfigDock
from Module.ui.docking.camera_preview import CameraPreviewDock
from Module.ui.docking.console_log import ConsoleDock

from Module.ui.workers.camera_worker import CameraWorker
from Module.ui.utils.overlay_label import cv_image_to_qpixmap
from Module.ui.utils.stylesheet import DARK_STYLESHEET


class TimelapseApp(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Planaria Tracker")
        self.resize(1400, 800)
        self.font_main = QFont("Helvetica", 10)
        self.font_console = QFont("Helvetica", 9)
        self.setFont(self.font_main)
        self.setStyleSheet(DARK_STYLESHEET)

        # Console dock
        self.console_dock = ConsoleDock(self)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.console_dock)
        self.console_dock.widget().setMinimumHeight(150)
        self.console_dock.widget().setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )

        # Camera config dock
        self.camera_dock = CameraConfigDock(self)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.camera_dock)

        # Camera preview dock (central widget)
        self.preview_dock = CameraPreviewDock(self)
        self.setCentralWidget(self.preview_dock)

        # Acquisition settings dock
        self.acquisition_dock = AcquisitionSettingsDock(self)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.acquisition_dock)
        self.tabifyDockWidget(self.camera_dock, self.acquisition_dock)
        self.camera_dock.raise_()

        # --- Camera Worker in a separate thread ---
        self.camera_thread = QThread()
        self.camera_worker = CameraWorker()
        self.camera_worker.moveToThread(self.camera_thread)
        self.camera_thread.start()

        # Connect worker signals to dock UI
        self.camera_worker.camera_opened.connect(self.camera_dock.on_camera_open)
        self.camera_worker.camera_closed.connect(self.camera_dock.on_camera_close)
        self.camera_worker.live_started.connect(self.camera_dock.on_live_start)
        self.camera_worker.live_stopped.connect(self.camera_dock.on_live_stopped)

        # Connect dock requests to worker slots
        self.camera_dock.request_open.connect(self.camera_worker.open_camera)
        self.camera_dock.request_close.connect(self.camera_worker.close_camera)
        self.camera_dock.request_snap.connect(self.camera_worker.snap_image)
        self.camera_dock.request_live.connect(
            lambda: self.camera_worker.start_stream(
                fps=self.camera_dock.frame_rate.value()
            )
        )
        self.camera_dock.request_live_stop.connect(self.camera_worker.stop_stream)

        # Parameter updates
        self.camera_dock.request_exposure_change.connect(
            lambda v: self.camera_worker.set_parameter("exposure", v)
        )
        self.camera_dock.request_gain_change.connect(
            lambda v: self.camera_worker.set_parameter("gain", v)
        )
        self.camera_dock.request_frame_rate_change.connect(
            lambda v: self.camera_worker.set_parameter(
                "fps", v
            )  # renamed from "frame_rate" to "fps"
        )

        # Preview: update QPixmap from frames
        self.camera_worker.frame_ready.connect(
            lambda frame: self.preview_dock.image_label.setPixmapSafe(
                cv_image_to_qpixmap(frame)
            )
        )

        self.ui_timer = QTimer()
        self.ui_timer.timeout.connect(
            lambda: self.camera_worker.snap_image()
        )  # or grab_last_frame()

        # Do NOT start the timer here
        # self.ui_timer.start(...)

        # Connect camera dock buttons to start/stop live stream
        self.camera_dock.request_live.connect(
            lambda: self.ui_timer.start(int(1000 / self.camera_dock.frame_rate.value()))
        )
        self.camera_dock.request_live_stop.connect(lambda: self.ui_timer.stop())

        # Stop the timer if the camera is closed
        self.camera_worker.camera_closed.connect(lambda: self.ui_timer.stop())
