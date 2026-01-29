from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QDockWidget,
    QWidget,
    QVBoxLayout,
    QComboBox,
    QPushButton,
    QFormLayout,
    QSpinBox,
    QDoubleSpinBox,
    QGroupBox
)

from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt, Signal, QRect, QTimer
import numpy as np

from Module.io.OpenCVCamera import OpenCVCamera
from Module.io.IDSCamera import IDSCamera
from Module.io.TestCamera import TestCamera
from Module.io.CameraStates import CameraState
from Module.ui.Timelapse.preview import PreviewLabel


class TimelapseApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Planaria Tracker")
        self.resize(1400, 800)
        
        self.camera = None
        self.stream_timer = QTimer(self)
        self.stream_timer.timeout.connect(self.update_live_frame)


        self.preview = PreviewLabel()
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setStyleSheet("background: black; color: white;")
        self.preview.rectSelected.connect(self.on_watch_window_set)

        self.setCentralWidget(self.preview)

        self.camera_dock = self.make_camera_dock()
        self.save_dock = self.make_dock("Save Settings")
        self.tracking_dock = self.make_dock("Tracking Configuration")
        self.console_dock = self.make_dock("Console")

        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.camera_dock)
        self.tabifyDockWidget(self.camera_dock, self.save_dock)

        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.tracking_dock)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.console_dock)

        self.tracking_dock.hide()
        self.console_dock.hide()


        self.create_view_menu()

    def make_dock(self, title: str) -> QDockWidget:
        dock = QDockWidget(title, self)
        dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea
            | Qt.DockWidgetArea.RightDockWidgetArea
            | Qt.DockWidgetArea.BottomDockWidgetArea
        )
        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetClosable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.addWidget(QLabel(title))
        layout.addStretch()

        dock.setWidget(content)
        return dock
    
    def make_camera_dock(self) -> QDockWidget:
        dock = QDockWidget("Camera Configuration", self)

        container = QWidget()
        layout = QVBoxLayout(container)

        self.camera_type = QComboBox()
        self.camera_type.addItems(["OpenCV", "IDS", "Test"])

        self.open_camera_btn = QPushButton("Open Camera")
        self.open_camera_btn.clicked.connect(self.open_camera)

        layout.addWidget(QLabel("Camera Type"))
        layout.addWidget(self.camera_type)
        layout.addWidget(self.open_camera_btn)

        self.camera_settings = QGroupBox("Camera Settings")
        self.camera_settings.setEnabled(False)
        form = QFormLayout(self.camera_settings)


        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 240)
        self.fps_spin.setValue(30)
        self.fps_spin.valueChanged.connect(
            lambda v: self.camera.set_framerate(v) if self.camera else None
        )


        self.exposure_spin = QDoubleSpinBox()
        self.exposure_spin.setRange(0.01, 1000.0)
        self.exposure_spin.setSuffix(" ms")
        self.exposure_spin.valueChanged.connect(
            lambda v: self.camera.set_parameter("exposure", v) if self.camera else None
        )


        self.gain_spin = QDoubleSpinBox()
        self.gain_spin.setRange(0.0, 24.0)
        self.gain_spin.valueChanged.connect(
            lambda v: self.camera.set_parameter("gain", v) if self.camera else None
        )


        self.snap_btn = QPushButton("Snap")
        self.snap_btn.clicked.connect(self.snap)

        self.live_btn = QPushButton("Live")
        self.live_btn.setCheckable(True)
        self.live_btn.clicked.connect(self.toggle_live)

        self.close_camera_btn = QPushButton("Close Camera")
        self.close_camera_btn.clicked.connect(self.close_camera)

        form.addRow("Frame rate", self.fps_spin)
        form.addRow("Exposure", self.exposure_spin)
        form.addRow("Gain", self.gain_spin)
        form.addRow(self.snap_btn)
        form.addRow(self.live_btn)
        form.addRow(self.close_camera_btn)

        layout.addWidget(self.camera_settings)


        self.watch_window_group = QGroupBox("Watch Window")
        self.watch_window_group.setEnabled(False)

        self.set_watch_btn = QPushButton("Set Watch Window")
        self.set_watch_btn.clicked.connect(self.enable_watch_window_selection)

        watch_layout = QVBoxLayout(self.watch_window_group)
        watch_layout.addWidget(self.set_watch_btn)

        layout.addWidget(self.watch_window_group)
        layout.addStretch()

        dock.setWidget(container)
        return dock


    def create_view_menu(self):
        view_menu = self.menuBar().addMenu("View")

        for dock in [
            self.camera_dock,
            self.save_dock,
            self.tracking_dock,
            self.console_dock,
        ]:
            view_menu.addAction(dock.toggleViewAction())

    def open_camera(self):
        camera_type = self.camera_type.currentText()

        if hasattr(self, "camera") and self.camera is not None:
            try:
                if self.camera.state == CameraState.STREAMING:
                    self.camera.stop_stream()
                if self.camera.state == CameraState.OPEN:
                    self.camera.close()
            except Exception as e:
                print(f"Error closing camera: {e}")

            self.camera = None

        try:
            if camera_type == "IDS":
                self.camera = IDSCamera()
            elif camera_type == "Test":
                self.camera = TestCamera()
            elif camera_type == "OpenCV":
                self.camera = OpenCVCamera()
            else:
                raise ValueError(f"Unknown camera type: {camera_type}")

            self.camera.open()
            self.open_camera_btn.setEnabled(False)


        except Exception as e:
            print(f"Failed to open {camera_type} camera: {e}")
            self.camera = None
            self.camera_settings.setEnabled(False)
            self.watch_window_group.setEnabled(False)
            return

        self.camera_settings.setEnabled(True)
        self.watch_window_group.setEnabled(True)


    def enable_watch_window_selection(self):
        self.preview.selection_rect = QRect()
        self.preview.update()

    def on_watch_window_set(self, rect: QRect):
        print(f"Watch window set: {rect}")
        
    def display_frame(self, frame):
        if frame is None:
            return

        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame)

        h, w, ch = frame.shape
        bytes_per_line = ch * w

        qimg = QImage(
            frame.data,
            w,
            h,
            bytes_per_line,
            QImage.Format_BGR888,
        )

        pixmap = QPixmap.fromImage(qimg)
        self.preview.setPixmap(
            pixmap.scaled(
                self.preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

        
    def snap(self):
        if not self.camera:
            return

        try:
            if self.camera.state == CameraState.OPEN:
                self.camera.start_stream()

            frame = self.camera.grab_frame()
            self.display_frame(frame)

            if self.camera.state == CameraState.STREAMING:
                self.camera.stop_stream()

        except Exception as e:
            print(f"Snap failed: {e}")

    def toggle_live(self, checked):
        if not self.camera:
            self.live_btn.setChecked(False)
            return

        try:
            if checked:
                self.camera.start_stream(self.fps_spin.value())
                interval_ms = int(1000 / self.fps_spin.value())
                self.stream_timer.start(interval_ms)
                self.live_btn.setText("Stop")
            else:
                self.stream_timer.stop()
                if self.camera.state == CameraState.STREAMING:
                    self.camera.stop_stream()
                self.live_btn.setText("Live")

        except Exception as e:
            print(f"Live failed: {e}")
            self.live_btn.setChecked(False)
            
    def update_live_frame(self):
        try:
            frame = self.camera.grab_frame()
            self.display_frame(frame)
        except Exception as e:
            print(f"Stream error: {e}")
            self.toggle_live(False)
            
    def on_watch_window_set(self, rect: QRect):
        if not self.camera:
            return

        self.camera.set_watch_window(
            rect.x(),
            rect.width(),
            rect.y(),
            rect.height()
        )
    
    def close_camera(self):
        if not self.camera:
            return

        try:
            if self.stream_timer.isActive():
                self.stream_timer.stop()

            if self.camera.state == CameraState.STREAMING:
                self.camera.stop_stream()

            if self.camera.state == CameraState.OPEN:
                self.camera.close()

        except Exception as e:
            print(f"Error closing camera: {e}")

        self.camera = None
        self.camera_settings.setEnabled(False)
        self.watch_window_group.setEnabled(False)
        self.open_camera_btn.setEnabled(True)
        self.live_btn.setChecked(False)
        self.live_btn.setText("Live")
        self.preview.clear()



