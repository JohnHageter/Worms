from PySide6.QtWidgets import (
    QLabel,
    QPushButton,
    QComboBox,
    QHBoxLayout,
    QSlider,
    QSpinBox,
)
from PySide6.QtCore import Qt, Signal
from Module.ui.docking.config_dock import ConfigPanel, ConfigDock
from Module.ui.utils.logger import logger

class CameraConfigDock(ConfigDock):
    request_open = Signal(str)
    request_close = Signal()
    request_exposure_change = Signal(int)
    request_gain_change = Signal(float)
    request_frame_rate_change = Signal(float)
    request_watch_set = Signal()
    request_watch_reset = Signal()
    request_snap = Signal()
    request_live = Signal()
    request_live_stop = Signal()

    def __init__(self, parent=None):
        super().__init__("Camera Configuration", parent)
        self._build_camera_selection()
        self._build_camera_settings()
        self._build_watch_window_settings()
        self._build_stream_settings()

        self.open_btn.clicked.connect(self._open_clicked)
        self.close_btn.clicked.connect(self._close_clicked)
        self.exposure.valueChanged.connect(self._exposure_changed)
        self.gain.valueChanged.connect(self._gain_changed)
        self.frame_rate.valueChanged.connect(self._frame_rate_changed)
        self.set_watch_window_btn.clicked.connect(self._set_watch_window)
        self.reset_watch_window_btn.clicked.connect(self._reset_watch_window)
        self.snap_btn.clicked.connect(self._snap)
        self.live_btn.clicked.connect(self._live)

        self.cam_streaming = False

    def _build_camera_selection(self):
        camera_panel = ConfigPanel("Camera")

        camera_row = QHBoxLayout()
        self.camera_type = QComboBox()
        self.camera_type.addItems(["IDS", "OpenCV", "Test"])

        camera_row.addWidget(QLabel("Type"))
        camera_row.addWidget(self.camera_type)
        camera_row.addStretch()

        camera_panel.add_row(camera_row)

        button_row = QHBoxLayout()
        self.open_btn = QPushButton("Open")
        self.close_btn = QPushButton("Close")

        self.close_btn.setEnabled(False)

        button_row.addWidget(self.open_btn)
        button_row.addWidget(self.close_btn)
        button_row.addStretch()

        camera_panel.add_row(button_row)

        self.add_panel(camera_panel)

    def _build_camera_settings(self):
        settings_panel = ConfigPanel("Settings")

        self.exposure = QSlider(Qt.Orientation.Horizontal)
        self.exposure.setRange(0, 100)
        self.exposure.setEnabled(False)
        self.exposure_value = QLabel("--")

        exposure_row = QHBoxLayout()
        exposure_row.addWidget(QLabel("Exposure"))
        exposure_row.addWidget(self.exposure)
        exposure_row.addWidget(self.exposure_value)
        settings_panel.add_row(exposure_row)

        self.gain = QSlider(Qt.Orientation.Horizontal)
        self.gain.setRange(0, 10)
        self.gain.setEnabled(False)

        self.gain_value = QLabel("--")

        gain_row = QHBoxLayout()
        gain_row.addWidget(QLabel("Gain"))
        gain_row.addWidget(self.gain)
        gain_row.addWidget(self.gain_value)

        settings_panel.add_row(gain_row)

        self.frame_rate = QSpinBox()
        self.frame_rate.setSingleStep(1)
        self.frame_rate.setValue(10)
        self.frame_rate.setRange(1, 200)
        self.frame_rate.setSuffix(" fps")
        self.frame_rate.setEnabled(False)

        frame_rate_row = QHBoxLayout()
        frame_rate_row.addWidget(QLabel("Frame rate"))
        frame_rate_row.addWidget(self.frame_rate)
        frame_rate_row.addStretch()

        settings_panel.add_row(frame_rate_row)

        self.add_panel(settings_panel)

    def _build_watch_window_settings(self):
        watch_panel = ConfigPanel("Watch Window")

        # ---------- Set / Reset buttons ----------
        button_row = QHBoxLayout()

        self.set_watch_window_btn = QPushButton("Set")
        self.reset_watch_window_btn = QPushButton("Reset")

        self.set_watch_window_btn.setEnabled(False)
        self.reset_watch_window_btn.setEnabled(False)

        button_row.addWidget(self.set_watch_window_btn)
        button_row.addWidget(self.reset_watch_window_btn)
        button_row.addStretch()

        watch_panel.add_row(button_row)

        # ---------- Aspect ratio ----------
        aspect_row = QHBoxLayout()

        self.watch_aspect_ratio = QComboBox()
        self.watch_aspect_ratio.addItems(
            [
                "Free",
                "1:1",
                "4:3",
                "16:9",
            ]
        )
        self.watch_aspect_ratio.setEnabled(False)

        aspect_row.addWidget(QLabel("Aspect ratio"))
        aspect_row.addWidget(self.watch_aspect_ratio)
        aspect_row.addStretch()

        watch_panel.add_row(aspect_row)

        # ---------- Explicit size ----------
        size_row = QHBoxLayout()

        self.watch_width = QSpinBox()
        self.watch_width.setRange(1, 10000)
        self.watch_width.setSuffix(" px")
        self.watch_width.setEnabled(False)

        self.watch_height = QSpinBox()
        self.watch_height.setRange(1, 10000)
        self.watch_height.setSuffix(" px")
        self.watch_height.setEnabled(False)

        size_row.addWidget(QLabel("Size"))
        size_row.addWidget(QLabel("W"))
        size_row.addWidget(self.watch_width)
        size_row.addWidget(QLabel("H"))
        size_row.addWidget(self.watch_height)
        size_row.addStretch()

        watch_panel.add_row(size_row)

        # ---------- Feedback: drawing state ----------
        feedback_row = QHBoxLayout()

        self.watch_draw_state = QLabel("Idle")
        self.watch_draw_state.setAlignment(Qt.AlignmentFlag.AlignRight)

        feedback_row.addWidget(QLabel("State"))
        feedback_row.addWidget(self.watch_draw_state)

        watch_panel.add_row(feedback_row)

        # ---------- Feedback: dimensions ----------
        dimensions_row = QHBoxLayout()

        self.watch_dimensions = QLabel("-- × -- px")
        self.watch_dimensions.setAlignment(Qt.AlignmentFlag.AlignRight)

        dimensions_row.addWidget(QLabel("Dimensions"))
        dimensions_row.addWidget(self.watch_dimensions)

        watch_panel.add_row(dimensions_row)

        # ---------- Add panel to dock ----------
        self.add_panel(watch_panel)

    def _build_stream_settings(self):
        """
        Builds the panel containing Snap and Live buttons for camera output.
        """
        stream_panel = ConfigPanel("Stream Settings")

        # Row for buttons
        button_row = QHBoxLayout()

        self.snap_btn = QPushButton("Snap")
        self.live_btn = QPushButton("Live")
        self.live_btn.setCheckable(True)

        # Initially disable buttons until camera is opened
        self.snap_btn.setEnabled(False)
        self.live_btn.setEnabled(False)

        # Add buttons to row
        button_row.addWidget(self.snap_btn)
        button_row.addWidget(self.live_btn)
        button_row.addStretch()  # Push buttons to the left

        # Add row to panel
        stream_panel.add_row(button_row)

        # Add panel to the dock
        self.add_panel(stream_panel)

    def _open_clicked(self):
        # self.open_btn.setEnabled(False)
        self.request_open.emit(self.camera_type.currentText())

    def on_camera_open(self):
        self.open_btn.setEnabled(False)
        self.close_btn.setEnabled(True)
        self.snap_btn.setEnabled(True)
        self.live_btn.setEnabled(True)
        self.exposure.setEnabled(True)
        self.gain.setEnabled(True)
        self.frame_rate.setEnabled(True)
        self.camera_type.setEnabled(False)
        self.watch_aspect_ratio.setEnabled(True)
        self.set_watch_window_btn.setEnabled(True)

    def _close_clicked(self):
        # self.close_btn.setEnabled(False)
        self.request_close.emit()

    def on_camera_close(self):
        self.open_btn.setEnabled(True)
        self.close_btn.setEnabled(False)
        self.snap_btn.setEnabled(False)
        self.live_btn.setEnabled(False)
        self.exposure.setEnabled(False)
        self.gain.setEnabled(False)
        self.frame_rate.setEnabled(False)
        self.camera_type.setEnabled(True)
        self.set_watch_window_btn.setEnabled(False)
        self.reset_watch_window_btn.setEnabled(False)

    def _exposure_changed(self):
        if not self.exposure.isEnabled():
            return
        self.request_exposure_change.emit(self.exposure.value())

    def _gain_changed(self):
        if not self.gain.isEnabled():
            return
        self.request_gain_change.emit(self.gain.value())

    def _frame_rate_changed(self):
        if not self.frame_rate.isEnabled():
            return
        self.request_frame_rate_change.emit(self.frame_rate.value())

    def _set_watch_window(self):
        self.request_watch_set.emit()
        
    def on_watch_window_set(self):
        self.set_watch_window_btn.setEnabled(False)
        self.reset_watch_window_btn.setEnabled(True)

    def _reset_watch_window(self):
        self.request_watch_reset.emit()

    def on_watch_window_reset(self):
        self.set_watch_window_btn.setEnabled(True)
        self.reset_watch_window_btn.setEnabled(False)

    def _snap(self):
        # self.snap_btn.setEnabled(False)
        self.request_snap.emit()

    def _live(self):
        if self.live_btn.isChecked():
            self.request_live.emit()
        else:
            self.request_live_stop.emit()

    def on_live_start(self):
        self.cam_streaming = True
        self.live_btn.setText("Stop")
        self.live_btn.setStyleSheet(
            "QPushButton { background-color: #28a745; color: white; }"
        )  # green
        self.snap_btn.setEnabled(False)
        self.live_btn.setEnabled(True)

    def on_live_stopped(self):
        self.cam_streaming = False
        self.live_btn.setText("Live")
        self.live_btn.setStyleSheet(
            "QPushButton { background-color: #1E1E1E; color: #E0E0E0; }"
        )  # normal dark
        self.snap_btn.setEnabled(True)
        self.live_btn.setEnabled(True)
