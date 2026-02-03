from PySide6.QtWidgets import (
    QLabel,
    QPushButton,
    QComboBox,
    QHBoxLayout,
    QSlider,
    QSpinBox,
)
from PySide6.QtCore import Qt, Signal
from Module.ui.docking.docking_panel import ConfigPanel, ConfigDock
from Module.ui.utils.logger import logger

class CameraConfigDock(ConfigDock):
    # Signals
    request_open = Signal(str)
    request_close = Signal()
    request_exposure_change = Signal(int)
    request_gain_change = Signal(float)
    request_frame_rate_change = Signal(float)
    request_watch_set = Signal()
    request_watch_reset = Signal()
    request_snap = Signal()
    request_live = Signal()

    def __init__(self, parent=None, console=None):
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

    def _build_camera_selection(self):
        # ---------- Camera panel ----------
        camera_panel = ConfigPanel("Camera")

        camera_row = QHBoxLayout()
        self.camera_type = QComboBox()
        self.camera_type.addItems(["OpenCV", "IDS", "Test"])

        camera_row.addWidget(QLabel("Type"))
        camera_row.addWidget(self.camera_type)
        camera_row.addStretch()

        camera_panel.add_row(camera_row)

        # ---------- Buttons ----------
        button_row = QHBoxLayout()
        self.open_btn = QPushButton("Open")
        self.close_btn = QPushButton("Close")

        self.close_btn.setEnabled(False)

        button_row.addWidget(self.open_btn)
        button_row.addWidget(self.close_btn)
        button_row.addStretch()

        camera_panel.add_row(button_row)

        # ---------- Add to dock ----------
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
        self.frame_rate.setRange(1, 1000)
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

    def _build_display(self):
        output_panel = ConfigPanel("Display")
        # Row for buttons
        button_row = QHBoxLayout()

        self.snap_btn = QPushButton("Snap")
        self.live_btn = QPushButton("Live")

        # Initially disable buttons until camera is open
        self.snap_btn.setEnabled(False)
        self.live_btn.setEnabled(False)

        button_row.addWidget(self.snap_btn)
        button_row.addWidget(self.live_btn)
        button_row.addStretch()  # push buttons to the left

        output_panel.add_row(button_row)

        # Add the panel to the dock
        self.add_panel(output_panel)

    def _build_stream_settings(self):
        """
        Builds the panel containing Snap and Live buttons for camera output.
        """
        stream_panel = ConfigPanel("Stream Settings")

        # Row for buttons
        button_row = QHBoxLayout()

        self.snap_btn = QPushButton("Snap")
        self.live_btn = QPushButton("Live")

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

    def _close_clicked(self):
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

    def _exposure_changed(self):
        self.request_exposure_change.emit(self.exposure.value())

    def _gain_changed(self):
        self.request_gain_change.emit(self.gain.value())

    def _frame_rate_changed(self):
        self.request_frame_rate_change.emit(self.frame_rate.value())

    def _set_watch_window(self):
        self.request_watch_set.emit()

    def _reset_watch_window(self):
        self.request_watch_reset.emit()

    def _snap(self):
        self.request_snap.emit()

    def _live(self):
        self.request_live.emit()
