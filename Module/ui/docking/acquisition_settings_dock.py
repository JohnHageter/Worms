from PySide6.QtWidgets import (
    QDockWidget,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QDoubleSpinBox,
    QPushButton,
    QTextEdit,
    QComboBox,
    QGroupBox,
    QFileDialog,
)
from PySide6.QtCore import Qt


class AcquisitionSettingsDock(QDockWidget):
    """Dock for acquisition-specific settings with Timelapse and Video sections."""

    def __init__(self, parent=None, console=None):
        super().__init__("Acquisition Settings", parent)

        self.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )


        self.container = QWidget(self)
        self._layout = QVBoxLayout(self.container)
        self._layout.setContentsMargins(6, 6, 6, 6)
        self._layout.setSpacing(6)
        self._layout.addStretch()
        self.setWidget(self.container)

        self._build_timelapse_panel()
        self._build_video_panel()

        self.timelapse_panel.toggled.connect(self._on_timelapse_toggled)
        self.video_panel.toggled.connect(self._on_video_toggled)

    def add_panel(self, panel: QWidget):
        self._layout.insertWidget(self._layout.count() - 1, panel)

    # Timelapse
    def _build_timelapse_panel(self):
        self.timelapse_panel = QGroupBox("Timelapse Acquisition")
        self.timelapse_panel.setCheckable(True)
        self.timelapse_panel.setChecked(True)  #start enabled
        layout = QVBoxLayout(self.timelapse_panel)

        # Duration (minutes)
        duration_row = QHBoxLayout()
        self.timelapse_duration = QDoubleSpinBox()
        self.timelapse_duration.setRange(0.1, 1440)
        self.timelapse_duration.setSuffix(" min")
        self.timelapse_duration.setDecimals(1)
        duration_row.addWidget(QLabel("Duration"))
        duration_row.addWidget(self.timelapse_duration)
        duration_row.addStretch()
        layout.addLayout(duration_row)

        # Interval (seconds)
        interval_row = QHBoxLayout()
        self.timelapse_interval = QDoubleSpinBox()
        self.timelapse_interval.setRange(0.1, 3600)
        self.timelapse_interval.setSuffix(" s")
        self.timelapse_interval.setDecimals(1)
        interval_row.addWidget(QLabel("Interval"))
        interval_row.addWidget(self.timelapse_interval)
        interval_row.addStretch()
        layout.addLayout(interval_row)

        # Save directory
        save_dir_row = QHBoxLayout()
        self.timelapse_save_dir = QLineEdit()
        self.browse_save_dir_btn = QPushButton("Browse")
        save_dir_row.addWidget(QLabel("Save directory"))
        save_dir_row.addWidget(self.timelapse_save_dir)
        save_dir_row.addWidget(self.browse_save_dir_btn)
        layout.addLayout(save_dir_row)

        # Output format
        format_row = QHBoxLayout()
        self.timelapse_format = QComboBox()
        self.timelapse_format.addItems(["PNG", "MP4"])
        format_row.addWidget(QLabel("Output format"))
        format_row.addWidget(self.timelapse_format)
        format_row.addStretch()
        layout.addLayout(format_row)

        # Picture prefix
        prefix_row = QHBoxLayout()
        self.timelapse_prefix = QLineEdit()
        prefix_row.addWidget(QLabel("Picture prefix"))
        prefix_row.addWidget(self.timelapse_prefix)
        prefix_row.addStretch()
        layout.addLayout(prefix_row)

        # Notes
        notes_layout = QVBoxLayout()
        self.timelapse_notes = QTextEdit()
        self.timelapse_notes.setPlaceholderText("Enter notes about this acquisition...")
        notes_layout.addWidget(QLabel("Notes"))
        notes_layout.addWidget(self.timelapse_notes)
        layout.addLayout(notes_layout)

        # Current camera settings: exposure, gain, watch window
        camera_row = QHBoxLayout()
        self.timelapse_exposure = QLabel("--")
        self.timelapse_gain = QLabel("--")
        self.timelapse_watch = QLabel("-- × --")
        camera_row.addWidget(QLabel("Exposure"))
        camera_row.addWidget(self.timelapse_exposure)
        camera_row.addWidget(QLabel("Gain"))
        camera_row.addWidget(self.timelapse_gain)
        camera_row.addWidget(QLabel("Watch window"))
        camera_row.addWidget(self.timelapse_watch)
        camera_row.addStretch()
        layout.addLayout(camera_row)

        # Run / Stop buttons
        run_row = QHBoxLayout()
        self.timelapse_run_btn = QPushButton("Run")
        self.timelapse_stop_btn = QPushButton("Stop")
        run_row.addWidget(self.timelapse_run_btn)
        run_row.addWidget(self.timelapse_stop_btn)
        run_row.addStretch()
        layout.addLayout(run_row)

        self.add_panel(self.timelapse_panel)

    # Video acquisition
    def _build_video_panel(self):
        self.video_panel = QGroupBox("Video Acquisition")
        self.video_panel.setCheckable(True)
        self.video_panel.setChecked(False)  # start disabled
        layout = QVBoxLayout(self.video_panel)

        # Duration (seconds)
        duration_row = QHBoxLayout()
        self.video_duration = QDoubleSpinBox()
        self.video_duration.setRange(1, 3600)
        self.video_duration.setSuffix(" s")
        duration_row.addWidget(QLabel("Duration"))
        duration_row.addWidget(self.video_duration)
        duration_row.addStretch()
        layout.addLayout(duration_row)

        # Number of videos
        num_row = QHBoxLayout()
        self.video_number = QSpinBox()
        self.video_number.setRange(1, 1000)
        num_row.addWidget(QLabel("Number of videos"))
        num_row.addWidget(self.video_number)
        num_row.addStretch()
        layout.addLayout(num_row)

        # Interval between videos
        interval_row = QHBoxLayout()
        self.video_interval = QDoubleSpinBox()
        self.video_interval.setRange(0, 3600)
        self.video_interval.setSuffix(" s")
        interval_row.addWidget(QLabel("Interval"))
        interval_row.addWidget(self.video_interval)
        interval_row.addStretch()
        layout.addLayout(interval_row)

        # Video prefix
        prefix_row = QHBoxLayout()
        self.video_prefix = QLineEdit("video_")
        prefix_row.addWidget(QLabel("Video prefix"))
        prefix_row.addWidget(self.video_prefix)
        prefix_row.addStretch()
        layout.addLayout(prefix_row)

        # Save directory with browse
        save_row = QHBoxLayout()
        self.video_save_dir = QLineEdit()
        self.video_browse_btn = QPushButton("Browse")
        save_row.addWidget(QLabel("Save directory"))
        save_row.addWidget(self.video_save_dir)
        save_row.addWidget(self.video_browse_btn)
        layout.addLayout(save_row)

        # Notes
        notes_layout = QVBoxLayout()
        self.video_notes = QTextEdit()
        self.video_notes.setPlaceholderText("Enter notes about this acquisition...")
        notes_layout.addWidget(QLabel("Notes"))
        notes_layout.addWidget(self.video_notes)
        layout.addLayout(notes_layout)

        # Current camera settings: exposure, gain, framerate, watch window
        camera_row = QHBoxLayout()
        self.video_exposure = QLabel("--")
        self.video_gain = QLabel("--")
        self.video_framerate = QLabel("--")
        self.video_watch = QLabel("-- × --")
        camera_row.addWidget(QLabel("Exposure"))
        camera_row.addWidget(self.video_exposure)
        camera_row.addWidget(QLabel("Gain"))
        camera_row.addWidget(self.video_gain)
        camera_row.addWidget(QLabel("Framerate"))
        camera_row.addWidget(self.video_framerate)
        camera_row.addWidget(QLabel("Watch window"))
        camera_row.addWidget(self.video_watch)
        camera_row.addStretch()
        layout.addLayout(camera_row)

        # Run / Stop buttons
        run_row = QHBoxLayout()
        self.video_run_btn = QPushButton("Run")
        self.video_stop_btn = QPushButton("Stop")
        run_row.addWidget(self.video_run_btn)
        run_row.addWidget(self.video_stop_btn)
        run_row.addStretch()
        layout.addLayout(run_row)

        self.add_panel(self.video_panel)


    def _on_timelapse_toggled(self, checked: bool):
        if checked:
            # Disable video acquisition panel if timelapse is enabled
            self.video_panel.setChecked(False)
            self.video_panel.setEnabled(False)
        else:
            # Re-enable video acquisition panel if timelapse is off
            self.video_panel.setEnabled(True)


    def _on_video_toggled(self, checked: bool):
        if checked:
            # Disable timelapse panel if video is enabled
            self.timelapse_panel.setChecked(False)
            self.timelapse_panel.setEnabled(False)
        else:
            # Re-enable timelapse panel if video is off
            self.timelapse_panel.setEnabled(True)
