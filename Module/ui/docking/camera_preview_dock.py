from PySide6.QtWidgets import QDockWidget, QWidget, QVBoxLayout, QSizePolicy
from PySide6.QtCore import QRect, Signal, Qt
from Module.ui.utils.camera_preview_label import CameraPreviewLabel, QPixmap


class CameraPreviewDock(QDockWidget):
    watch_window_changed = Signal(QRect)

    def __init__(self, parent=None):
        super().__init__("Preview", parent)
        self.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)

        self.container = QWidget(self)
        self._layout = QVBoxLayout(self.container)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)

        self.image_label = CameraPreviewLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("background-color: black;")
        self.image_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._layout.addWidget(self.image_label)
        self.setWidget(self.container)

        # forward signal
        self.image_label.watch_window_changed.connect(self.watch_window_changed.emit)

    def update_frame(self, frame_pixmap: QPixmap):
        self.image_label.setPixmap(frame_pixmap)

    def start_watch_drawing(self, aspect_text: str):
        if aspect_text == "Free":
            aspect_ratio = None
        else:
            w, h = map(int, aspect_text.split(":"))
            aspect_ratio = w / h
        self.image_label.begin_watch_selection(aspect_ratio)

    def reset_watch_window(self):
        self.image_label.set_overlay(QRect())
        self.watch_window_changed.emit(QRect())
