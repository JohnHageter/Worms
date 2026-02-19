from PySide6.QtWidgets import QDockWidget, QWidget, QVBoxLayout, QSizePolicy
from PySide6.QtCore import QRect, Qt, Slot, Signal
from Module.ui.utils.camera_preview_label import CameraPreviewLabel, QPixmap


class CameraPreviewDock(QDockWidget):
    drawn_watch_window_changed = Signal(QRect)

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
        
        #Local signal to the QLabel that the user drew a new watch window 
        #Should be managed by the QLabel so the worker can set
        self.image_label.watch_window_drawn.connect(self.drawn_watch_window_changed.emit)

    @Slot(QPixmap)
    def update_frame(self, frame_pixmap: QPixmap):
        self.image_label.setPixmap(frame_pixmap)

    @Slot(str)
    def start_watch_drawing(self, aspect_text: str):
        if aspect_text == "Free":
            aspect_ratio = None
        else:
            w, h = map(int, aspect_text.split(":"))
            aspect_ratio = w / h
        self.image_label.begin_watch_selection(aspect_ratio)

    @Slot(QRect)
    def on_drawn_watch_window_update(self, rect: QRect):
        self.image_label.set_overlay(rect)
