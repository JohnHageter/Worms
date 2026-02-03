from PySide6.QtWidgets import QDockWidget, QWidget, QVBoxLayout
from PySide6.QtCore import Qt, QPoint, QRect, Signal
from PySide6.QtGui import QPixmap

from Module.ui.utils.overlay_label import OverlayLabel


class CameraPreviewDock(QDockWidget):
    """Dock widget to display camera frames and draw a watch window rectangle."""

    watch_window_changed = Signal(QRect)

    def __init__(self, parent=None):
        super().__init__("Preview", parent)

        self.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea
            | Qt.DockWidgetArea.RightDockWidgetArea
            | Qt.DockWidgetArea.TopDockWidgetArea
            | Qt.DockWidgetArea.BottomDockWidgetArea
        )

        # ---------- Container ----------
        self.container = QWidget(self)
        self._layout = QVBoxLayout(self.container)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)

        # ---------- Label to display frames ----------
        self.image_label = OverlayLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("background-color: black;")
        self._layout.addWidget(self.image_label)

        self.setWidget(self.container)

        # ---------- Drawing state ----------
        self.drawing = False
        self.start_point = QPoint()
        self.end_point = QPoint()
        self.watch_rect = QRect()

        # Enable mouse events
        self.image_label.setMouseTracking(True)
        self.image_label.installEventFilter(self)

    # ---------- Feed frames from camera ----------
    def update_frame(self, frame_pixmap: QPixmap):
        """Call this with a new QPixmap frame from the camera."""
        self.image_label.setPixmapSafe(frame_pixmap)

    # ---------- Mouse events for drawing ----------
    def eventFilter(self, source, event):
        if source is self.image_label:
            if event.type() == event.Type.MouseButtonPress:
                if event.button() == Qt.MouseButton.LeftButton:
                    self.drawing = True
                    self.start_point = event.position().toPoint()
                    self.end_point = self.start_point
                    return True

            elif event.type() == event.Type.MouseMove:
                if self.drawing:
                    self.end_point = event.position().toPoint()
                    rect = QRect(self.start_point, self.end_point).normalized()
                    self.image_label.set_overlay(rect)
                    return True

            elif event.type() == event.Type.MouseButtonRelease:
                if event.button() == Qt.MouseButton.LeftButton and self.drawing:
                    self.drawing = False
                    self.end_point = event.position().toPoint()
                    self.watch_rect = QRect(
                        self.start_point, self.end_point
                    ).normalized()
                    self.image_label.set_overlay(self.watch_rect)
                    self.watch_window_changed.emit(self.watch_rect)
                    return True

        return super().eventFilter(source, event)
