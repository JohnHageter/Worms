from PySide6.QtWidgets import QLabel
from PySide6.QtGui import QPainter, QPen, QColor, QPixmap, QImage
from PySide6.QtCore import QRect, Qt

import numpy as np

class OverlayLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.overlay_rect: QRect | None = None
        self.current_pixmap: QPixmap | None = None

    def setPixmapSafe(self, pixmap: QPixmap):
        # Store the frame; do not call super().setPixmap()
        self.current_pixmap = pixmap
        self.update()  # trigger paintEvent

    def set_overlay(self, rect: QRect):
        self.overlay_rect = rect
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        # Fill black background
        painter.fillRect(self.rect(), Qt.GlobalColor.black)

        # Draw frame if pixmap exists
        if self.current_pixmap is not None and isinstance(self.current_pixmap, QPixmap):
            scaled = self.current_pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)

        # Draw overlay rectangle
        if self.overlay_rect:
            pen = QPen(QColor(0, 255, 0), 2)
            painter.setPen(pen)
            painter.drawRect(self.overlay_rect)

        painter.end()


def cv_image_to_qpixmap(frame: np.ndarray) -> QPixmap:
    """Convert OpenCV BGR/GRAY image to QPixmap"""
    if frame.ndim == 2:  # grayscale
        h, w = frame.shape
        qimg = QImage(frame.data, w, h, w, QImage.Format.Format_Grayscale8)
    else:  # BGR to RGB
        frame_rgb = frame[..., ::-1].copy()
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)
