from PySide6.QtWidgets import QLabel
from PySide6.QtCore import QRect, Qt, Signal, QPoint
from PySide6.QtGui import QPainter, QPen, QColor, QPixmap, QImage
import numpy as np


class CameraPreviewLabel(QLabel):
    watch_window_changed = Signal(QRect)  # emits the final rectangle

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pixmap: QPixmap | None = None
        self.overlay_rect: QRect | None = None

        # Drawing state
        self.selection_mode = False
        self.drawing = False
        self.start_point: QPoint | None = None
        self.end_point: QPoint | None = None
        self.locked_aspect_ratio: float | None = None

    def setPixmap(self, pixmap: QPixmap):
        self.pixmap = pixmap
        self.update()

    def set_overlay(self, rect: QRect):
        self.overlay_rect = rect
        self.update()

    def begin_watch_selection(self, aspect_ratio: float | None):
        self.selection_mode = True
        self.locked_aspect_ratio = aspect_ratio
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.set_overlay(QRect())

    def mousePressEvent(self, event):
        if self.selection_mode and event.button() == Qt.MouseButton.LeftButton:
            self.start_point = event.position().toPoint()
            self.end_point = self.start_point
            self.drawing = True

    def mouseMoveEvent(self, event):
        if self.drawing and self.start_point:
            self.end_point = event.position().toPoint()
            if self.start_point and self.end_point:
                rect = QRect(self.start_point, self.end_point).normalized()

            if self.locked_aspect_ratio:
                w = rect.width()
                h = int(w / self.locked_aspect_ratio)
                rect.setHeight(h)

            self.overlay_rect = rect
            self.update()


    def mouseReleaseEvent(self, event):
        if self.drawing and event.button() == Qt.MouseButton.LeftButton:
            self.drawing = False
            self.selection_mode = False

            if self.start_point and self.end_point:
                rect = QRect(self.start_point, self.end_point).normalized()

                if self.locked_aspect_ratio:
                    w = rect.width()
                    h = int(w / self.locked_aspect_ratio)
                    rect.setHeight(h)

                frame_rect = self.label_rect_to_frame_rect(rect)

                self.watch_window_changed.emit(frame_rect)

            self.set_overlay(QRect())

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)

        if self.pixmap:
            scaled = self.pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)

        if self.overlay_rect:
            painter.setPen(QPen(QColor(0, 255, 0), 2))
            painter.drawRect(self.overlay_rect)

        painter.end()

    def label_rect_to_frame_rect(self, rect: QRect) -> QRect:
        if not self.pixmap:
            return QRect()

        pixmap_size = self.pixmap.size()
        label_size = self.size()

        scale_w = pixmap_size.width() / label_size.width()
        scale_h = pixmap_size.height() / label_size.height()

        scaled = self.pixmap.scaled(
            label_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        x_offset = (label_size.width() - scaled.width()) // 2
        y_offset = (label_size.height() - scaled.height()) // 2

        rect_clamped = rect.intersected(
            QRect(x_offset, y_offset, scaled.width(), scaled.height())
        )

        frame_x0 = int((rect_clamped.x() - x_offset) * pixmap_size.width() / scaled.width())
        frame_y0 = int(
            (rect_clamped.y() - y_offset) * pixmap_size.height() / scaled.height()
        )
        frame_x1 = int(
            (rect_clamped.right() - x_offset) * pixmap_size.width() / scaled.width()
        )
        frame_y1 = int(
            (rect_clamped.bottom() - y_offset) * pixmap_size.height() / scaled.height()
        )

        return QRect(frame_x0, frame_y0, frame_x1 - frame_x0, frame_y1 - frame_y0)

    def mouse_pos_to_pixmap_pos(self, pos: QPoint) -> QPoint:
        if not self.pixmap:
            return pos

        scaled = self.pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        x_offset = (self.width() - scaled.width()) // 2
        y_offset = (self.height() - scaled.height()) // 2

        x = pos.x() - x_offset
        y = pos.y() - y_offset

        x = max(0, min(x, scaled.width() - 1))
        y = max(0, min(y, scaled.height() - 1))

        return QPoint(int(x), int(y))
