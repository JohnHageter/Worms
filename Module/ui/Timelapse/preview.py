from PySide6.QtGui import QPainter, QPen, QColor, QMouseEvent
from PySide6.QtCore import Qt, QRect, QPoint, Signal
from PySide6.QtWidgets import QLabel

class PreviewLabel(QLabel):
    rectSelected = Signal(QRect)

    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        self.drawing = False
        self.start = QPoint()
        self.end = QPoint()
        self.selection_rect = QRect()

    def mousePressEvent(self, event: QMouseEvent):
        if event.buttons() & Qt.MouseButton.LeftButton:
            self.drawing = True
            self.start = event.pos()
            self.end = event.pos()
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.drawing:
            self.end = event.pos()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self.drawing:
            self.drawing = False
            self.selection_rect = QRect(self.start, self.end).normalized()
            self.rectSelected.emit(self.selection_rect)
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)

        if self.drawing or not self.selection_rect.isNull():
            painter = QPainter(self)
            painter.setPen(QPen(QColor(0, 255, 0), 2))
            rect = QRect(self.start, self.end).normalized() if self.drawing else self.selection_rect
            painter.drawRect(rect)
