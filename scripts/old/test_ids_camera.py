from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import sys
import threading
from typing import Optional

import numpy as np
import cv2

from PySide6.QtCore import QPointF, QRectF, Qt, QRect, QPoint, Signal, QObject, QThread
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QTransform
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QHBoxLayout,
    QGraphicsView,
    QGraphicsScene,
)

from Module.io.OpenCVCamera import OpenCVCamera
from Module.io.IDSCamera import IDSCamera
from Module.io.Camera import Parameter


# =============================================================================
# Utilities
# =============================================================================


def cv_to_qpixmap(frame: np.ndarray) -> QPixmap:
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(img)


# =============================================================================
# Camera worker thread
# =============================================================================


class CameraWorker(QObject):
    frame_ready = Signal(np.ndarray)

    def __init__(self, camera: IDSCamera):
        super().__init__()
        self.camera = camera
        self.running = True

    def run(self):
        while self.running:
            success, frame, _ = self.camera.read()
            if success and frame is not None:
                self.frame_ready.emit(frame)

    def stop(self):
        self.running = False


# =============================================================================
# ROI-capable image viewer
# =============================================================================


class ImageView(QGraphicsView):
    roi_changed = Signal(QRect)

    def __init__(self):
        super().__init__()

        self.setScene(QGraphicsScene(self))
        self.pixmap_item = None

        self._drawing = False
        self._start = None
        self._rect_item = None

        # Rendering options
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        self._zoom = 1.0

    # -------------------------------------------------------------

    def set_frame(self, frame: np.ndarray):
        pixmap = cv_to_qpixmap(frame)

        if self.pixmap_item is None:
            self.scene().clear()
            self.pixmap_item = self.scene().addPixmap(pixmap)
            self.scene().setSceneRect(pixmap.rect())
            self.reset_zoom()
        else:
            self.pixmap_item.setPixmap(pixmap)

    # -------------------------------------------------------------
    # Zoom controls
    # -------------------------------------------------------------

    def zoom_in(self):
        self._apply_zoom(1.25)

    def zoom_out(self):
        self._apply_zoom(0.8)

    def reset_zoom(self):
        self.setTransform(QTransform())
        self._zoom = 1.0
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def _apply_zoom(self, factor: float):
        self._zoom *= factor
        self.scale(factor, factor)

    # -------------------------------------------------------------
    # ROI drawing (scene coordinates == camera pixels)
    # -------------------------------------------------------------

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drawing = True

            self._start: QPointF = self.mapToScene(event.position().toPoint())

            self._rect_item = self.scene().addRect(QRectF(), QPen(Qt.GlobalColor.red, 2))

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drawing and self._rect_item and self._start is not None:
            scene_pos: QPointF = self.mapToScene(event.position().toPoint())
            rect = QRectF(self._start, scene_pos).normalized()
            self._rect_item.setRect(rect)

        super().mouseMoveEvent(event)


    def mouseReleaseEvent(self, event):
        if self._drawing and self._rect_item:
            self._drawing = False

            rect_f: QRectF = self._rect_item.rect()
            rect = rect_f.toAlignedRect()  # QPoint-safe QRect
            rect = rect.intersected(self.sceneRect().toRect())

            self.roi_changed.emit(rect)

        super().mouseReleaseEvent(event)


# =============================================================================
# Main window
# =============================================================================


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IDS Camera Control")

        # Camera
        self.camera = IDSCamera()
        self.camera.open()

        # Worker thread
        self.worker_thread = QThread()
        self.worker = CameraWorker(self.camera)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.frame_ready.connect(self.on_frame)
        self.worker_thread.start()

        # UI --------------------------------------------------------
        self.view = ImageView()
        self.view.roi_changed.connect(self.on_roi_drawn)

        zoom_in = QPushButton("Zoom +")
        zoom_out = QPushButton("Zoom −")
        zoom_reset = QPushButton("Reset Zoom")

        zoom_in.clicked.connect(self.view.zoom_in)
        zoom_out.clicked.connect(self.view.zoom_out)
        zoom_reset.clicked.connect(self.view.reset_zoom)

        self.gain_slider = QSlider(Qt.Orientation.Horizontal)
        self.gain_slider.setRange(0, 10)
        self.gain_slider.sliderReleased.connect(self.on_gain)

        self.exp_slider = QSlider(Qt.Orientation.Horizontal)
        self.exp_slider.setRange(100, 5000000)
        self.exp_slider.sliderReleased.connect(self.on_exposure)

        reset_roi = QPushButton("Reset ROI")
        reset_roi.clicked.connect(self.reset_roi)

        controls = QVBoxLayout()
        controls.addWidget(QLabel("Gain"))
        controls.addWidget(self.gain_slider)
        controls.addWidget(QLabel("Exposure (µs)"))
        controls.addWidget(self.exp_slider)
        controls.addWidget(reset_roi)

        controls.addSpacing(20)
        controls.addWidget(QLabel("View"))
        controls.addWidget(zoom_in)
        controls.addWidget(zoom_out)
        controls.addWidget(zoom_reset)

        layout = QHBoxLayout()
        layout.addWidget(self.view)
        layout.addLayout(controls)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    # -----------------------------------------------------------------

    def on_frame(self, frame: np.ndarray):
        self.view.set_frame(frame)

    def on_gain(self):
        value = self.gain_slider.value()
        self.camera.set(Parameter.GAIN, float(value))

    def on_exposure(self):
        value = self.exp_slider.value()
        self.camera.set(Parameter.EXPOSURE, value)

    def on_roi_drawn(self, rect: QRect):
        x = int(rect.x())
        y = int(rect.y())
        w = int(rect.width())
        h = int(rect.height())
        self.camera.watch(x, w, y, h)

    def reset_roi(self):
        self.camera.watch()

    # -----------------------------------------------------------------

    def closeEvent(self, event):
        self.worker.stop()
        self.worker_thread.quit()
        self.worker_thread.wait()
        self.camera.close()
        event.accept()


# =============================================================================
# Entry point
# =============================================================================


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(1200, 800)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
