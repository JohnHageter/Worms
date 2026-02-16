from PySide6.QtGui import QPixmap, QImage
import numpy as np

def cv_image_to_qpixmap(frame: np.ndarray) -> QPixmap:
    if frame.ndim == 2:
        h, w = frame.shape
        qimg = QImage(frame.data, w, h, w, QImage.Format.Format_Grayscale8)
    else:
        frame_rgb = frame[..., ::-1].copy()
        h, w, ch = frame_rgb.shape
        qimg = QImage(frame_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)
