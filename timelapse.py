from Module.ui.main import TimelapseApp
import sys
from PySide6.QtWidgets import QApplication
from Module.io.OpenCVCamera import OpenCVCamera
import cv2

cam = OpenCVCamera()
cam.set()
cam.open()

stamp = 0
fps_avg = 0.0
alpha = 0.1

while True:
    ret, frame, this_stamp = cam.read()
    if not ret:
        continue

    dt = this_stamp - stamp
    stamp = this_stamp

    if dt > 0:
        fps_inst = 1.0 / dt
        fps_avg = alpha * fps_inst + (1 - alpha) * fps_avg

    cv2.putText(
        frame,
        f"{fps_avg:.1f} FPS",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    cv2.imshow("Live", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.close()

# app = QApplication(sys.argv)
# win = TimelapseApp()
# win.show()
# sys.exit(app.exec())
