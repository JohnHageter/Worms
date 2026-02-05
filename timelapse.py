from Module.ui.main import TimelapseApp
import sys, time
from PySide6.QtWidgets import QApplication
from Module.io.OpenCVCamera import OpenCVCamera
import cv2

cam = OpenCVCamera()
cam.open()
cam.set("gain", 1)
cam.set("exposure", 1)
cam.set("fps", 3)
# cam.watch(0, 50, 0, 50)

stamp = 0
fps_avg = 0.0
alpha = 0.1

requested_fps = 10  # whatever you want
frame_interval = 1.0 / requested_fps
last_stamp = None

while True:
    ret, frame, this_stamp = cam.read()
    if not ret:
        continue

    if last_stamp is not None:
        elapsed = this_stamp - last_stamp
        fps = 1.0 / elapsed if elapsed > 0 else 0.0
    else:
        fps = 0.0

    last_stamp = this_stamp

    # Display
    cv2.putText(
        frame, f"{fps:.2f} FPS", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
    )
    cv2.imshow("Live", frame)

    # Wait so we don’t oversample
    sleep_time = frame_interval - (time.perf_counter() - this_stamp)
    if sleep_time > 0:
        time.sleep(sleep_time)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.close()

# app = QApplication(sys.argv)
# win = TimelapseApp()
# win.show()
# sys.exit(app.exec())
