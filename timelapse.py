import tkinter as tk
from Module.ui.TimelapseUI import TimelapseApp

if __name__ == "__main__":
    root = tk.Tk()
    app = TimelapseApp(root)
    root.mainloop()


'''
import sys
sys.path.insert(0, "dependencies")

import time
from datetime import datetime, timedelta
import cv2
import numpy as np
from Module.io.Camera import Camera

# -----------------------------
# Settings
# -----------------------------
OUT_FILE = "Video2.mp4"

INTERVAL_SECONDS = 0.1
FPS = 1.0 / INTERVAL_SECONDS

DURATION_DAYS = 14
start_time = datetime.now()
end_time = start_time + timedelta(days=DURATION_DAYS)

print("Starting run at:", start_time)
print("Run will end at:", end_time)

# -----------------------------
# Camera
# -----------------------------
cam = Camera(index=0)
cam.set_watch_window(0, 1280, 0, 1024)
cam.set_attr("gain", 1)
cam.set_attr("exposure", 60000)

# Grab first frame
img, ok = cam.grab_next_frame(to_bgr=False)
if not ok or img is None:
    raise RuntimeError("Failed to grab initial frame")

# -----------------------------
# Resize ONCE and lock size
# -----------------------------
img = cv2.resize(img, (img.shape[1] // 1, img.shape[0] // 1),
                 interpolation=cv2.INTER_AREA)

h, w = img.shape

# Convert to BGR for MP4 compatibility
img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# -----------------------------
# Video writer (single MP4)
# -----------------------------
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(
    OUT_FILE,
    fourcc,
    FPS,
    (w, h),
    isColor=True
)

if not writer.isOpened():
    raise RuntimeError("Could not open video writer")

writer.write(img_bgr)

# -----------------------------
# Capture loop
# -----------------------------
try:
    while datetime.now() < end_time:
        img, ok = cam.grab_next_frame(to_bgr=False)
        if not ok or img is None:
            print("Frame grab failed")
            continue

        # Resize EXACTLY the same way
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

        # Display
        cv2.imshow("Live", img)
        cv2.waitKey(1)

        # Convert & write
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        writer.write(img_bgr)

        time.sleep(INTERVAL_SECONDS)

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    writer.release()
    cam.close()
    cv2.destroyAllWindows()
    print("Camera closed. Single MP4 saved.")
