import os
import time
from datetime import datetime, timedelta
import cv2
import numpy as np
from Module.io.IDSCamera import Camera


OUT_FILE = "timelapse_images/Video.mp4"
OUT_DIR = "timelapse_images"
os.makedirs(OUT_DIR, exist_ok=True)

INTERVAL_SECONDS = 0.1
FPS = 1.0 / INTERVAL_SECONDS
PNG_INTERVAL_SECONDS = 120  


DURATION_DAYS = 14
start_time = datetime.now()
end_time = start_time + timedelta(days=DURATION_DAYS)

print("Starting run at:", start_time)
print("Run will end at:", end_time)


cam = Camera(index=0)
cam.set_watch_window(0, 1280, 0, 1024)
cam.set_attr("gain", 1)
cam.set_attr("exposure", 60000)

img, ok = cam.grab_next_frame(to_bgr=False)
if not ok or img is None:
    raise RuntimeError("Failed to grab initial frame")


img = cv2.resize(img, (img.shape[1] // 1, img.shape[0] // 1),
                 interpolation=cv2.INTER_AREA)

h, w = img.shape
img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

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
last_png_save_time = datetime.min


try:
    while datetime.now() < end_time:
        now = datetime.now()

        img, ok = cam.grab_next_frame(to_bgr=False)
        if not ok or img is None:
            print("Frame grab failed")
            time.sleep(INTERVAL_SECONDS)
            continue

        # PNG save every 2 minutes
        if (now - last_png_save_time).total_seconds() >= PNG_INTERVAL_SECONDS:
            timestamp = now.strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(OUT_DIR, f"image_{timestamp}.png")
            cv2.imwrite(filename, img)
            last_png_save_time = now
            print(f"[{now}] Saved image: {filename}")

        # Resize for mp4 
        img_resized = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        img_bgr = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)

        writer.write(img_bgr)

        cv2.imshow("Live", img_resized)
        cv2.waitKey(1)

        time.sleep(INTERVAL_SECONDS)


except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    writer.release()
    cam.close()
    cv2.destroyAllWindows()
    print("Camera closed. Single MP4 saved.")
