import sys
sys.path.insert(0, "dependencies")

import os
import time
from datetime import datetime, timedelta
import cv2
import numpy as np
from Camera.Camera import Camera

SAVE_DIR = "timelapse_images" 
os.makedirs(SAVE_DIR, exist_ok=True) 

INTERVAL_SECONDS = 120 
DURATION_DAYS = 14 
start_time = datetime.now() 
end_time = start_time + timedelta(days=DURATION_DAYS) 

print("Starting run at: " + str(start_time))
print("Run will end at: " + str(end_time))

cam = Camera(index=0) 
cam.set_watch_window(0, 1280, 0, 1024) 
cam.set_attr("gain", 3) 

img, ok = cam.grab_next_frame(to_bgr=False)  

try:
    while datetime.now() < end_time: 
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") 
        filename = os.path.join(SAVE_DIR, f"image_{timestamp}.png") 

        img, ok = cam.grab_next_frame(to_bgr=False) 
        if ok and img is not None: 
            img = np.ascontiguousarray(img.copy(), dtype=np.uint8) 
            cv2.imwrite(filename, img) 
            print(f"[{datetime.now()}] Saved image: {filename}") 

        else:
            print(f"[{datetime.now()}] Failed to grab image.") 

        time.sleep(INTERVAL_SECONDS) 

except KeyboardInterrupt:
    print("Timelapse interrupted by user.") 

finally:
    cam.close() 
    print("Camera closed. Timelapse finished.")
