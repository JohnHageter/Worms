import os
import time
from datetime import datetime, timedelta
import cv2
import numpy as np
from Camera import Camera

SAVE_DIR = "timelapse_images" #All the output images will be saved to this directory which will be made when you start the script
os.makedirs(SAVE_DIR, exist_ok=True) 

INTERVAL_SECONDS = 5  # Set the tiem between pictures to 120 seconds (2 minutes)
DURATION_DAYS = 14 # Run the script for 14 days
start_time = datetime.now() #This is the time we start at
end_time = start_time + timedelta(days=DURATION_DAYS) # This is the time we end at

print("Starting run at: " + str(start_time))
print("Run will end at: " + str(end_time))

cam = Camera(index=0) #This opens the camera. should we have multiple cameras, we can open them individually
cam.set_watch_window(0, 1280, 0, 1024) # This just sets it so we're taking a picture from the camera's entire field of view
cam.set_attr("gain", 3) #This just controls the brighness of the image

img, ok = cam.grab_next_frame(to_bgr=False)  # This grabs teh very first image from the camera

try:
    while datetime.now() < end_time: #Everything indented beyond this line will run in a loop until we reach the end time
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # This gets the timestamp of when the current image is captured
        filename = os.path.join(SAVE_DIR, f"image_{timestamp}.png") # This designates the path for the image to be saved at. The timestamp will make every image name unique and you can read the file name to figure out when the image was captured.

        img, ok = cam.grab_next_frame(to_bgr=False) #Captures an image
        if ok and img is not None: # If the camera didn't fail to capture and image, we'll run the indented lines
            img = np.ascontiguousarray(img.copy(), dtype=np.uint8) #This converts the image type so we can save it properly
            cv2.imwrite(filename, img) # This line saves the image
            print(f"[{datetime.now()}] Saved image: {filename}") #This will log that the image was captured and saved properly

        else:
            print(f"[{datetime.now()}] Failed to grab image.") # If we failed to capture an image, the log will let us know

        time.sleep(INTERVAL_SECONDS) # This will wait the two minutes until we capture a new image

except KeyboardInterrupt:
    print("Timelapse interrupted by user.") # Ctrl+C will stop the script at any time

finally:
    cam.close() #If the script is stopped or finished we need to make sure to close out of the camera
    print("Camera closed. Timelapse finished.")
