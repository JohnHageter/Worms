import sys
from Camera import Camera
import cv2
import numpy as np

cam = Camera(index=0)

cam.set_attr("exposure", 500)
cam.set_watch_window(0,512,0,512)
image, _ = cam.grab_image()

cv2.imshow("Live", image.astype(np.uint8))
cv2.waitKey()
image, _ = cam.grab_image()

cv2.imshow("Live", image.astype(np.uint8))
cv2.waitKey()
image, _ = cam.grab_image()

cv2.imshow("Live", image.astype(np.uint8))
cv2.waitKey()
image, _ = cam.grab_image()

cv2.imshow("Live", image.astype(np.uint8))
cv2.waitKey()

cam.close()