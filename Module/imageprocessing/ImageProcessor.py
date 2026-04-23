import numpy as np
import cv2

def mask_to_well(binary, well):
    mask = np.zeros_like(binary, dtype=np.uint8)
    x, y, r = map(int, well)
    cv2.circle(mask, (x, y), r, 255, -1)
    return cv2.bitwise_and(binary, mask)


def filter_worms(worms, min_area=10, max_area=500):
    return [w for w in worms if min_area <= w.area <= max_area]



