import cv2
import numpy as np

from Module.Tracking.Detection import BlobDetection
from Module.imageprocessing.ImageProcessor import find_components

# def extract_foreground(img, background, thresh_val=35):
#     fg = cv2.subtract(background, img)
#     _, binary = cv2.threshold(fg, thresh_val, 255, cv2.THRESH_BINARY)
#     return fg, binary


def extract_foreground(
    img,
    background,
    thresh_val=30,
    blur_ksize=5,
    min_area=10,
):
    """
    Extract foreground mask from background-subtracted image.

    Parameters
    ----------
    img : np.ndarray
        Grayscale image
    background : np.ndarray
        Grayscale background image
    thresh_val : int
        Threshold value for foreground detection
    blur_ksize : int
        Gaussian blur kernel size (odd)
    min_area : int
        Minimum connected component area to keep

    Returns
    -------
    fg : np.ndarray
        Absolute difference image
    mask : np.ndarray
        Clean binary foreground mask (uint8, 0 or 255)
    """

    fg = cv2.absdiff(background, img)
    fg_blur = cv2.GaussianBlur(fg, (blur_ksize, blur_ksize), 0)

    _, mask = cv2.threshold(
        fg_blur,
        thresh_val,
        255,
        cv2.THRESH_BINARY,
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    clean = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == i] = 255

    return fg, clean


def detect_blobs(thresh, region_mask=None):
    """
    Detect connected blobs from a binary threshold image.

    Returns a list[BlobDetection] where each detection contains:
      - bbox: (x, y, w, h) in full-frame coordinates
      - mask: uint8 0/1 mask in bbox coordinates
      - region: int (0 unknown, 1 core, 2 wall)
    """
    fg = thresh.astype(np.uint8)
    _, labels, stats, _ = find_components(fg)

    detections = []

    for i in range(1, stats.shape[0]):
        x, y, w, h, _area = stats[i]

        mask = (labels[y : y + h, x : x + w] == i).astype(np.uint8)

        region = 0
        if region_mask is not None:
            region_slice = region_mask[y : y + h, x : x + w]
            core = int(np.sum((mask > 0) & (region_slice == 1)))
            wall = int(np.sum((mask > 0) & (region_slice == 2)))
            if core > 0 or wall > 0:
                region = 2 if wall > core else 1

        detections.append(
            BlobDetection(
                bbox=(int(x), int(y), int(w), int(h)),
                mask=mask,
                region=region,
            )
        )

    return detections
