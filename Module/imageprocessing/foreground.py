import cv2
import numpy as np

from Module.Tracking.Detection import BlobDetection
from Module.imageprocessing.ImageProcessor import find_components


def get_wall_foreground(frame_gray, wall_mask, canny_low=100, canny_high=200):
    """
    Detect worms on the wall using Canny edges.
    wall_mask: binary mask (same size as frame_gray) where 1 = wall region.
    """
    edges = cv2.Canny(frame_gray, canny_low, canny_high)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=2)
    return cv2.bitwise_and(edges, wall_mask)


def extract_foreground_advanced(
    img,
    background,  # can be static background OR RollingMedianBackground instance
    wall_mask=None,  # binary mask for wall region (optional)
    thresh_val=30,
    blur_ksize=5,
    min_area=10,
    use_rolling=False,  # if True, background is a RollingMedianBackground object
    motion_memory=None,  # optional MotionMemory instance to suppress ghosts
):
    """
    Improved foreground extraction for worm tracking at low frame rates.

    Returns:
        fg_diff : np.ndarray - absolute difference image
        mask    : np.ndarray - cleaned binary mask (0/255)
    """
    # 1. Get base foreground mask (adaptive or static)
    if use_rolling and hasattr(background, "get_foreground"):
        mask = background.get_foreground(img, thresh_val)
        fg_diff = (
            cv2.absdiff(img, background.background)
            if background.background is not None
            else np.zeros_like(img)
        )
    else:
        fg_diff = cv2.absdiff(background, img)
        fg_blur = cv2.GaussianBlur(fg_diff, (blur_ksize, blur_ksize), 0)
        _, mask = cv2.threshold(fg_blur, thresh_val, 255, cv2.THRESH_BINARY)

    # 2. Wall-specific detection (if mask provided)
    if wall_mask is not None:
        wall_fg = get_wall_foreground(img, wall_mask)
        mask = cv2.bitwise_or(mask, wall_fg)

    # 3. Morphological cleaning (preserve worm shape)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 4. Remove small components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    clean = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == i] = 255

    # 5. Optional motion memory suppression (remove static ghosts)
    if motion_memory is not None and clean.any():
        # motion_memory should have method `is_moving(contour)` or similar
        # For simplicity, we skip here; can be integrated in tracker later.
        pass

    return fg_diff, clean


def detect_blobs(
    thresh,
    region_mask=None,
    min_area=10,
    max_area=5000,
    min_aspect_ratio=1.5,  # worms are elongated
):
    """
    Detect connected blobs with additional filtering for worm shape.
    Returns list[BlobDetection].
    """
    fg = thresh.astype(np.uint8)
    _, labels, stats, _ = find_components(fg)

    detections = []
    for i in range(1, stats.shape[0]):
        x, y, w, h, area = stats[i]

        # Area filtering
        if area < min_area or area > max_area:
            continue

        # Aspect ratio filtering (width/height or height/width)
        aspect = max(w, h) / (min(w, h) + 1e-6)
        if aspect < min_aspect_ratio:
            continue

        mask = (labels[y : y + h, x : x + w] == i).astype(np.uint8)

        region = 0
        if region_mask is not None:
            region_slice = region_mask[y : y + h, x : x + w]
            core = np.sum((mask > 0) & (region_slice == 1))
            wall = np.sum((mask > 0) & (region_slice == 2))
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
