import cv2
import numpy as np

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

    # 1. Absolute difference (robust to sign)
    fg = cv2.absdiff(background, img)

    # 2. Denoise
    fg_blur = cv2.GaussianBlur(fg, (blur_ksize, blur_ksize), 0)

    # 3. Threshold (global)
    _, mask = cv2.threshold(
        fg_blur,
        thresh_val,
        255,
        cv2.THRESH_BINARY,
    )

    # 4. Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 5. Remove tiny components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    clean = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == i] = 255

    return fg, clean
