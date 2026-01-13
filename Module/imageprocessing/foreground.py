import cv2


def extract_foreground(img, background, thresh_val=35):
    fg = cv2.subtract(background, img)
    _, binary = cv2.threshold(fg, thresh_val, 255, cv2.THRESH_BINARY)
    # binary = cv2.adaptiveThreshold(fg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                                    cv2.THRESH_BINARY, 15, -5)
    return fg, binary