import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_wells(frame, debug=False,
                 hough_dp=1.2, hough_minDist=200,
                 hough_param1=100, hough_param2=30,
                 minRadius=180, maxRadius=320,
                 fallback_min_area_ratio=0.001, fallback_max_area_ratio=0.6):
    
    if frame is None:
        return [], [], None if debug else ([], [])
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy()
    gray_blur = cv2.medianBlur(gray, 7)
    
    wells = []
    well_masks = []
    debug_img = None

    try:
        circles = cv2.HoughCircles(
            gray_blur,
            cv2.HOUGH_GRADIENT,
            dp=hough_dp,
            minDist=hough_minDist,
            param1=hough_param1,
            param2=hough_param2,
            minRadius=minRadius,
            maxRadius=maxRadius
        )
    except Exception:
        circles = None

    if circles is not None and len(circles) > 0:
        circles = np.round(circles[0]).astype(int)
        H, W = gray.shape
        for (x, y, r) in circles:
            if r <= 0 or x < 0 or y < 0 or x >= W or y >= H:
                continue
            wells.append((float(x), float(y), float(r)))
            mask = np.zeros_like(gray, dtype=np.uint8)
            # FULL radius now
            cv2.circle(mask, (int(x), int(y)), int(r), 255, thickness=-1)
            well_masks.append(mask.astype(bool))
    else:
        H, W = gray.shape
        img_area = float(H * W)
        found = False

        for invert in (False, True):
            th = cv2.adaptiveThreshold(
                gray_blur, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY,
                51, 7
            )

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)

            cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                area = cv2.contourArea(c)
                if area < img_area * fallback_min_area_ratio or area > img_area * fallback_max_area_ratio:
                    continue
                (cx, cy), radius = cv2.minEnclosingCircle(c)
                circle_area = np.pi * (radius**2 + 1e-9)
                circularity = area / circle_area
                if 0.4 < circularity <= 1.6 and radius > minRadius * 0.5:
                    wells.append((float(cx), float(cy), float(radius)))
                    mask = np.zeros_like(gray, dtype=np.uint8)
                    cv2.circle(mask, (int(round(cx)), int(round(cy))), int(round(radius)), 255, thickness=-1)
                    well_masks.append(mask.astype(bool))
                    found = True
            if found:
                break

    # Merge overlapping wells
    merged = []
    merged_masks = []
    used = [False] * len(wells)
    for i, (x, y, r) in enumerate(wells):
        if used[i]:
            continue
        group = [(x, y, r)]
        used[i] = True
        for j in range(i + 1, len(wells)):
            if used[j]:
                continue
            x2, y2, r2 = wells[j]
            if np.hypot(x - x2, y - y2) < max(10, 0.25 * ((r + r2) / 2.0)):
                group.append((x2, y2, r2))
                used[j] = True

        xs, ys, rs = [g[0] for g in group], [g[1] for g in group], [g[2] for g in group]
        mx, my, mr = float(np.mean(xs)), float(np.mean(ys)), float(np.mean(rs))
        merged.append((mx, my, mr))

        if len(group) == 1:
            idx = wells.index(group[0])
            merged_masks.append(well_masks[idx])
        else:
            mm = np.zeros_like(gray, dtype=bool)
            for (gx, gy, gr) in group:
                mask = np.zeros_like(gray, dtype=np.uint8)
                cv2.circle(mask, (int(round(gx)), int(round(gy))), int(round(gr)), 255, thickness=-1)
                mm = mm | mask.astype(bool)
            merged_masks.append(mm)

    wells = merged
    well_masks = merged_masks

    if debug:
        debug_img = frame.copy() if frame.ndim == 3 else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for idx, (x, y, r) in enumerate(wells):
            cv2.circle(debug_img, (int(round(x)), int(round(y))), int(round(r)), (0, 255, 0), 2)
            cv2.circle(debug_img, (int(round(x)), int(round(y))), max(2, int(round(r * 0.05))), (0, 0, 255), -1)
            cv2.putText(debug_img, f"{idx}", (int(round(x)) + 6, int(round(y)) + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    return (wells, well_masks, debug_img) if debug else (wells, well_masks)

def mask_wells(gray, circles):
    """
    Input:
        gray    - grayscale full frame
        circles - list of wells: [(x, y, r), ...]
    
    Output:
        List of boolean masks, one per well, fully filling the well interior.
    """
    well_masks = []

    for (x, y, r) in circles:
        mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.circle(mask, (int(round(x)), int(round(y))), int(round(r)), 255, thickness=-1)
        well_masks.append(mask.astype(bool))

    return well_masks

def sort_wells(wells, well_masks, row_tol=50):
    """
    Sort wells top-left to bottom-right.

    wells: list of (x, y, r)
    well_masks: list of corresponding masks
    row_tol: pixels to group wells in the same row
    """

    combined = list(zip(wells, well_masks))
    combined = sorted(combined, key=lambda wm: wm[0][1])  # sort by y first

    sorted_rows = []
    current_row = []
    current_y = None

    for well, mask in combined:
        y = well[1]
        if current_y is None or abs(y - current_y) <= row_tol:
            current_row.append((well, mask))
            current_y = y if current_y is None else (current_y + y)/2
        else:
            # sort the current row by x (left to right)
            current_row.sort(key=lambda wm: wm[0][0])
            sorted_rows.extend(current_row)
            current_row = [(well, mask)]
            current_y = y
    if current_row:
        current_row.sort(key=lambda wm: wm[0][0])
        sorted_rows.extend(current_row)

    wells_sorted, masks_sorted = zip(*sorted_rows)
    return list(wells_sorted), list(masks_sorted)

def subtract_background(video, n_bg=5, alpha = 0.1, minimum=35, maximum=255, display=True):
    """
    Compute background with adaptation to global brightness changes and subtract it.

    Args:
        video_frames : list or array of frames (H x W x C or H x W)
        n_bg         : number of initial frames to compute the initial background
        threshold    : threshold for foreground mask
        alpha        : running average update rate for adaptive background
        display      : whether to display first frame, background, and mask

    Returns:
        background : final background image
        fg_masks   : list of foreground masks for each frame
    """

    gray_stack = []
    for frame in video[:n_bg]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy()
        gray_stack.append(gray.astype(np.float32))
    background = np.median(np.stack(gray_stack, axis=0), axis=0)

    fg_masks = []

    for i, frame in enumerate(video):
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy()
        gray = gray.astype(np.float32)

        # Normalize frame to match background mean (handles brightness shifts)
        bg_mean = np.mean(background)
        frame_mean = np.mean(gray)
        if frame_mean > 0:
            gray_norm = gray * (bg_mean / frame_mean)
        else:
            gray_norm = gray

        # Compute foreground mask
        diff = np.abs(gray_norm - background)
        _, fg_mask = cv2.threshold(diff.astype(np.uint8), minimum, maximum, cv2.THRESH_BINARY)
        fg_masks.append(fg_mask)

        # Update background adaptively
        background = (1 - alpha) * background + alpha * gray_norm

    if display:
        plt.figure(figsize=(12,4))
        plt.subplot(1,3,1)
        plt.title("First frame")
        plt.imshow(video[0], cmap='gray' if video[0].ndim==2 else None)
        plt.axis('off')

        plt.subplot(1,3,2)
        plt.title("Final background")
        plt.imshow(background.astype(np.uint8), cmap='gray')
        plt.axis('off')

        plt.subplot(1,3,3)
        plt.title("Foreground mask (first frame)")
        plt.imshow(fg_masks[0], cmap='gray')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    return background.astype(np.uint8), fg_masks

def normalize_frame(frame, reference):
    """
    Normalize frame intensity to match the reference (background) mean.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim==3 else frame.copy()
    ref_mean = np.mean(reference)
    frame_mean = np.mean(gray)
    if frame_mean > 0:
        gray_norm = (gray.astype(np.float32) * (ref_mean / frame_mean)).clip(0,255).astype(np.uint8)
    else:
        gray_norm = gray
    return gray_norm