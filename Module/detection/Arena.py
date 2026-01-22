import cv2
import numpy as np
from dataclasses import dataclass, replace

@dataclass(frozen=True)
class WellDetectionParams:
    well_tolerance: float = 0.15
    n_wells: int = 24
    hough_dp: float = 1.5
    min_dist: int = 50
    hough_param1: int = 24
    hough_param2: int = 25
    min_radius: int = 10
    max_radius: int = 100

def detect_wells(
    cap: cv2.VideoCapture, params: WellDetectionParams = WellDetectionParams()
):
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = cap.read()
    if not ret:
        raise ValueError("Cannot read the first frame from the input video.")

    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    wells, masks = _detect_wells_core(gray, params)
    return wells, masks, params

def _detect_wells_core(
    gray: np.ndarray, params: WellDetectionParams = WellDetectionParams()
):
    assert gray.ndim == 2

    gray_blur = cv2.medianBlur(gray, 7)
    H, W = gray.shape

    circles = cv2.HoughCircles(
        gray_blur,
        cv2.HOUGH_GRADIENT,
        dp=params.hough_dp,
        minDist=params.min_dist,
        param1=params.hough_param1,
        param2=params.hough_param2,
        minRadius=params.min_radius,
        maxRadius=params.max_radius,
    )

    wells, masks = [], []
    if circles is None:
        return wells, masks

    for x, y, r in np.round(circles[0]).astype(int):
        if 0 <= x < W and 0 <= y < H and r > 0:
            wells.append((float(x), float(y), float(r)))
            mask = np.zeros((H, W), np.uint8)
            cv2.circle(mask, (x, y), r, 255, -1)
            masks.append(mask.astype(bool))

    if wells and params.n_wells:
        rs = np.array([r for _, _, r in wells])
        r_med = np.median(rs)
        keep = np.abs(rs - r_med) / r_med < params.well_tolerance

        wells = [w for w, k in zip(wells, keep) if k][: params.n_wells]
        masks = [m for m, k in zip(masks, keep) if k][: params.n_wells]
        wells, masks = _sort_wells(wells, masks)

    return wells, masks

def _sort_wells(wells, well_masks, row_tol=50):
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
            current_y = y if current_y is None else (current_y + y) / 2
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

def detect_wells_interactive(
    cap: cv2.VideoCapture,
    params: WellDetectionParams=WellDetectionParams(),
    window_name: str = "Well Detection (Interactive)",
):

    # Read first frame only
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = cap.read()
    if not ret:
        raise ValueError("Cannot read first frame from video")

    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    def nothing(_):
        pass

    # ---- Trackbars ----
    cv2.createTrackbar(
        "well_tolerance x100",
        window_name,
        int(params.well_tolerance * 100),
        50,
        nothing,
    )
    cv2.createTrackbar("n_wells", window_name, params.n_wells, 96, nothing)
    cv2.createTrackbar(
        "hough_dp x10", window_name, int(params.hough_dp * 10), 50, nothing
    )
    cv2.createTrackbar("min_dist", window_name, params.min_dist, 500, nothing)
    cv2.createTrackbar("param1", window_name, params.hough_param1, 300, nothing)
    cv2.createTrackbar("param2", window_name, params.hough_param2, 300, nothing)
    cv2.createTrackbar("min_radius", window_name, params.min_radius, 500, nothing)
    cv2.createTrackbar("max_radius", window_name, params.max_radius, 500, nothing)

    last_params = params
    last_update = 0.0
    wells, masks = [], []

    while True:
        p = replace(
            last_params,
            well_tolerance=cv2.getTrackbarPos("well_tolerance x100", window_name)
            / 100.0,
            n_wells=max(1, cv2.getTrackbarPos("n_wells", window_name)),
            hough_dp=max(0.1, cv2.getTrackbarPos("hough_dp x10", window_name) / 10.0),
            min_dist=max(1, cv2.getTrackbarPos("min_dist", window_name)),
            hough_param1=max(1, cv2.getTrackbarPos("param1", window_name)),
            hough_param2=max(1, cv2.getTrackbarPos("param2", window_name)),
            min_radius=max(1, cv2.getTrackbarPos("min_radius", window_name)),
            max_radius=max(1, cv2.getTrackbarPos("max_radius", window_name)),
        )

        wells, masks = _detect_wells_core(gray, p)
        last_params = p


        disp = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for i, (x, y, r) in enumerate(wells):
            xi, yi, ri = int(round(x)), int(round(y)), int(round(r))
            cv2.circle(disp, (xi, yi), ri, (0, 255, 0), 2)
            cv2.putText(
                disp,
                str(i),
                (xi - 10, yi - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                3,
            )

        cv2.putText(
            disp,
            f"{len(wells)} wells",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.imshow(window_name, disp)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    cv2.destroyWindow(window_name)
    return wells, masks, last_params
