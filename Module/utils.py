import cv2
import numpy as np


def append_row(ds, row):
    ds.resize(ds.shape[0] + 1, axis=0)
    ds[-1] = row


def display_frame(title, frame):
    cv2.imshow(title, frame)
    cv2.waitKey(1)


def annotate_frame_metadata(frame, frame_idx=0, video_idx=0, num_worms=0, fps=2):
    current_sec = frame_idx / fps

    if frame.ndim != 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    H, _,_ = frame.shape

    m, s = divmod(int(current_sec), 60)
    h, m = divmod(m, 60)
    cv2.putText(
        frame,
        f"Video: {video_idx}",
        (5, 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1,
    )
    cv2.putText(
        frame,
        f"Frame: {frame_idx}",
        (5, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1,
    )
    cv2.putText(
        frame,
        f"Time: {h:03d}:{m:02d}:{s:02d}",
        (5, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 255, 0),
        1,
    )

    cv2.putText(
        frame,
        f"Num worms: {num_worms}",
        (5, 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 255, 0),
        1,
    )

    return frame


def annotate_detections(frame, dets):
    # Ensure color image for drawing

    if frame.ndim == 2:
        annotated = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    else:
        annotated = frame.copy()

    for det in dets:
        # Choose color by region
        color = (
            (0, 255, 0) if det.region == 'core' else (0, 165, 255)
        )

        # ---- Draw centroid ----
        if det.centroid is not None:
            x, y = map(int, det.centroid)
            cv2.circle(annotated, (x, y), 3, color, -1)

        # ---- Draw endpoints ----
        for endpoint in (det.end1, det.end2):
            if endpoint is None:
                continue

            ex, ey = map(int, endpoint)
            cv2.circle(annotated, (ex, ey), 3, color, -1)

    return annotated


def annotate_region_circles(
    well_crop,
    radius,
    wall_width=8,
    color_well=(0, 255, 0),  # green
    color_core=(255, 0, 0),  # blue (BGR)
    thickness=2,
):
    """
    Annotate a well image with region boundaries.

    Parameters
    ----------
    well_crop : np.ndarray
        Cropped well image (grayscale or BGR)
    radius : int
        Full well radius (in pixels)
    wall_width : int
        Width of wall band (in pixels)
    color_well : tuple
        BGR color for outer well boundary
    color_core : tuple
        BGR color for core-wall boundary
    thickness : int
        Circle line thickness

    Returns
    -------
    annotated : np.ndarray
        Annotated BGR image
    """

    # Ensure BGR image for drawing
    if well_crop.ndim == 2:
        annotated = cv2.cvtColor(well_crop, cv2.COLOR_GRAY2BGR)
    else:
        annotated = well_crop.copy()

    h, w = annotated.shape[:2]
    cx, cy = w // 2, h // 2

    # 1. Draw outer well boundary (green)
    cv2.circle(
        annotated,
        (cx, cy),
        radius,
        color_well,
        thickness,
    )

    # 2. Draw core/wall boundary (blue)
    cv2.circle(
        annotated,
        (cx, cy),
        radius - wall_width,
        color_core,
        thickness,
    )

    return annotated


def annotate_tracks(
    frame, tracks, offset=(0, 0), scale=1.0
):
    ox, oy = offset

    for t in tracks:
        if t.centroid is None:
            continue

        cx = int((t.centroid[0] - ox) * scale)
        cy = int((t.centroid[1] - oy) * scale)

        cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)
        cv2.putText(
            frame,
            str(t.id),
            (cx + 5, cy - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )


        if t.head is not None:
            hx = int((t.head[0] - ox) * scale)
            hy = int((t.head[1] - oy) * scale)
            cv2.rectangle(frame, (hx - 3, hy - 3), (hx + 3, hy + 3), (255, 0, 0), -1)

        if t.tail is not None:
            tx = int((t.tail[0] - ox) * scale)
            ty = int((t.tail[1] - oy) * scale)
            cv2.circle(frame, (tx, ty), 3, (0, 0, 255), -1)
    
    return frame


def stitch_wells(well_crops, crop_shape):
    """
    well_crops: list of images or None (in ID order)
    crop_shape: (H, W)
    """
    H, W = crop_shape

    filled = [
        crop if crop is not None else np.zeros((H, W), dtype=np.uint8)
        for crop in well_crops
    ]

    # Pad if odd
    if len(filled) % 2 == 1:
        filled.append(np.zeros((H, W), dtype=np.uint8))

    # Build rows (2 wells per row)
    rows = []
    for i in range(0, len(filled), 2):
        row = np.hstack(filled[i : i + 2])
        rows.append(row)

    composite = np.vstack(rows)
    return composite


def create_region_mask(shape, radius, wall_width=8):
    """
    Create a region mask for a single well crop.

    Parameters
    ----------
    shape : tuple
        (height, width) of well_crop
    radius : int
        Well radius in pixels (approximately width // 2)
    wall_width : int
        Thickness of wall region in pixels

    Returns
    -------
    region_mask : np.ndarray (uint8)
        0 = outside well
        1 = core
        2 = wall
    """
    h, w = shape
    cx, cy = w // 2, h // 2

    region_mask = np.zeros((h, w), dtype=np.uint8)

    # Core region
    cv2.circle(
        region_mask,
        (cx, cy),
        radius - wall_width,
        1,
        -1,
    )

    # Wall region
    cv2.circle(
        region_mask,
        (cx, cy),
        radius,
        2,
        -1,
    )
    cv2.circle(
        region_mask,
        (cx, cy),
        radius - wall_width,
        0,
        -1,
    )

    return region_mask


def crop_well(frame, well):
    """
    Crop a circular well region from the frame and mask outside the circle.
    """
    
    if frame.ndim == 3:
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    x, y, r = map(int, well)

    # Compute bounding box
    y1, y2 = y - r, y + r
    x1, x2 = x - r, x + r

    # Crop square region
    crop = frame[y1:y2, x1:x2].copy()

    # Create circular mask
    h, w = crop.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (r, r), r, 255, -1)

    # Apply mask
    if crop.ndim == 2:
        crop[mask == 0] = 0
    else:
        crop[mask == 0] = (0, 0, 0)

    return crop


def find_endpoints(skel):
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)

    neighbors = cv2.filter2D(skel, -1, kernel)
    ys, xs = np.where((skel > 0) & (neighbors == 255))
    return list(zip(xs, ys))


def trace_path(skel, start):
    h, w = skel.shape
    path = [start]
    visited = set([start])
    cur = start

    while True:
        x, y = cur
        next_pixel = None

        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    if skel[ny, nx] and (nx, ny) not in visited:
                        next_pixel = (nx, ny)
                        break
            if next_pixel is not None:
                break

        if next_pixel is None:
            break

        visited.add(next_pixel)
        path.append(next_pixel)
        cur = next_pixel

    return path


def create_writer(
    file: str, width: int, height: int, frame_rate: float = 10.0, is_color: bool = True
) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        file, fourcc, frame_rate, (width, height), isColor=is_color
    )

    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {file}")

    return writer
