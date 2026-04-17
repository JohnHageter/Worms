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
    if frame.ndim == 2:
        annotated = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    else:
        annotated = frame.copy()

    for det in dets:
        color = (
            (0, 255, 0) if det.region == 'core' else (0, 165, 255)
        )

        if det.centroid is not None:
            x, y = map(int, det.centroid)
            cv2.circle(annotated, (x, y), 2, color, -1)

        for endpoint in (det.end1, det.end2):
            if endpoint is None:
                continue

            ex, ey = map(int, endpoint)
            cv2.circle(annotated, (ex, ey), 2, color, -1)

    return annotated


def annotate_region_boundaries(
    well_crop,
    well,
    wall_width=8,
    color_outer=(0, 255, 0),  # green
    color_inner=(255, 0, 0),  # blue
    thickness=2,
):
    """
    Annotate region boundaries for circular or rectangular ROIs.

    Parameters
    ----------
    well_crop : np.ndarray
        Cropped well image (grayscale or BGR)
    well : tuple
        Either:
            (cx, cy, r)     → circular well
            (x, y, w, h)    → rectangular / square ROI
        NOTE: Only geometry is used, not absolute position.
    wall_width : int
        Thickness of wall region (pixels)
    color_outer : tuple
        Color of outer boundary
    color_inner : tuple
        Color of core-wall boundary
    thickness : int
        Line thickness

    Returns
    -------
    annotated : np.ndarray
        Annotated BGR image
    """

    # Ensure BGR
    if well_crop.ndim == 2:
        annotated = cv2.cvtColor(well_crop, cv2.COLOR_GRAY2BGR)
    else:
        annotated = well_crop.copy()

    h, w = annotated.shape[:2]


    if len(well) == 3:
        _, _, r = map(int, well)
        cx, cy = w // 2, h // 2

        # Outer boundary
        cv2.circle(
            annotated,
            (cx, cy),
            r,
            color_outer,
            thickness,
        )

        # Core / wall boundary
        cv2.circle(
            annotated,
            (cx, cy),
            max(r - wall_width, 1),
            color_inner,
            thickness,
        )


    elif len(well) == 4:
        # Outer rectangle
        cv2.rectangle(
            annotated,
            (0, 0),
            (w - 1, h - 1),
            color_outer,
            thickness,
        )

        # Inner (core) rectangle
        cv2.rectangle(
            annotated,
            (wall_width, wall_width),
            (w - wall_width - 1, h - wall_width - 1),
            color_inner,
            thickness,
        )

    else:
        raise ValueError(
            "Invalid well format. Expected (cx, cy, r) or (x, y, w, h)."
        )

    return annotated


def annotate_tracks(
    frame, tracks, offset=(0, 0), scale=1.0
):
    ox, oy = offset

    if frame.ndim == 2:
        annotated = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    else:
        annotated = frame.copy()

    for t in tracks:
        if t.centroid is None:
            continue

        cx = int((t.centroid[0] - ox) * scale)
        cy = int((t.centroid[1] - oy) * scale)

        cv2.circle(annotated, (cx, cy), 2, (0, 255, 0), -1)
        cv2.putText(
            annotated,
            str(t.id),
            (cx + 5, cy - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

        if t.features.get('head') is not None:
            head = t.features.get("head")
            hx, hy = map(int, head)
            cv2.rectangle(
                annotated, (hx - 2, hy - 2), (hx + 2, hy + 2), (255, 0, 0), -1
            )

        if t.features.get('tail') is not None:
            tail = t.features.get("tail")
            tx, ty = map(int, tail)
            cv2.circle(annotated, (tx, ty), 2, (0, 0, 255), -1)

    return annotated


def stitch_wells(well_crops, crop_shape):
    """
    Stitch well crops into a composite image.

    Parameters
    ----------
    well_crops : list[np.ndarray or None]
        Each crop must be either:
            - (H, W) grayscale
            - (H, W, 3) BGR
        None entries are replaced with black images.
    crop_shape : tuple
        (H, W) target spatial size
    """
    H, W = crop_shape

    filled = []
    
    if len(well_crops) ==1:
        composite = well_crops[0]
        return composite

    for crop in well_crops:
        # Replace None with black BGR
        if crop is None:
            filled.append(np.zeros((H, W, 3), dtype=np.uint8))
            continue

        # Convert grayscale → BGR
        if crop.ndim == 2:
            crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)

        # Sanity check
        if crop.shape[:2] != (H, W):
            raise ValueError(f"Crop has shape {crop.shape[:2]}, expected {(H, W)}")

        filled.append(crop)

    # Pad to even count (2 wells per row)
    if len(filled) % 2 == 1:
        filled.append(np.zeros((H, W, 3), dtype=np.uint8))

    # Build rows
    rows = []
    for i in range(0, len(filled), 2):
        rows.append(np.hstack(filled[i : i + 2]))

    composite = np.vstack(rows)
    return composite


def create_square_region_mask(shape, margin):
    h, w = shape
    mask = np.ones((h, w), dtype=np.uint8)

    mask[:margin, :] = 2  # top wall
    mask[-margin:, :] = 2  # bottom wall
    mask[:, :margin] = 2  # left wall
    mask[:, -margin:] = 2  # right wall

    mask[mask != 2] = 1  # core
    return mask


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


def pad_to_size(img, target_h, target_w):
    h, w = img.shape[:2]

    pad_h = target_h - h
    pad_w = target_w - w

    if pad_h < 0 or pad_w < 0:
        raise ValueError("Target size is smaller than image")

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    return cv2.copyMakeBorder(
        img, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=0
    )


def crop_well(frame, well):
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    well = list(map(int, well))
    H, W = frame.shape[:2]

    if len(well) == 3:
        cx, cy, r = well
        x1, x2 = cx - r, cx + r
        y1, y2 = cy - r, cy + r

        x1c, x2c = max(0, x1), min(W, x2)
        y1c, y2c = max(0, y1), min(H, y2)

        crop = frame[y1c:y2c, x1c:x2c].copy()

        h, w = crop.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        cx_crop = cx - x1c
        cy_crop = cy - y1c
        cv2.circle(mask, (cx_crop, cy_crop), r, 255, -1)

        crop[mask == 0] = 0
        return crop

    elif len(well) == 4:
        x, y, w, h = well
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(W, x + w), min(H, y + h)
        return frame[y1:y2, x1:x2].copy()

    raise ValueError("ROI format not recognized. Expected (cx, cy, r) or (x, y, w, h).")


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
