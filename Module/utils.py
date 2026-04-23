from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Tuple, Optional

import cv2
import numpy as np


def append_row(ds, row) -> None:
    """Append one row to a resizable HDF5 dataset."""
    n = ds.shape[0]
    ds.resize((n + 1,))
    ds[n] = row


def open_video(video_path: str | Path) -> cv2.VideoCapture:
    """Open a video safely."""
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    return cap


def create_writer(
    file: str | Path,
    *,
    width: int,
    height: int,
    frame_rate: float,
    is_color: bool = True,
) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(file), fourcc, frame_rate, (width, height), is_color)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {file}")
    return writer


def load_grayscale_image(path: str | Path, scale: float = 1.0) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image {path}")
    if scale != 1.0:
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img


def display_frame(title: str, frame: np.ndarray, delay: int = 1) -> None:
    """
    Display a frame in an OpenCV window for live preview/debug.

    Parameters
    ----------
    title : str
        Window title.
    frame : np.ndarray
        Grayscale or BGR image.
    delay : int
        Delay in milliseconds passed to cv2.waitKey.
    """
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    cv2.imshow(title, frame)
    cv2.waitKey(delay)


def crop_well(frame: np.ndarray, well) -> np.ndarray:
    """
    Crop circular or rectangular ROI from frame.
    Well formats:
        (cx, cy, r)
        (x, y, w, h)
    """
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    H, W = frame.shape[:2]

    if len(well) == 3:
        cx, cy, r = well
        x1, y1 = cx - r, cy - r
        x2, y2 = cx + r, cy + r

        x1c, x2c = max(0, x1), min(W, x2)
        y1c, y2c = max(0, y1), min(H, y2)

        crop = frame[y1c:y2c, x1c:x2c].copy()
        h, w = crop.shape[:2]

        mask = np.zeros((h, w), np.uint8)
        cv2.circle(mask, (cx - x1c, cy - y1c), r, 255, -1)
        crop[mask == 0] = 0
        return crop

    elif len(well) == 4:
        x, y, w, h = well
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(W, x + w), min(H, y + h)
        return frame[y1:y2, x1:x2].copy()

    raise ValueError("Invalid well format")


def roi_origin_xy(well) -> Tuple[int, int]:
    """Return top-left origin of ROI crop in full-frame coords."""
    well = list(map(int, well))
    if len(well) == 3:
        cx, cy, r = well
        return cx - r, cy - r
    if len(well) == 4:
        x, y, _, _ = well
        return x, y
    raise ValueError("Invalid well format")



def create_region_mask(
    shape: Tuple[int, int],
    *,
    radius: int,
    wall_width: int,
) -> np.ndarray:
    """
    0 = outside well
    1 = core
    2 = wall
    """
    h, w = shape
    cx, cy = w // 2, h // 2
    mask = np.zeros((h, w), np.uint8)

    cv2.circle(mask, (cx, cy), radius - wall_width, 1, -1)
    cv2.circle(mask, (cx, cy), radius, 2, -1)
    cv2.circle(mask, (cx, cy), radius - wall_width, 0, -1)

    return mask



def annotate_frame_metadata(
    frame: np.ndarray,
    *,
    frame_idx: int,
    video_idx: int,
    num_worms: int,
    fps: float,
) -> np.ndarray:
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    t = frame_idx / max(fps, 1e-6)
    h, rem = divmod(int(t), 3600)
    m, s = divmod(rem, 60)

    cv2.putText(
        frame,
        f"Video {video_idx}  Frame {frame_idx}  {h:02d}:{m:02d}:{s:02d}  Worms {num_worms}",
        (5, 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return frame


def annotate_region_boundaries(
    crop: np.ndarray,
    *,
    wall_width: int,
    thickness: int = 1,
) -> np.ndarray:
    if crop.ndim == 2:
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)

    h, w = crop.shape[:2]
    r = min(h, w) // 2
    cx, cy = w // 2, h // 2

    cv2.circle(crop, (cx, cy), r, (0, 255, 0), thickness)
    cv2.circle(crop, (cx, cy), max(1, r - wall_width), (255, 0, 0), thickness)

    return crop


def annotate_tracks(
    frame: np.ndarray,
    tracks,
) -> np.ndarray:

    if frame.ndim == 2:
        annotated = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    else:
        annotated = frame.copy()

    for t in tracks:
        if t.centroid is None:
            continue

        cx, cy = int(t.centroid[0]), int(t.centroid[1])

        cv2.circle(annotated, (cx, cy), 2, (0, 255, 0), -1)

        cv2.putText(
            annotated,
            f"{t.id}",
            (cx + 4, cy - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

        if hasattr(t, "predicted") and t.predicted is not None:
            px, py = int(t.predicted[0]), int(t.predicted[1])
            cv2.circle(annotated, (px, py), 2, (0, 0, 255), -1)
            cv2.line(annotated, (cx, cy), (px, py), (255, 0, 0), 1)

        endpoints = getattr(t, "endpoints", None)
        ht = getattr(t, "ht_state", None)

        if ht is not None and endpoints is not None:
            if ht.head is not None:
                hx, hy = int(ht.head[0]), int(ht.head[1])
                cv2.circle(annotated, (hx, hy), 3, (255, 0, 0), -1)

            if ht.tail is not None:
                tx, ty = int(ht.tail[0]), int(ht.tail[1])
                cv2.circle(annotated, (tx, ty), 3, (0, 0, 255), -1)

    return annotated


def stitch_wells(crops: List[np.ndarray], crop_shape: Tuple[int, int]) -> np.ndarray:

    if len(crops) == 1:
        c = crops[0]
        if c is None:
            H, W = crop_shape
            return np.zeros((H, W, 3), np.uint8)
        if c.ndim == 2:
            return cv2.cvtColor(c, cv2.COLOR_GRAY2BGR)
        return c

    H, W = crop_shape

    filled = []
    for c in crops:
        if c is None:
            filled.append(np.zeros((H, W, 3), np.uint8))
        else:
            if c.ndim == 2:
                c = cv2.cvtColor(c, cv2.COLOR_GRAY2BGR)
            filled.append(c)

    if len(filled) % 2 == 1:
        filled.append(np.zeros((H, W, 3), np.uint8))

    rows = [np.hstack(filled[i : i + 2]) for i in range(0, len(filled), 2)]
    return np.vstack(rows)
