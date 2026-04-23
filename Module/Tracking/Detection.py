from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Tuple
import numpy as np
import cv2


DetectionFate = Literal[
    "created",
    "matched_active",
    "suppressed_near_active",
    "suppressed_interaction",
    "matched_tentative",
    "promoted_to_track",
    "expired_tentative",
    "ignored_ambiguous",
]


@dataclass(frozen=True)
class BlobDetection:
    """
    bbox: (x, y, w, h) in full-frame coordinates
    mask: uint8 binary mask in bbox coordinates (0/1 or 0/255)
    region: region label int
    """

    bbox: Tuple[int, int, int, int]
    mask: np.ndarray
    region: int = 0
    fate: DetectionFate = "created"


def find_components(binary):
    return cv2.connectedComponentsWithStats(binary, connectivity=8)


def detect_blobs(
    binary_fg: np.ndarray,
    *,
    region_mask=None,
    min_area: int,
    max_area: int | None = None,
):
    """
    Detect connected components from a binary foreground image.

    This function is deliberately conservative: it only filters by
    simple geometric constraints and does NOT make identity decisions.

    Returns: list[BlobDetection]
    """
    fg = binary_fg.astype(np.uint8)
    _, labels, stats, _ = find_components(fg)

    detections = []

    for i in range(1, stats.shape[0]):
        x, y, w, h, area = stats[i]

        if area < min_area:
            continue
        if max_area is not None and area > max_area:
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


def blob_area(det: BlobDetection) -> int:
    m = det.mask
    return int(np.count_nonzero(m))


def blob_centroid(det: BlobDetection) -> np.ndarray:
    """
    Returns centroid in full-frame coordinates as float array [cx, cy].
    """
    x0, y0, w, h = det.bbox
    ys, xs = np.where(det.mask > 0)
    if xs.size == 0:
        return np.array([x0 + w / 2.0, y0 + h / 2.0], dtype=float)
    cx = x0 + float(xs.mean())
    cy = y0 + float(ys.mean())
    return np.array([cx, cy], dtype=float)
