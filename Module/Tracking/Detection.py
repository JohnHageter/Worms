from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np


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
