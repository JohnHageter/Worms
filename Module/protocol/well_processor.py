from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from Module.Tracking.Tracker import MultiOrganismTracker
from Module.imageprocessing.foreground import extract_foreground, detect_blobs
from Module.utils import (
    annotate_region_boundaries,
    annotate_tracks,
    create_region_mask,
    crop_well,
)


@dataclass
class WellContext:
    well_id: int
    roi: np.ndarray
    background: np.ndarray
    tracker: MultiOrganismTracker  # MultiOrganismTracker
    region_mask: Optional[np.ndarray] = None  # cached per-well crop size


class WellProcessor:
    """
    Processes a single well for one frame.
    Returns (annotated_well_bgr, tracks).
    """

    def __init__(
        self,
        well_contexts: List[WellContext],
        *,
        foreground_thresh_val: int = 20,
        wall_width_region: int = 16,
        wall_width_draw: int = 8,
        region_boundary_thickness: int = 1,
        debug = False
    ):
        self.ctx = well_contexts
        self.foreground_thresh_val = foreground_thresh_val
        self.wall_width_region = wall_width_region
        self.wall_width_draw = wall_width_draw
        self.region_boundary_thickness = region_boundary_thickness
        self.debug = debug

    @staticmethod
    def _attach_features_from_ht(track) -> None:
        """
        annotate_tracks expects:
          track.features['head'], ['tail'], ['confidence']
        We provide compatibility by copying from track.ht if present.
        """
        ht = getattr(track, "ht", None)
        has_endpoints = getattr(track, "endpoints", None) is not None

        head = ht.head if (ht is not None and has_endpoints) else None
        tail = ht.tail if (ht is not None and has_endpoints) else None
        conf = ht.confidence if (ht is not None and has_endpoints) else 0.0

        if not hasattr(track, "features") or track.features is None:
            track.features = {}

        track.features["head"] = head
        track.features["tail"] = tail
        track.features["confidence"] = conf

    def process(self, frame_gray: np.ndarray, well_id: int) -> Tuple[np.ndarray, list]:
        wc = self.ctx[well_id]

        well_crop = crop_well(frame_gray, wc.roi)
        h, w = well_crop.shape[:2]
        r = w // 2

        if wc.region_mask is None or wc.region_mask.shape[:2] != (h, w):
            wc.region_mask = create_region_mask(
                shape=(h, w),
                radius=r,
                wall_width=self.wall_width_region,
            )

        _, fg_mask = extract_foreground(
            well_crop,
            wc.background,
            thresh_val=self.foreground_thresh_val,
        )

        detections = detect_blobs(fg_mask, region_mask=wc.region_mask)
        tracks = wc.tracker.update(detections)
        for t in tracks:
            self._attach_features_from_ht(t)


        annotated = annotate_region_boundaries(
            well_crop,
            wc.roi,
            wall_width=self.wall_width_draw,
            thickness=self.region_boundary_thickness,
        )
        annotated = annotate_tracks(annotated, tracks)


        if getattr(self, "debug", False):
            if annotated.ndim == 2:
                annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)

            status = getattr(wc.tracker, "_debug_det_status", {})
            counts = getattr(wc.tracker, "_debug_counts", {})

            # BGR colors
            COLOR_FILTERED = (0, 0, 255)  # red
            COLOR_TENTATIVE = (255, 0, 0) # blue
            COLOR_KEPT = (0, 255, 0)      # green

            for det in detections:
                x, y, w2, h2 = map(int, det.bbox)
                s = status.get(id(det), 0)

                if s == 2:
                    color = COLOR_KEPT
                elif s == 1:
                    color = COLOR_TENTATIVE
                else:
                    color = COLOR_FILTERED

                cv2.rectangle(annotated, (x, y), (x + w2, y + h2), color, 1)

            cv2.putText(
                annotated,
                f"raw={counts.get('raw', len(detections))} "
                f"kept={counts.get('kept', 0)} "
                f"filt={counts.get('filtered', 0)} "
                f"tent={counts.get('tentative', 0)} "
                f"conf={counts.get('confirmed', 0)} "
                f"active={counts.get('active_tracks', len(tracks))}",
                (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )



        return annotated, tracks
