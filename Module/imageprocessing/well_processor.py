from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import numpy as np

from Module.Tracking.Tracker import MultiOrganismTracker
from Module.imageprocessing.foreground import extract_foreground_advanced, detect_blobs
from Module.imageprocessing.background import RollingMedianBackground
from Module.utils import (
    annotate_region_boundaries,
    annotate_tracks,
    create_region_mask,
    crop_well,
)


@dataclass
class WellContext:
    well_id: int
    roi: np.ndarray  # (x, y, r) well definition
    rolling_bg: RollingMedianBackground 
    tracker: MultiOrganismTracker
    region_mask: Optional[np.ndarray] = None  # cached per well


class WellProcessor:
    """
    Processes a single well for one frame using:
      - Rolling median background
      - Wall edge detection
      - Improved blob filtering
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
        debug: bool = False,
    ):
        self.ctx = well_contexts
        self.foreground_thresh_val = foreground_thresh_val
        self.wall_width_region = wall_width_region
        self.wall_width_draw = wall_width_draw
        self.region_boundary_thickness = region_boundary_thickness
        self.debug = debug

    @staticmethod
    def _attach_features_from_ht(track) -> None:
        """Copy head/tail from track.ht into track.features for annotation."""
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
        wall_mask = (wc.region_mask == 2).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        wall_mask = cv2.erode(wall_mask, kernel, iterations=3)

        _, fg_mask = extract_foreground_advanced(
            img=well_crop,
            background=wc.rolling_bg, 
            wall_mask=wall_mask,
            thresh_val=self.foreground_thresh_val,
            blur_ksize=5,
            min_area=10,
            use_rolling=True,
        )
        
        fg_pixels = np.count_nonzero(fg_mask)
        print(f"Well {well_id}: fg_mask non-zero pixels = {fg_pixels}")

        detections = detect_blobs(fg_mask, region_mask=wc.region_mask)
        print(f"Well {well_id}: {len(detections)} blobs detected")
        
        cv2.imwrite(f"debug_well{well_id}_frame{1}_fgmask.png", fg_mask)
        cv2.imwrite(f"debug_well{well_id}_frame{1}_crop.png", well_crop)

        if not detections:
            # No blobs → nothing to track
            return well_crop, []

        tracks = wc.tracker.update(detections)
        print(f"Well {well_id}: {len(tracks)} tracks after update")


        for i, d in enumerate(detections):
            area = np.count_nonzero(d.mask)
            print(f"  blob {i}: area={area}, region={d.region}, bbox={d.bbox}")

        tracks = wc.tracker.update(detections)
        
        if tracks:
            print(f"Well {well_id}, first track centroid = {tracks[0].centroid}")
            print(f"Crop shape = {well_crop.shape}")
        
        for t in tracks:
            self._attach_features_from_ht(t)
            print(f"Track {t.id}: head = {t.ht.head}, tail = {t.ht.tail}")

        annotated = annotate_region_boundaries(
            well_crop,
            wc.roi,
            wall_width=self.wall_width_draw,
            thickness=self.region_boundary_thickness,
        )
        annotated = annotate_tracks(annotated, tracks)

        if self.debug:
            if annotated.ndim == 2:
                annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)

            status = getattr(wc.tracker, "_debug_det_status", {})
            counts = getattr(wc.tracker, "_debug_counts", {})

            COLOR_FILTERED = (0, 0, 255)  # red
            COLOR_TENTATIVE = (255, 0, 0)  # blue
            COLOR_KEPT = (0, 255, 0)  # green

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
