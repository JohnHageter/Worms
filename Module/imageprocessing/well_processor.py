from __future__ import annotations
from dataclasses import dataclass
from os import path
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

from Module.Tracking.Tracker import MultiOrganismTracker
from Module.imageprocessing.foreground import extract_foreground_advanced
from Module.Tracking.Detection import BlobDetection, detect_blobs
from Module.utils import annotate_tracks, crop_well, create_region_mask, annotate_region_boundaries


@dataclass
@dataclass
class WellContext:
    well_id: int
    roi: np.ndarray
    background: np.ndarray     
    tracker: MultiOrganismTracker
    last_detections = List[BlobDetection]
    region_mask: np.ndarray



class WellProcessor:
    """
    Stateless per-frame well processor.
    Image -> foreground -> detections -> tracker.update.
    """

    def __init__(
        self,
        contexts: List[WellContext],
        *,
        foreground_thresh_val: int,
        wall_width_region: int,
        region_boundary_thickness: int,
        min_length_for_skeleton: int
    ):
        self.contexts = contexts
        self.foreground_thresh_val = foreground_thresh_val
        self.wall_width_region = wall_width_region
        self.region_boundary_thickness = region_boundary_thickness
        self.min_length_for_skeleton = min_length_for_skeleton

    def process(self, frame_gray: np.ndarray, well_id: int):
        wc = self.contexts[well_id]

        well_crop = crop_well(frame_gray, wc.roi)

        if wc.region_mask is None or wc.region_mask.shape != well_crop.shape:
            wc.region_mask = create_region_mask(
                well_crop.shape,
                radius=well_crop.shape[1] // 2,
                wall_width=self.wall_width_region,
            )

        _, fg = extract_foreground_advanced(
            img=well_crop,
            background=wc.background,
            wall_mask=None,
            thresh_val=self.foreground_thresh_val,
            use_rolling=False,  
        )

        h, w = fg.shape
        cy, cx = h // 2, w // 2
        Y, X = np.ogrid[:h, :w]
        radius = min(h, w) // 2
        fg[
            np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) > (radius - self.wall_width_region)
        ] = 0


        detections = detect_blobs(
            fg,
            region_mask=wc.region_mask,
            min_area=wc.tracker.p.min_blob_area,
        )
                
        wc.last_detections = detections # type: ignore

        # for d in detections[:3]:
        #     print("DETECTION bbox:", d.bbox)

        all_tracks = wc.tracker.update(detections)

        # for t in wc.tracker.active_tracks.values():
        #     print("TRACK centroid:", t.centroid)


        active_tracks = [t for t in all_tracks if t.state.current.name == "ACTIVE"]


        annotated = annotate_region_boundaries(
            well_crop,
            wall_width=self.wall_width_region,
            thickness=self.region_boundary_thickness,
        )
        
        annotated = annotate_tracks(
            annotated,
            all_tracks
        )

        return annotated, active_tracks
