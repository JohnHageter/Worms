from Module.imageprocessing.foreground import extract_foreground
from Module.utils import (
    annotate_detections,
    annotate_region_boundaries,
    annotate_tracks,
    create_region_mask,
    crop_well,
)


class WellProcessor:
    def __init__(self, well_id, well_roi, tracker, background):
        self.well_id = well_id
        self.well = well_roi
        self.tracker = tracker
        self.background = background

    def process_frame(self, frame):
        well_crop = crop_well(frame, self.well)
        h, w = well_crop.shape
        r = w // 2

        region_mask = create_region_mask((h, w), r, wall_width=16)

        fg, fg_mask = extract_foreground(
            well_crop,
            self.background,
            thresh_val=20,
        )

        detections = self.tracker.detect(
            fg_mask,
            mask=fg_mask,
            region_mask=region_mask,
        )

        active_tracks = self.tracker.update(detections)

        annotated = annotate_region_boundaries(
            well_crop,
            self.well,
            wall_width=16,
        )
        annotated = annotate_detections(annotated, detections)
        annotated = annotate_tracks(annotated, active_tracks)

        return {
            "crop": well_crop,
            "annotated": annotated,
            "detections": detections,
            "tracker": self.tracker,
        }
