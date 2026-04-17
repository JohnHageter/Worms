from __future__ import annotations

from typing import Callable, List, Tuple, Any, Optional

import numpy as np
import cv2

from Module.dataset.h5 import TrackRow


def _to_int_pt(x: float, y: float) -> Tuple[int, int]:
    return int(round(float(x))), int(round(float(y)))


def roi_origin_xy(roi: Any) -> Tuple[int, int]:
    """
      - circle: (cx, cy, r)  -> origin (cx-r, cy-r)
      - rect:   (x, y, w, h) -> origin (x, y)
    """
    if isinstance(roi, np.ndarray):
        roi = roi.tolist()

    roi = list(map(int, roi))

    if len(roi) == 3:
        cx, cy, r = roi
        return cx - r, cy - r
    if len(roi) == 4:
        x, y, w, h = roi
        return x, y

    raise ValueError(f"Unsupported ROI format: {roi!r}")


class MultiWellAnnotator:
    def __init__(
        self,
        wells: Any,
        *,
        crop_well_fn: Callable[[np.ndarray, Any], np.ndarray],
        stitch_wells_fn: Callable[[List[np.ndarray], Tuple[int, int]], np.ndarray],
        annotate_frame_fn: Optional[
            Callable[[np.ndarray, int, int, int, int], np.ndarray]
        ] = None,
        annotate_region_boundaries_fn: Optional[Callable[..., np.ndarray]] = None,
        wall_width_draw: int = 8,
        region_boundary_thickness: int = 1,
        capture_fps: int = 2,
        draw_ids: bool = True,
        draw_head_tail: bool = True,
        coords_are_full_frame: bool = True,
    ):
        """
        coords_are_full_frame:
            - True: TrackView coordinates are in full-frame space (typical during tracking)
                    so we subtract ROI crop origin before drawing.
            - False: TrackView coordinates are already crop-relative, so no subtraction.
        """
        self.wells = wells
        self.crop_well = crop_well_fn
        self.stitch_wells = stitch_wells_fn
        self.annotate_frame = annotate_frame_fn
        self.annotate_region_boundaries = annotate_region_boundaries_fn
        self.wall_width_draw = wall_width_draw
        self.region_boundary_thickness = region_boundary_thickness
        self.capture_fps = capture_fps
        self.draw_ids = draw_ids
        self.draw_head_tail = draw_head_tail
        self.coords_are_full_frame = coords_are_full_frame

        # colors (BGR)
        self.color_centroid = (0, 255, 255)  # yellow
        self.color_head = (255, 0, 0)  # blue
        self.color_tail = (0, 0, 255)  # red
        self.color_text = (255, 255, 255)  # white

    def annotate_frame_multiwell(
        self,
        frame_gray: np.ndarray,
        tracks_by_well: List[List[TrackRow]],
        *,
        video_idx: int,
        frame_idx: int,
    ) -> np.ndarray:
        annotated_wells: List[np.ndarray] = []
        total = 0

        for well_id, roi in enumerate(self.wells):
            crop = self.crop_well(frame_gray, roi)

            # ensure BGR for drawing
            if crop.ndim == 2:
                crop_bgr = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
            else:
                crop_bgr = crop.copy()

            if self.annotate_region_boundaries is not None:
                crop_bgr = self.annotate_region_boundaries(
                    crop_bgr,
                    roi,
                    wall_width=self.wall_width_draw,
                    thickness=self.region_boundary_thickness,
                )

            rows = tracks_by_well[well_id]
            total += len(rows)

            crop_bgr = self._draw_tracks_on_crop(crop_bgr, roi, rows)
            annotated_wells.append(crop_bgr)

        h, w = annotated_wells[0].shape[:2]
        stitched = self.stitch_wells(annotated_wells, (h, w))

        if self.annotate_frame is not None:
            stitched = self.annotate_frame(
                stitched, frame_idx, video_idx, total, self.capture_fps
            )

        return stitched

    def _draw_tracks_on_crop(
        self, crop_bgr: np.ndarray, roi: Any, rows: List[TrackRow]
    ) -> np.ndarray:
        ox, oy = roi_origin_xy(roi) if self.coords_are_full_frame else (0, 0)

        for tr in rows:
            cx, cy = _to_int_pt(tr.cx - ox, tr.cy - oy)
            cv2.circle(crop_bgr, (cx, cy), 2, self.color_centroid, -1)

            if self.draw_ids:
                cv2.putText(
                    crop_bgr,
                    str(tr.track_id),
                    (cx + 4, cy - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    self.color_text,
                    1,
                    cv2.LINE_AA,
                )

            if self.draw_head_tail:
                if tr.hx >= 0 and tr.hy >= 0:
                    hx, hy = _to_int_pt(tr.hx - ox, tr.hy - oy)
                    cv2.circle(crop_bgr, (hx, hy), 3, self.color_head, -1)

                if tr.tx >= 0 and tr.ty >= 0:
                    tx, ty = _to_int_pt(tr.tx - ox, tr.ty - oy)
                    cv2.circle(crop_bgr, (tx, ty), 3, self.color_tail, -1)

        return crop_bgr


class LiveTrackProvider:
    """
    Converts in-memory Track objects (from live tracking) into TrackRow records.
    """

    def tracks_by_well(self, tracks_per_well: List[List[Any]]) -> List[List[TrackRow]]:
        out: List[List[TrackRow]] = []

        for tracks in tracks_per_well:
            rows: List[TrackRow] = []

            for t in tracks:
                cx, cy = float(t.centroid[0]), float(t.centroid[1])
                region = int(getattr(t, "region", 0))

                ht = getattr(t, "ht", None)
                has_ep = getattr(t, "endpoints", None) is not None

                if (
                    ht is not None
                    and has_ep
                    and ht.head is not None
                    and ht.tail is not None
                ):
                    hx, hy = float(ht.head[0]), float(ht.head[1])
                    tx, ty = float(ht.tail[0]), float(ht.tail[1])
                    conf = float(getattr(ht, "confidence", 0.0))
                else:
                    hx = hy = tx = ty = -1.0
                    conf = 0.0

                rows.append(
                    TrackRow(
                        track_id=int(t.id),
                        cx=cx,
                        cy=cy,
                        hx=hx,
                        hy=hy,
                        tx=tx,
                        ty=ty,
                        region=region,
                        conf=conf,
                    )
                )

            out.append(rows)

        return out


class H5TrackProvider:
    """
    Converts H5 reader rows into TrackRow records.
    Expects each reader implements:
        rows_for_frame(frame_idx) -> iterable of rows
    Where row has:
        track_id, cx, cy, hx, hy, tx, ty, region, conf
    """

    def __init__(self, well_readers: List[Any]):
        self.readers = well_readers

    def tracks_by_well(self, frame_idx: int) -> List[List[TrackRow]]:
        out: List[List[TrackRow]] = []
        for r in self.readers:
            rows: List[TrackRow] = []
            for tr in r.rows_for_frame(frame_idx):
                rows.append(
                    TrackRow(
                        track_id=int(tr.track_id),
                        cx=float(tr.cx),
                        cy=float(tr.cy),
                        hx=float(tr.hx),
                        hy=float(tr.hy),
                        tx=float(tr.tx),
                        ty=float(tr.ty),
                        region=int(tr.region),
                        conf=float(tr.conf),
                    )
                )
            out.append(rows)
        return out
