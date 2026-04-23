from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from tqdm import tqdm

from Module.Tracking.Configuration import TrackingParameters
from Module.Tracking.utils import id_to_color
from Module.dataset.Dataset import TrackingDataset
from Module.dataset.video import open_dataset
from Module.dataset.Drawer import ROIDrawer, RectROIDrawer
from Module.imageprocessing.background import RollingMedianBackground
from Module.imageprocessing.foreground import extract_foreground_advanced
from Module.utils import (
    annotate_frame_metadata,
    annotate_tracks,
    create_region_mask,
    crop_well,
    display_frame,
    stitch_wells,
)
from Module.imageprocessing.well_processor import WellContext, WellProcessor
from Module.Tracking.Tracker import MultiOrganismTracker
from Module.Tracking.StateManager import TrackState


@dataclass
class AppConfig:
    video_dir: Path
    out_dir: Path

    frame_step: int = 1
    capture_fps: int = 2

    roi_path: Optional[Path] = None
    well_shape: str = "circle"

    max_workers: int = 4

    save_video: bool = False
    preview: bool = False
    out_video_name: str = "Annotated.mp4"
    video_write_every_n: int = 5
    video_fps_out: int = 30

    foreground_thresh_val: int = 20
    wall_width_region: int = 16
    region_boundary_thickness: int = 1

    debug: bool = False


class TrackingRunner:
    def __init__(self, cfg: AppConfig, params: TrackingParameters):
        self.cfg = cfg
        self.params = params

        self.cfg.out_dir.mkdir(parents=True, exist_ok=True)
        self.video_paths = sorted(self.cfg.video_dir.glob("*.mp4"))
        if not self.video_paths:
            raise RuntimeError(f"No videos in {self.cfg.video_dir}")

        if self.cfg.roi_path is None:
            cap = open_dataset(self.video_paths[0])
            ret, frame = cap.read()
            cap.release()
            drawer = (
                ROIDrawer(frame)
                if self.cfg.well_shape == "circle"
                else RectROIDrawer(frame)
            )
            self.wells = drawer.draw()
            drawer.save("AgarWell.npy")
        else:
            self.wells = ROIDrawer.load(self.cfg.roi_path)

        self.static_backgrounds = self._sample_backgrounds()
        self.contexts: List[WellContext] = []
        self.datasets: Dict[int, TrackingDataset] = {}

        for well_id, roi in enumerate(self.wells):
            tracker = MultiOrganismTracker(self.params)

            well_h, well_w = self.static_backgrounds[well_id].shape
            well_r = well_w // 2

            region_mask = create_region_mask(
                shape=(well_h, well_w),
                radius=well_r,  
                wall_width=self.cfg.wall_width_region,   
            )

            self.contexts.append(
                WellContext(
                    well_id=well_id,
                    roi=roi,
                    background=self.static_backgrounds[well_id],
                    tracker=tracker,
                    region_mask=region_mask,
                )
            )

            self.datasets[well_id] = TrackingDataset(
                str(self.cfg.out_dir / f"well_{well_id}.h5"),
                self.params,
            )

        self.processor = WellProcessor(
            self.contexts,
            foreground_thresh_val=self.cfg.foreground_thresh_val,
            wall_width_region=self.cfg.wall_width_region,
            region_boundary_thickness=self.cfg.region_boundary_thickness,
            min_length_for_skeleton=self.params.min_length_for_skeleton
        )

        self.out_video = None
        self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    def _sample_backgrounds(self) -> Dict[int, np.ndarray]:
        caps = [open_dataset(vp) for vp in self.video_paths[:20]]
        accum = {i: [] for i in range(len(self.wells))}

        for cap in caps:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            idxs = np.random.choice(total, size=min(50, total), replace=False)

            for idx in idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if not ret:
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                for i, roi in enumerate(self.wells):
                    accum[i].append(crop_well(gray, roi))
            cap.release()

        return {
            i: np.median(np.stack(crops), axis=0).astype(np.uint8)
            for i, crops in accum.items()
        }

    def _write_tracks(self, video_idx: int, frame_idx: int, tracks_per_well):
        for well_id, tracks in enumerate(tracks_per_well):
            ds = self.datasets[well_id]
            for t in tracks:
                if t.state.current != TrackState.ACTIVE:
                    continue

                head = t.ht_state.head if t.ht_state.head is not None else None
                tail = t.ht_state.tail if t.ht_state.tail is not None else None
                ht_conf = float(t.ht_state.confidence)

                ds.append_track_row(
                    video_idx=video_idx,
                    frame_idx=frame_idx,
                    track_id=t.id,
                    centroid=t.centroid.astype(float),
                    area=t.area,
                    head=head,
                    tail=tail,
                    ht_conf=ht_conf,)


    def _show_debug_view(
        self,
        annotated_wells: Tuple[np.ndarray, ...],
        tracks_per_well: Tuple[list, ...],
        video_idx: int,
        frame_idx: int,
    ) -> None:

        debug_panels = []

        for well_id, wc in enumerate(self.contexts):
            well_crop = crop_well(self.last_frame_gray, wc.roi)
            bg = wc.background

            _, fg = extract_foreground_advanced(
                img=well_crop,
                background=bg,
                wall_mask=None,
                thresh_val=self.cfg.foreground_thresh_val,
                use_rolling=False,
            )

            well_bgr = cv2.cvtColor(well_crop, cv2.COLOR_GRAY2BGR)
            bg_bgr = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
            fg_bgr = cv2.cvtColor(fg, cv2.COLOR_GRAY2BGR)

            track_colors = {
                t.id: id_to_color(t.id) for t in wc.tracker.active_tracks.values()
            }

            color_overlay = well_bgr.copy()

            for t in wc.tracker.active_tracks.values():
                det = getattr(t, "last_detection", None)
                if det is None:
                    continue

                x, y, w, h = det.bbox
                mask = det.mask.astype(bool)
                color = track_colors[t.id]

                roi = color_overlay[y : y + h, x : x + w]
                roi[mask] = 0.6 * roi[mask] + 0.4 * np.array(color)

            well_bgr[:] = color_overlay

            y0 = 15
            for t in wc.tracker.active_tracks.values():
                state = t.state.current.name
                reason = t.state.history[-1].reason
                age = t.age
                missed = t.missed

                label = f"ID {t.id} | age={age} miss={missed} | {state} | {reason}"

                cv2.putText(
                    well_bgr,
                    label,
                    (5, y0),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                y0 += 14

                c = t.centroid.astype(int)
                cv2.circle(well_bgr, tuple(c), 3, (0, 255, 0), -1)

                p = t.predicted.astype(int)
                cv2.circle(well_bgr, tuple(p), 3, (0, 0, 255), -1)

                cv2.line(well_bgr, tuple(c), tuple(p), (255, 0, 0), 1)

                if t.ht_state.head is not None:
                    cv2.circle(
                        well_bgr,
                        tuple(t.ht_state.head.astype(int)),
                        4,
                        (0, 0, 255),  # head = red
                        -1,
                    )

                if t.ht_state.tail is not None:
                    cv2.circle(
                        well_bgr,
                        tuple(t.ht_state.tail.astype(int)),
                        4,
                        (255, 0, 0),  # tail = blue
                        -1,
                    )


            for det in wc.last_detections:  # type: ignore
                x, y, w, h = det.bbox

                cv2.rectangle(
                    well_bgr,
                    (x, y),
                    (x + w, y + h),
                    (255, 255, 0),
                    1,
                )

                cv2.putText(
                    well_bgr,
                    det.fate,
                    (x, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (255, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

            panel = np.hstack([well_bgr, fg_bgr, bg_bgr])
            debug_panels.append(panel)

        debug_view = np.vstack(debug_panels)
        display_frame("DEBUG", debug_view)

    def run(self):
        if self.cfg.debug:
            max_workers = 1
        else:
            max_workers = min(self.cfg.max_workers, len(self.wells))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for video_idx, video_path in enumerate(self.video_paths):
                # if video_idx <= 100:
                #     video_idx += 1
                #     continue
                
                cap = open_dataset(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_idx = 0

                pbar = tqdm(total=total_frames, desc=f"Video {video_idx}", unit="frame")

                start_time = time.time()
                processed_frames = 0

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    self.last_frame_gray = gray

                    if frame_idx % self.cfg.frame_step != 0:
                        frame_idx += 1
                        pbar.update(1)
                        continue

                    results = list(
                        executor.map(
                            self.processor.process,
                            [gray] * len(self.wells),
                            range(len(self.wells)),
                        )
                    )

                    annotated_wells, tracks_per_well = zip(*results)

                    self._write_tracks(video_idx, frame_idx, tracks_per_well)

                    if self.cfg.debug:
                        self._show_debug_view(
                            annotated_wells,
                            tracks_per_well,
                            video_idx,
                            frame_idx,
                        )

                    stitched = stitch_wells(
                        annotated_wells,  # type: ignore
                        annotated_wells[0].shape[:2],
                    )
                    stitched = annotate_frame_metadata(
                        stitched,
                        frame_idx=frame_idx,
                        video_idx=video_idx,
                        num_worms=sum(len(t) for t in tracks_per_well),
                        fps=self.cfg.capture_fps,
                    )

                    if self.cfg.preview:
                        display_frame("Tracking", stitched)

                    if self.cfg.save_video:
                        if self.out_video is None:
                            h, w = stitched.shape[:2]
                            self.out_video = cv2.VideoWriter(
                                str(self.cfg.out_dir / self.cfg.out_video_name),
                                self.fourcc,
                                self.cfg.video_fps_out,
                                (w, h),
                            )
                        if frame_idx % self.cfg.video_write_every_n == 0:
                            self.out_video.write(stitched)

                    processed_frames += 1
                    elapsed = time.time() - start_time

                    if elapsed > 1.0:
                        fps = processed_frames / elapsed
                        pbar.set_postfix(fps=f"{fps:.2f}")
                        start_time = time.time()
                        processed_frames = 0

                    frame_idx += 1
                    pbar.update(1)

                cap.release()
                pbar.close()

        for ds in self.datasets.values():
            ds.close()

        if self.out_video is not None:
            self.out_video.release()

        cv2.destroyAllWindows()
