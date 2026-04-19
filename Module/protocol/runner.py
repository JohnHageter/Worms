from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Optional

import numpy as np
import cv2

from Module.Tracking.Configuration import TrackingParameters
from Module.Tracking.Tracker import MultiOrganismTracker
from Module.dataset.Dataset import TrackingDataset
from Module.dataset.video import open_dataset
from Module.dataset.Drawer import ROIDrawer, RectROIDrawer
from Module.imageprocessing.background import sample_per_well_backgrounds
from Module.imageprocessing.background import RollingMedianBackground
from Module.utils import (
    annotate_frame_metadata,
    display_frame,
    stitch_wells,
)
from Module.imageprocessing.well_processor import (
    WellContext,
    WellProcessor,
)


@dataclass
class AppConfig:
    video_dir: Path
    out_dir: Path
    frame_step: int = 1
    capture_fps: int = 2
    roi_path: Optional[Path] = None
    well_shape: Optional[str] = "circle"

    max_workers: int = 8

    save_video: bool = False
    preview: bool = False
    out_video_name: str = "Annotated.mp4"
    video_write_every_n: int = 5
    video_fps_out: int = 30

    foreground_thresh_val: int = 20
    wall_width_region: int = 16
    wall_width_draw: int = 8
    region_boundary_thickness: int = 1
    coords_are_full_frame: bool = True

    debug: bool = False


class TrackingRunner:
    def __init__(self, cfg: AppConfig, params: TrackingParameters):
        self.cfg = cfg
        self.params = params

        self.cfg.out_dir.mkdir(parents=True, exist_ok=True)
        self.video_paths = sorted(self.cfg.video_dir.glob("*.mp4"))

        if not self.video_paths:
            raise RuntimeError(f"No videos found in: {self.cfg.video_dir}")

        if self.cfg.roi_path is None:
            cap = open_dataset(self.video_paths[0])
            ret, frame = cap.read()

            if self.cfg.well_shape == "circle":
                drawer = ROIDrawer(frame)
                self.wells = drawer.draw()
            elif self.cfg.well_shape == "square":
                drawer = RectROIDrawer(frame)
                self.wells = drawer.draw()
        else:
            self.wells = ROIDrawer.load(self.cfg.roi_path)

        # 1. Sample a static background (only for seeding the rolling background)
        self.static_backgrounds = self._sample_backgrounds()

        # 2. Build contexts (now with RollingMedianBackground)
        self.contexts = self._build_contexts()

        # 3. Instantiate the improved WellProcessor
        self.processor = WellProcessor(
            self.contexts,
            foreground_thresh_val=self.cfg.foreground_thresh_val,
            wall_width_region=self.cfg.wall_width_region,
            wall_width_draw=self.cfg.wall_width_draw,
            region_boundary_thickness=self.cfg.region_boundary_thickness,
            debug=self.cfg.debug,
        )

        self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.out_video = None

        self.start_time = time.perf_counter()
        self.proc_frames = 0

    def _sample_backgrounds(self) -> Dict[int, np.ndarray]:
        """Sample a clean static background from the first video."""
        cap_bg = open_dataset(self.video_paths[0])
        bgs = sample_per_well_backgrounds(cap_bg, self.wells, n_frames=1500)
        cap_bg.release()
        return bgs

    def _build_contexts(self) -> List[WellContext]:
        contexts: List[WellContext] = []
        for well_id in range(len(self.wells)):
            # Create the improved tracker
            tracker = MultiOrganismTracker(self.params)

            # Create a rolling background per well, seeded with the static background
            rolling_bg = RollingMedianBackground(history=300, update_every=10)
            rolling_bg.background = self.static_backgrounds[well_id].copy()

            # Build WellContext (now expects rolling_bg, not background)
            wc = WellContext(
                well_id=well_id,
                roi=self.wells[well_id],
                rolling_bg=rolling_bg,  # <-- key change
                tracker=tracker,
                region_mask=None,
            )
            # Attach dataset (remains the same)
            h5_path = self.cfg.out_dir / f"well_{well_id}.h5"
            wc.dataset = TrackingDataset(str(h5_path), self.params)  # type: ignore[attr-defined]
            contexts.append(wc)
        return contexts

    def _write_tracks(
        self, video_idx: int, frame_idx: int, tracks_per_well: Tuple[list, ...]
    ) -> None:
        for well_id, tracks in enumerate(tracks_per_well):
            ds = self.contexts[well_id].dataset  # type: ignore[attr-defined]

            for t in tracks:
                centroid = np.asarray(t.centroid, dtype=float)
                region_code = int(getattr(t, "region", 0))

                ht = getattr(t, "ht", None)
                has_endpoints = getattr(t, "endpoints", None) is not None

                head = ht.head if (ht is not None and has_endpoints) else None
                tail = ht.tail if (ht is not None and has_endpoints) else None
                conf = ht.confidence if (ht is not None and has_endpoints) else 0.0

                ds.append_track_row(
                    video_idx=video_idx,
                    frame_idx=frame_idx,
                    track_id=t.id,
                    centroid=centroid,
                    region=region_code,
                    head=head,
                    tail=tail,
                    ht_conf=conf,
                )

    def _stitch_display_save(
        self,
        annotated_wells: Tuple[np.ndarray, ...],
        tracks_per_well: Tuple[list, ...],
        video_idx: int,
        frame_idx: int,
    ) -> None:
        crop_h, crop_w = annotated_wells[0].shape[:2]
        stitched = stitch_wells(annotated_wells, (crop_h, crop_w))

        stitched = annotate_frame_metadata(
            stitched,
            frame_idx,
            video_idx,
            sum(len(tr) for tr in tracks_per_well),
            self.cfg.capture_fps,
        )

        if self.cfg.save_video:
            if self.out_video is None:
                sH, sW = stitched.shape[:2]
                out_path = self.cfg.out_dir / self.cfg.out_video_name
                self.out_video = cv2.VideoWriter(
                    str(out_path),
                    self.fourcc,
                    self.cfg.video_fps_out,
                    (sW, sH),
                )
            if frame_idx % self.cfg.video_write_every_n == 0:
                self.out_video.write(stitched)

        if self.cfg.preview and frame_idx % 5 == 0:
            display_frame("All Wells", stitched)

    def run(self) -> None:
        max_workers = min(self.cfg.max_workers, len(self.wells))

        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for video_idx, video_path in enumerate(self.video_paths):
                    print(f"\nProcessing video {video_idx}: {video_path.name}")

                    cap = open_dataset(video_path)
                    frame_idx = 0

                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break

                        if frame.ndim == 3:
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                        if frame_idx % self.cfg.frame_step != 0:
                            frame_idx += 1
                            continue

                        # Process wells in parallel (each well uses its own rolling background)
                        well_results = list(
                            executor.map(
                                self.processor.process,
                                [frame] * len(self.wells),
                                range(len(self.wells)),
                            )
                        )

                        annotated_wells, tracks_per_well = zip(*well_results)

                        self._write_tracks(video_idx, frame_idx, tracks_per_well)
                        self._stitch_display_save(
                            annotated_wells, tracks_per_well, video_idx, frame_idx
                        )

                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                            print("Skipping to next video...")
                            break

                        frame_idx += 1
                        self.proc_frames += 1
                        elapsed = time.perf_counter() - self.start_time
                        fps = self.proc_frames / max(elapsed, 1e-9)
                        print(
                            f"Video {video_idx}, frame {frame_idx}, proc/sec {fps:.2f}"
                        )

                    cap.release()

        finally:
            for wc in self.contexts:
                wc.dataset.close()  # type: ignore[attr-defined]

            if self.out_video is not None:
                self.out_video.release()

            cv2.destroyAllWindows()
            print("\nDone.")
