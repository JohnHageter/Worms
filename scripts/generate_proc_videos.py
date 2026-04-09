from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import cv2
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from Module.dataset.video import open_dataset
from Module.Worms import WormTracker
from Module.imageprocessing.background import sample_per_well_backgrounds
from Module.imageprocessing.foreground import extract_foreground
from Module.detection.Drawer import ROIDrawer
from Module.utils import (
    crop_well,
    create_region_mask,
    annotate_region_boundaries,
    annotate_detections,
    annotate_tracks,
    stitch_wells,
)

# ============================================================
# CONFIG
# ============================================================
VIDEO_DIR = Path("D:/Tracking_Data/Sachi/T3/week_1/")
OUT_DIR = Path("D:/Tracking_Data/Sachi/T3/debug_out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FPS = 60
FRAME_STEP = 1
N_WORKERS = 4

WELLS = ROIDrawer.load("wells.npy")

tracker_params = dict(
    max_dist=500,
    max_missed=1000,
    min_new_area=30,
    min_new_dist=50,
    area_weight=50,
    min_skel_length=10,
    min_motion=5,
    min_area=10,
)

# ============================================================
# HELPERS
# ============================================================


def to_bgr(img):
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def pad_to_size(img, H, W):
    h, w = img.shape[:2]
    return cv2.copyMakeBorder(
        img,
        (H - h) // 2,
        H - h - (H - h) // 2,
        (W - w) // 2,
        W - w - (W - w) // 2,
        cv2.BORDER_CONSTANT,
        value=0,
    )


# ============================================================
# INIT TRACKERS & BACKGROUNDS
# ============================================================

TRACKERS = [WormTracker(i, **tracker_params) for i in range(len(WELLS))]

cap_bg = open_dataset(sorted(VIDEO_DIR.glob("*.mp4"))[0])
WELL_BACKGROUNDS = sample_per_well_backgrounds(cap_bg, WELLS, n_frames=500)
cap_bg.release()

# ============================================================
# VIDEO WRITERS (created lazily)
# ============================================================

writers = {}
fourcc = cv2.VideoWriter_fourcc(*"mp4v")


def init_writers(example_frames):
    for k, img in example_frames.items():
        h, w = img.shape[:2]
        writers[k] = cv2.VideoWriter(
            str(OUT_DIR / f"{k}.mp4"),
            fourcc,
            FPS,
            (w, h),
        )


def write_stages(stage_frames):
    for k, img in stage_frames.items():
        writers[k].write(img)


# ============================================================
# PER-WELL PROCESSING
# ============================================================


def process_well(frame, well_id, well):
    crop = crop_well(frame, well)
    h, w = crop.shape
    r = w // 2

    region_mask = create_region_mask((h, w), r, wall_width=16)
    bg = WELL_BACKGROUNDS[well_id]

    fg, fg_mask = extract_foreground(crop, bg, thresh_val=20)

    dets = TRACKERS[well_id].detect(fg_mask, mask=fg_mask, region_mask=region_mask)
    tracks = TRACKERS[well_id].update(dets)

    annotated = annotate_region_boundaries(crop, well, wall_width=16)
    annotated = annotate_detections(annotated, dets)
    annotated = annotate_tracks(annotated, tracks)

    return dict(
        raw=to_bgr(crop),
        background=to_bgr(bg),
        foreground=to_bgr(fg),
        mask=to_bgr(fg_mask),
        annotated=to_bgr(annotated),
    )


# ============================================================
# MAIN LOOP
# ============================================================

video_paths = sorted(VIDEO_DIR.glob("*.mp4"))

with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
    for video_idx, video_path in enumerate(video_paths):
        print(f"Processing {video_path.name}")

        cap = open_dataset(video_path)
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame.ndim == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if frame_idx % FRAME_STEP != 0:
                frame_idx += 1
                continue

            per_well = list(
                executor.map(
                    process_well, [frame] * len(WELLS), range(len(WELLS)), WELLS
                )
            )

            # stitch each stage
            stage_names = per_well[0].keys()
            stitched = {}

            for stage in stage_names:
                imgs = [pw[stage] for pw in per_well]
                H = max(i.shape[0] for i in imgs)
                W = max(i.shape[1] for i in imgs)
                imgs = [pad_to_size(i, H, W) for i in imgs]
                stitched[stage] = stitch_wells(imgs, (H, W))

            if not writers:
                init_writers(stitched)

            write_stages(stitched)

            frame_idx += 1

        cap.release()

for w in writers.values():
    w.release()
