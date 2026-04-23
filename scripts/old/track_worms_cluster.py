from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import cv2
import h5py
import time
from concurrent.futures import ThreadPoolExecutor
from Module.imageprocessing.foreground import extract_foreground
from Module.Tracking.Tracker import WormTracker
from Module.dataset.video import open_dataset
from Module.dataset.Drawer import ROIDrawer
from Module.imageprocessing.background import sample_per_well_backgrounds
from Module.utils import (
    annotate_detections,
    annotate_frame_metadata,
    annotate_region_circles,
    annotate_tracks,
    create_region_mask,
    crop_well,
    display_frame,
    stitch_wells,
    append_row,
)

# Global input settings
VIDEO_DIR = Path("/scratch/jwhageter/Worm_Tracking/Sachi/T3/week_1/")
DATA_OUT_DIR = Path("/scratch/jwhageter/Worm_Tracking/Sachi/T3/week_1_out")
FRAME_STEP = 1  # process every nth frame
CAPTURE_FPS = 2
WELLS = ROIDrawer.load("/scratch/jwhageter/Worm_Tracking/worms/wells.npy")

video_paths = sorted(VIDEO_DIR.glob("*.mp4"))
DATA_OUT_DIR.mkdir(parents=True, exist_ok=True)


# Tracker settings
WELL_BACKGROUNDS = {}
cap_bg = open_dataset(video_paths[0])
WELL_BACKGROUNDS = sample_per_well_backgrounds(
    cap_bg,
    WELLS,
    n_frames=500,
)
cap_bg.release()


N_WORKERS = min(8, len(WELLS))
tracker_params = dict(
    max_dist=int(500),
    max_missed=1000,
    min_new_area=int(30),
    min_new_dist=int(50),
    area_weight=50,
    min_skel_length=int(10),
    min_motion=5,
    min_area=int(10),
)

TRACKERS = [WormTracker(well_id=i, **tracker_params) for i in range(len(WELLS))]


#### Dataset creation
WELL_H5 = {}


for well_id in range(len(WELLS)):
    h5_path = DATA_OUT_DIR / f"well_{well_id}.h5"
    h5 = h5py.File(h5_path, "w")

    # MAIN WORM
    # [video_idx, frame_idx, track_id, well_id,
    #  cx, cy, hx, hy, tx, ty,
    #  region, age, is_primary]
    h5.create_dataset(
        "main",
        shape=(0, 13),
        maxshape=(None, 13),
        dtype="i4",
        chunks=(1024, 13),
    )

    # FISSION WORM
    # [video_idx, frame_idx, track_id, well_id,
    #  cx, cy, area, region, age]
    h5.create_dataset(
        "fission",
        shape=(0, 9),
        maxshape=(None, 9),
        dtype="i4",
        chunks=(1024, 9),
    )

    WELL_H5[well_id] = h5


# Video creation
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out_video = None


# Debug
start_time = time.perf_counter()
proc_frames = 0


# MAIN LOOP
def track_well(frame, well_id, well):
    well_crop = crop_well(frame, well)
    h, w = well_crop.shape
    r = w // 2

    region_mask = create_region_mask(shape=(h, w), radius=r, wall_width=16)

    well_bg = WELL_BACKGROUNDS[well_id]

    fg, fg_mask = extract_foreground(
        well_crop,
        well_bg,
        thresh_val=20,  # adjust per your data
    )

    # ---- Detect worms using mask ----
    all_dets = TRACKERS[well_id].detect(
        fg_mask,
        mask=fg_mask,
        region_mask=region_mask,
    )

    active_tracks = TRACKERS[well_id].update(all_dets)

    annotated_well_image = annotate_region_circles(
        well_crop,
        r,
        wall_width=16,
    )

    annotated_well_image = annotate_detections(
        annotated_well_image,
        all_dets,
    )

    annotated_well_image = annotate_tracks(
        annotated_well_image,
        active_tracks,
    )

    return well_crop, annotated_well_image, all_dets


try:
    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:

        for video_idx, video_path in enumerate(video_paths):
            print(f"\nProcessing video {video_idx}: {video_path.name}")

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

                # -------- Parallel well processing --------
                well_data = list(
                    executor.map(
                        track_well,
                        [frame] * len(WELLS),
                        range(len(WELLS)),
                        WELLS,
                    )
                )

                well_crops, annotated_well_images, all_dets = zip(*well_data)

                for well_id, dets in enumerate(all_dets):
                    tracker = TRACKERS[well_id]
                    h5 = WELL_H5[well_id]

                    main_ds = h5["main"]
                    fiss_ds = h5["fission"]

                    primary = tracker.primary_track

                    # ------------------
                    # Save MAIN worm
                    # ------------------
                    if primary is None or primary.centroid is None:
                        row = [
                            video_idx,
                            frame_idx,
                            -1,  # track_id
                            well_id,
                            -1,
                            -1,  # cx, cy
                            -1,
                            -1,  # hx, hy
                            -1,
                            -1,  # tx, ty
                            -1,  # region
                            -1,  # age
                            1,  # is_primary (always 1)
                        ]
                    else:
                        cx, cy = map(int, primary.centroid)
                        hx, hy = (
                            (-1, -1) if primary.head is None else map(int, primary.head)
                        )
                        tx, ty = (
                            (-1, -1) if primary.tail is None else map(int, primary.tail)
                        )

                        region = 1 if primary.region == "core" else 2
                        track_num = int(primary.id[1:])  # strip letter

                        row = [
                            video_idx,
                            frame_idx,
                            track_num,
                            well_id,
                            cx,
                            cy,
                            hx,
                            hy,
                            tx,
                            ty,
                            region,
                            primary.age,
                            1,  # is_primary
                        ]

                    append_row(main_ds, row)

                    # ------------------
                    # Save FISSION events (secondary tracks)
                    # ------------------
                    for t in tracker.tracks:
                        if (
                            t.role != "secondary"
                            or t.age < tracker.min_secondary_persistence
                            or t.centroid is None
                        ):
                            continue

                        cx, cy = map(int, t.centroid)
                        region = 1 if t.region == "core" else 2
                        track_num = int(t.id[1:])

                        fiss_row = [
                            video_idx,
                            frame_idx,
                            track_num,
                            well_id,
                            cx,
                            cy,
                            int(t.area),
                            region,
                            t.age,
                        ]

                        append_row(fiss_ds, fiss_row)

                total_dets = sum(len(dets) for dets in all_dets)

                crop_h, crop_w = well_crops[0].shape
                stitched = stitch_wells(annotated_well_images, (crop_h, crop_w))
                stitched = annotate_frame_metadata(
                    stitched, frame_idx, video_idx, total_dets, CAPTURE_FPS
                )

                if out_video is None:
                    sH, sW, _ = stitched.shape
                    path = DATA_OUT_DIR / "Annotated.mp4"
                    out_video = cv2.VideoWriter(str(path), fourcc, 30, (sW, sH))

                if frame_idx % 100 == 0:
                    out_video.write(stitched)


                frame_idx += 1
                proc_frames += 1
                elapsed = time.perf_counter() - start_time
                proc_time = proc_frames / elapsed
                print(
                    f"Processing video: {video_idx}, frame: {frame_idx}, proc/sec: {proc_time}"
                )

            cap.release()
except Exception as e:
    print(str(e))

finally:
    for h5 in WELL_H5.values():
        h5.close()

    out_video.release()
    cv2.destroyAllWindows()
    print("\nDone.")
