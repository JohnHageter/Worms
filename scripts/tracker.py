# Working with Trial1 White light

import cv2
import numpy as np
from pathlib import Path
from Module.Worms import WormTracker
from Module.imageprocessing.background import sample_background
from Module.imageprocessing.foreground import extract_foreground
from Module.utils import *
from Module.dataset.Dataset import *
from Module.imageprocessing.ImageProcessor import *
from Module.detection.Arena import (
    detect_wells,
    detect_wells_interactive,
    WellDetectionParams,
)
import csv
from collections import defaultdict

# generate_dataset_from_timelapse(
#     "D:/Sachi/T1 4233A raw images/timelapse_images"
# )
video = open_dataset("D:/Sachi/T2 4233A White Light/timelapse_images_WL/Trial1_WL.mp4")

# ---- output directories ----
output_dir = Path("D:/Sachi/T2 4233A White Light/timelapse_images_WL/output")
frames_dir = output_dir / "frames"
tracks_dir = output_dir / "tracks"

frames_dir.mkdir(parents=True, exist_ok=True)
tracks_dir.mkdir(parents=True, exist_ok=True)

# ---- background ----
background = sample_background(video, n_frames=200)
background = background.astype(np.uint8)

params = WellDetectionParams(
    well_tolerance=0.15,
    n_wells=6,
    hough_dp=1.5,
    min_dist=50,
    hough_param1=24,
    hough_param2=25,
    min_radius=10,
    max_radius=117,
)

wells, masks, params = detect_wells_interactive(video, params)
wells = expand_well_radius(wells, 1.2)

tracker = WormTracker(max_dist=300, max_missed=100)

track_records = []
frame_idx = 0

video.set(cv2.CAP_PROP_POS_FRAMES, 0)

print("tracking")

while True:
    ret, img = video.read()
    if not ret:
        break

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    H, W = img.shape

    fg, thresh = extract_foreground(img, background, thresh_val=35)
    num_labels, labels, stats, centroids = find_components(thresh)

    worms = build_worms(
        labels,
        stats,
        centroids,
        wells,
        minimum_skeleton_area=100,
    )

    worms_to_keep = filter_worms(worms, min_area=10)
    worm_tracks = tracker.update(worms_to_keep, frame_idx=frame_idx)

    visual = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for worm in worms_to_keep:
        cx, cy = map(int, worm["centroid"])
        end1 = tuple(map(int, worm["end1"])) if worm["end1"] is not None else None
        end2 = tuple(map(int, worm["end2"])) if worm["end2"] is not None else None

        cv2.circle(visual, (cx, cy), 3, (0, 255, 0), -1)

        if end1 is not None and end2 is not None:
            cv2.circle(visual, end1, 3, (255, 0, 0), -1)
            cv2.circle(visual, end2, 3, (0, 0, 255), -1)

    for well in wells:
        x, y, r = map(int, well)
        cv2.circle(visual, (x, y), r, (255, 0, 0), 1)

    for t in worm_tracks:
        cx, cy = map(int, t.last_centroid)
        cv2.putText(
            visual,
            f"ID:{t.id}",
            (cx + 5, cy - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

    # ---- save frame as PNG ----
    frame_path = frames_dir / f"frame_{frame_idx:06d}.png"
    cv2.imwrite(str(frame_path), cv2.resize(visual, (W//2, H//2), cv2.INTER_AREA))

    cv2.imshow(
        "Video",
        cv2.resize(visual, (W // 2, H // 2), cv2.INTER_AREA),
    )

    for t in worm_tracks:
        cx, cy = t.last_centroid
        e1 = t.last_end1
        e2 = t.last_end2

        track_records.append(
            {
                "frame": frame_idx,
                "time (min)": frame_idx * 2,
                "track_id": t.id,
                "well_id": t.well_id,
                "center_x": int(cx),
                "center_y": int(cy),
                "end1_x": e1[0] if e1 is not None else "",
                "end1_y": e1[1] if e1 is not None else "",
                "end2_x": e2[0] if e2 is not None else "",
                "end2_y": e2[1] if e2 is not None else "",
                "age": t.age,
                "missed": t.missed,
            }
        )

    if cv2.waitKey(1) & 0xFF == 27:
        break

    frame_idx += 1

video.release()
cv2.destroyAllWindows()

# ---- save per-track CSVs ----
fields = [
    "frame",
    "time (min)",
    "track_id",
    "well_id",
    "center_x",
    "center_y",
    "end1_x",
    "end1_y",
    "end2_x",
    "end2_y",
    "age",
    "missed",
]

tracks_by_id = defaultdict(list)
for row in track_records:
    tracks_by_id[row["track_id"]].append(row)

for track_id, rows in tracks_by_id.items():
    out_file = tracks_dir / f"track_{track_id:04d}.csv"

    with open(out_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
