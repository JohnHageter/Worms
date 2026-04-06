import sys
from pathlib import Path
import cv2
import numpy as np
import h5py

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from Module.Worms import WormTracker, WellTracker, WormDetection
from Module.imageprocessing.background import sample_background
from Module.imageprocessing.foreground import extract_foreground
from Module.imageprocessing.ImageProcessor import build_worms_single_well, filter_worms, find_components, mask_to_well
from Module.utils import *
from Module.dataset.video import open_dataset
from Module.dataset.Dataset import expand_well_radius
from Module.detection.Drawer import ROIDrawer

# ---------------- CONFIG ----------------
input_dir = Path("D:/TestData/week_1/")
output_h5 = Path("D:/TestData/output_tracks_week_1.h5")
output_video_path = Path("D:/TestData/annotated_week_1.mp4")

save_every = 5
fps = 2

video_paths = sorted(input_dir.glob("*.mp4"))

# ---------------- INIT ----------------
print("Initializing from first video...")
video0 = open_dataset(str(video_paths[0]))
background = sample_background(video0, n_frames=200).astype(np.uint8)

_, frame = video0.read()
wells = ROIDrawer.load("wells.npy")
wells = expand_well_radius(wells, 1.1)

H, W = frame.shape[:2]
frame_size = (W // 2, H // 2)
video0.release()

# Per-well trackers
trackers = {i: WellTracker(i, max_dist=500, max_missed=20000) for i in range(len(wells))}

# ---------------- OUTPUT ----------------
h5_file = h5py.File(output_h5, "w")
track_ds = h5_file.create_dataset(
    "tracks",
    shape=(0, 9),
    maxshape=(None, 9),
    dtype=h5py.string_dtype(encoding="utf-8"),
    compression="gzip"
)
h5_file.create_dataset("wells", data=np.array(wells))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(str(output_video_path), fourcc, fps, frame_size)

global_frame_idx = 0
video_count = 1

# ---------------- PROCESS ----------------
for video_path in video_paths:
    print(f"Processing: {video_path.name}")
    video = open_dataset(str(video_path))

    while True:
        ret, img = video.read()
        if not ret:
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

        # ---- foreground ----
        _, thresh = extract_foreground(gray, background, thresh_val=15)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        all_tracks = []

        # ---- process per well ----
        for well_id, well in enumerate(wells):
            thresh_well = mask_to_well(thresh, well)

            _, labels, stats, centroids = find_components(thresh_well)

            worms = build_worms_single_well(labels, stats, centroids, well_id, minimum_area=10)
            #worms = filter_worms(worms, min_area=5, max_area=1000)

            tracks = trackers[well_id].update(worms)
            all_tracks.extend(tracks)

        # ---- visualization ----
        visual = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # Draw wells
        for well in wells:
            x, y, r = map(int, well)
            cv2.circle(visual, (x, y), r, (255, 0, 0), 1)

        print(len(all_tracks))
        for t in all_tracks:
            if t.centroid is None or np.isnan(t.centroid[0]):
                continue

            cx, cy = int(t.centroid[0]), int(t.centroid[1])
            cv2.circle(visual, (cx, cy), 3, (0, 255, 0), -1)
            cv2.putText(visual, str(t.id), (cx+5, cy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

            if t.head is not None:
                hx, hy = int(t.head[0]), int(t.head[1])
                cv2.rectangle(visual, (hx-4, hy-4), (hx+4, hy+4), (255,0,0), -1)
            if t.tail is not None:
                tx, ty = int(t.tail[0]), int(t.tail[1])
                cv2.circle(visual, (tx, ty), 3, (0,0,255), -1)

        # ---- frame info ----
        current_sec = global_frame_idx / fps
        m, s = divmod(int(current_sec), 60)
        h, m = divmod(m, 60)

        cv2.putText(visual, f"Video #{video_count}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(visual, f"Frame: {global_frame_idx}", (10,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(visual, f"Time: {h:03d}:{m:02d}:{s:02d}", (10,80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        display = cv2.resize(visual, (visual.shape[1]//2, visual.shape[0]//2), interpolation=cv2.INTER_AREA)
        cv2.imshow("Tracking", display)
        #cv2.imshow("Threshold", cv2.resize(thresh, (thresh.shape[1]//2, thresh.shape[0]//2), interpolation=cv2.INTER_AREA))
        if global_frame_idx % save_every == 0:
            out.write(display)

        # ---- save frame records ----
        frame_data = np.array([
            (
                str(global_frame_idx),
                str(t.id),
                str(t.well_id),
                str(int(t.centroid[0]) if not np.isnan(t.centroid[0]) else -1),
                str(int(t.centroid[1]) if not np.isnan(t.centroid[0]) else -1),
                str(int(t.head[0]) if t.head is not None else -1),
                str(int(t.head[1]) if t.head is not None else -1),
                str(int(t.tail[0]) if t.tail is not None else -1),
                str(int(t.tail[1]) if t.tail is not None else -1)
            )
            for t in all_tracks
        ], dtype=object)

        track_ds.resize(track_ds.shape[0] + len(frame_data), axis=0)
        track_ds[-len(frame_data):] = frame_data

        global_frame_idx += 1
        if cv2.waitKey(1) & 0xFF == 27:
            break

    video.release()
    video_count += 1

# ---- cleanup ----
out.release()
cv2.destroyAllWindows()
h5_file.close()
print("Done:", output_h5)