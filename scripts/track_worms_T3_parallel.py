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
input_dirs = [Path("D:/TestData/week_1/"), Path("D:/TestData/week_2/")]
output_h5 = [Path("D:/TestData/output_tracks_week_1.h5"), Path("D:/TestData/output_tracks_week_2.h5")]
output_video_path = [Path("D:/TestData/annotated_week_1.mp4"), Path("D:/TestData/annotated_week_2.mp4")]

save_every = 5
fps = 2

# ---------------- INIT ----------------
# Initialize WormTracker for all wells globally
print("Initializing from first video of first folder...")
first_video_path = sorted(input_dirs[0].glob("*.mp4"))[0]
video0 = open_dataset(str(first_video_path))
background = sample_background(video0, n_frames=200).astype(np.uint8)
_, frame = video0.read()

wells = ROIDrawer.load("wells.npy")
wells = expand_well_radius(wells, 1.1)
video0.release()

H, W = frame.shape[:2]
frame_size = (W // 2, H // 2)

# Global WormTracker to carry IDs across folders
from Module.Worms import WormTracker
tracker = WormTracker(max_dist=500, max_missed=20000)

# ---------------- PROCESS ----------------
global_frame_idx = 0
video_count = 1

for folder_idx, input_dir in enumerate(input_dirs):
    video_paths = sorted(input_dir.glob("*.mp4"))

    # Create H5 and VideoWriter for this folder
    h5_file = h5py.File(output_h5[folder_idx], "w")
    track_ds = h5_file.create_dataset(
        "tracks",
        shape=(0, 9),
        maxshape=(None, 9),
        dtype=h5py.string_dtype(encoding="utf-8"),
        compression="gzip"
    )
    h5_file.create_dataset("wells", data=np.array(wells))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_video_path[folder_idx]), fourcc, fps, frame_size)

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
                tracks = tracker.update(worms)  # <-- use global tracker
                all_tracks.extend(tracks)

            # ---- visualization ----
            visual = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            for well in wells:
                x, y, r = map(int, well)
                cv2.circle(visual, (x, y), r, (255, 0, 0), 1)

            for t in all_tracks:
                if t.centroid is None or np.isnan(t.centroid[0]):
                    continue
                cx, cy = int(t.centroid[0]), int(t.centroid[1])
                cv2.circle(visual, (cx, cy), 3, (0, 255, 0), -1)
                cv2.putText(visual, str(t.id), (cx+5, cy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

            # ---- save frame records ----
            frame_data = np.array([
                (
                    str(global_frame_idx),
                    str(t.id),
                    str(t.well_id),
                    str(int(t.centroid[0]) if not np.isnan(t.centroid[0]) else -1),
                    str(int(t.centroid[1]) if not np.isnan(t.centroid[1]) else -1),
                    str(int(t.head[0]) if t.head is not None else -1),
                    str(int(t.head[1]) if t.head is not None else -1),
                    str(int(t.tail[0]) if t.tail is not None else -1),
                    str(int(t.tail[1]) if t.tail is not None else -1)
                )
                for t in all_tracks
            ], dtype=object)

            track_ds.resize(track_ds.shape[0] + len(frame_data), axis=0)
            track_ds[-len(frame_data):] = frame_data

            # ---- display ----
            display = cv2.resize(visual, (visual.shape[1]//2, visual.shape[0]//2), interpolation=cv2.INTER_AREA)
            cv2.imshow("Tracking", display)
            if global_frame_idx % save_every == 0:
                out.write(display)

            global_frame_idx += 1
            if cv2.waitKey(1) & 0xFF == 27:
                break

        video.release()
        video_count += 1

    out.release()
    h5_file.close()

cv2.destroyAllWindows()
print("Done processing all folders. Global tracker IDs preserved across folders.")