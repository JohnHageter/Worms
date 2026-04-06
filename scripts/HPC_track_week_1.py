import sys
from pathlib import Path
import cv2
import numpy as np
import h5py

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from Module.Worms import WellTracker
from Module.imageprocessing.background import sample_background
from Module.imageprocessing.foreground import extract_foreground
from Module.imageprocessing.ImageProcessor import (
    build_worms_single_well,
    find_components,
    mask_to_well,
)
from Module.dataset.video import open_dataset
from Module.detection.Drawer import ROIDrawer

# ---------------- CONFIG ----------------
input_dir = Path("D:/Sachi/T3/week_1/")
output_h5 = Path("D:/Sachi/T3/output_tracks_week_1.h5")
output_video_path = Path("D:/Sachi/T3/annotated_week_1.mp4")

save_every = 5
fps = 2
video_paths = sorted(input_dir.glob("*.mp4"))

# ---------------- INIT ----------------
print("Initializing from first video...")
video0 = open_dataset(str(video_paths[0]))
background = sample_background(video0, n_frames=200).astype(np.uint8)

ret, frame = video0.read()
if not ret:
    raise RuntimeError("Failed to read first frame")

if frame.ndim == 3:
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

wells = ROIDrawer.load("wells.npy")

H, W = frame.shape[:2]
frame_size = (W // 2, H // 2)

video0.release()

# Precompute kernel ONCE
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

# Pre-create trackers
trackers = {
    i: WellTracker(i, max_dist=500, max_missed=1000) for i in range(len(wells))
}

# ---------------- OUTPUT ----------------
h5_file = h5py.File(output_h5, "w")
track_ds = h5_file.create_dataset(
    "tracks",
    shape=(0, 9),
    maxshape=(None, 9),
    dtype=h5py.string_dtype(encoding="utf-8"),
    compression="gzip",
)

h5_file.create_dataset("wells", data=np.array(wells))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(str(output_video_path), fourcc, fps, frame_size)

global_frame_idx = 0
video_count = 1

# Buffer to reduce HDF5 writes (BIG speedup)
write_buffer = []
buffer_size = 1000

# ---------------- PROCESS ----------------
for video_path in video_paths:
    print(f"Processing: {video_path.name}")
    video = open_dataset(str(video_path))

    while True:
        ret, img = video.read()
        if not ret:
            break

        # ---- grayscale ----
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

        # ---- foreground ----
        _, thresh = extract_foreground(gray, background, thresh_val=13)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        all_tracks = []

        # ---- per well ----
        for well_id, well in enumerate(wells):
            thresh_well = mask_to_well(thresh, well)

            _, labels, stats, centroids = find_components(thresh_well)

            worms = build_worms_single_well(
                labels, stats, centroids, well_id, minimum_area=10
            )

            tracks = trackers[well_id].update(worms)
            all_tracks.extend(tracks)

        # ---- visualization ----
        visual = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # Draw wells
        for x, y, r in wells:
            cv2.circle(visual, (int(x), int(y)), int(r), (255, 0, 0), 1)

        # Draw tracks
        for t in all_tracks:
            c = t.centroid
            if c is None or np.isnan(c[0]):
                continue

            cx, cy = int(c[0]), int(c[1])

            cv2.circle(visual, (cx, cy), 3, (0, 255, 0), -1)
            cv2.putText(
                visual,
                t.id,
                (cx + 5, cy - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

            if t.head is not None and t.prev_head is not None:
                hx, hy = int(t.head[0]), int(t.head[1])
                cv2.rectangle(
                    visual, (hx - 4, hy - 4), (hx + 4, hy + 4), (255, 0, 0), -1
                )

            if t.tail is not None and t.prev_tail is not None:
                tx, ty = int(t.tail[0]), int(t.tail[1])
                cv2.circle(visual, (tx, ty), 3, (0, 0, 255), -1)

        # ---- frame info ----
        current_sec = global_frame_idx / fps
        m, s = divmod(int(current_sec), 60)
        h, m = divmod(m, 60)

        cv2.putText(
            visual,
            f"Video #{video_count}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            visual,
            f"Frame: {global_frame_idx}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            visual,
            f"Time: {h:03d}:{m:02d}:{s:02d}",
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        display = cv2.resize(visual, frame_size, interpolation=cv2.INTER_AREA)

        if global_frame_idx % save_every == 0:
            out.write(display)

        # ---- buffered HDF5 write ----
        for t in all_tracks:
            c = t.centroid
            cx = int(c[0]) if not np.isnan(c[0]) else -1
            cy = int(c[1]) if not np.isnan(c[1]) else -1

            hx = int(t.head[0]) if t.head is not None else -1
            hy = int(t.head[1]) if t.head is not None else -1
            tx = int(t.tail[0]) if t.tail is not None else -1
            ty = int(t.tail[1]) if t.tail is not None else -1

            write_buffer.append(
                (
                    str(global_frame_idx),
                    t.id,
                    str(t.well_id),
                    str(cx),
                    str(cy),
                    str(hx),
                    str(hy),
                    str(tx),
                    str(ty),
                )
            )

        # flush buffer
        if len(write_buffer) >= buffer_size:
            arr = np.array(write_buffer, dtype=object)
            track_ds.resize(track_ds.shape[0] + len(arr), axis=0)
            track_ds[-len(arr) :] = arr
            write_buffer.clear()

        global_frame_idx += 1

    video.release()
    video_count += 1

# ---- final flush ----
if write_buffer:
    arr = np.array(write_buffer, dtype=object)
    track_ds.resize(track_ds.shape[0] + len(arr), axis=0)
    track_ds[-len(arr) :] = arr

# ---- cleanup ----
out.release()
h5_file.close()
cv2.destroyAllWindows()

print("Done:", output_h5)
