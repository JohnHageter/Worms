import sys
from pathlib import Path
import cv2
import numpy as np
import h5py

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from Module.Worms import WormTracker
from Module.imageprocessing.background import sample_background
from Module.imageprocessing.foreground import extract_foreground
from Module.imageprocessing.ImageProcessor import *
from Module.utils import *
from Module.dataset.video import *
from Module.detection.Drawer import ROIDrawer

input_dir = Path("D:/Sachi/T3/week_1/")
output_h5 = Path("D:/Sachi/T3/output_tracks_week_1.h5")
output_video_path = Path("D:/Sachi/T3/annotated_week_1.mp4")
save_every = 5
fps = 2

video_paths = sorted(input_dir.glob("*.mp4"))

print("Initializing from first video...")
video0 = open_dataset(str(video_paths[0]))
background = sample_background(video0, n_frames=200).astype(np.uint8)

_, frame = video0.read()
wells = ROIDrawer.load("wells.npy")
wells = expand_well_radius(wells, 1.1)

H, W = frame.shape[:2]
frame_size = (W // 2, H // 2)
video0.release()

tracker = WormTracker(max_dist=500, max_missed=1000)

h5_file = h5py.File(output_h5, "w")
track_ds = h5_file.create_dataset(
    "tracks", shape=(0, 9), maxshape=(None, 9), dtype=np.int32, compression="gzip"
)
wells_ds = h5_file.create_dataset("wells", data=np.array(wells))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(str(output_video_path), fourcc, fps, frame_size)

global_frame_idx = 0
video_count = 1

for video_path in video_paths:
    print(f"Processing: {video_path.name}")
    video = open_dataset(str(video_path))

    while True:
        ret, img = video.read()
        if not ret:
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

        # ---- foreground extraction ----
        fg, thresh = extract_foreground(gray, background, thresh_val=20)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        num_labels, labels, stats, centroids = find_components(thresh)

        worms = build_worms(labels, stats, centroids, wells, minimum_skeleton_area=100)
        worms_to_keep = filter_worms(worms, min_area=30)

        # ---- Update tracker ----
        worm_tracks = tracker.update(worms_to_keep)

        # ---- Visualization ----
        visual = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # Draw wells
        for well in wells:
            x, y, r = map(int, well)
            cv2.circle(visual, (x, y), r, (255, 0, 0), 1)

        for t in worm_tracks:
            if not np.isnan(t.centroid[0]):
                cx, cy = map(int, t.centroid)
                cv2.circle(visual, (cx, cy), 3, (0, 255, 0), -1)
                cv2.putText(visual, f"ID:{t.id}", (cx+5, cy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),1)

                if t.head is not None:
                    hx, hy = map(int, t.head)
                    cv2.rectangle(visual, (hx-4, hy-4), (hx+4, hy+4), (255,0,0), -1)
                if t.tail is not None:
                    tx, ty = map(int, t.tail)
                    cv2.circle(visual, (tx, ty), 3, (0,0,255), -1)
                            
            if t.id == 1:
                cx, cy = (int(t.centroid[0]), int(t.centroid[1])) if not np.isnan(t.centroid[0]) else (None, None)
                hx, hy = (int(t.head[0]), int(t.head[1])) if t.head is not None else (None, None)
                tx, ty = (int(t.tail[0]), int(t.tail[1])) if t.tail is not None else (None, None)
                print(f"Worm 1: centroid: {cx},{cy}: head: {hx},{hy}: tail: {tx},{ty}")

        # Frame info
        current_sec = global_frame_idx / fps
        minutes, seconds = divmod(int(current_sec), 60)
        hours, minutes = divmod(minutes, 60)
        time_str = f"Time: {hours:03d}:{minutes:02d}:{seconds:02d}"

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
            visual, time_str, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )

        display = cv2.resize(
            visual,
            (visual.shape[1] // 2, visual.shape[0] // 2),
            interpolation=cv2.INTER_AREA,
        )
        cv2.imshow("Tracking", display)
        if global_frame_idx % save_every == 0:
            out.write(display)

        # ---- Save frame records ----
        frame_data = np.array(
            [
                (
                    global_frame_idx,
                    t.id,
                    t.well_id,
                    (
                        int(t.centroid[0])
                        if t.centroid is not None and not np.isnan(t.centroid[0])
                        else -1
                    ),
                    (
                        int(t.centroid[1])
                        if t.centroid is not None and not np.isnan(t.centroid[0])
                        else -1
                    ),
                    t.head[0] if t.head is not None else -1,
                    t.head[1] if t.head is not None else -1,
                    t.tail[0] if t.tail is not None else -1,
                    t.tail[1] if t.tail is not None else -1,
                )
                for t in worm_tracks
            ],
            dtype=np.int32,
        )

        track_ds.resize(track_ds.shape[0] + frame_data.shape[0], axis=0)
        track_ds[-frame_data.shape[0] :] = frame_data

        global_frame_idx += 1
        if cv2.waitKey(1) & 0xFF == 27:
            break

    video.release()
    video_count += 1

out.release()
cv2.destroyAllWindows()
h5_file.close()
print("Done:", output_h5)
