import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import cv2
import numpy as np
from pathlib import Path
import h5py

from Module.Worms import WormTracker
from Module.imageprocessing.background import sample_background
from Module.imageprocessing.foreground import extract_foreground
from Module.imageprocessing.ImageProcessor import *
from Module.utils import *
from Module.dataset.video import *
from Module.detection.Arena import (
    detect_wells,
    WellDetectionParams,
    detect_wells_interactive,
)

input_dir = Path("D:/Sachi/T3/week_1/")
output_h5 = Path("D:/Sachi/T3/output_tracks_week_1.h5")
output_video_path = Path("D:/Sachi/T3/annotated_week_1.mp4")
save_every = 6

video_paths = sorted(input_dir.glob("*.mp4"))

print("Initializing from first video...")
video0 = open_dataset(str(video_paths[0]))

background = sample_background(video0, n_frames=200).astype(np.uint8)

params = WellDetectionParams(
    well_tolerance=0.15,
    n_wells=6,
    hough_dp=1.5,
    min_dist=50,
    hough_param1=24,
    hough_param2=25,
    min_radius=24,
    max_radius=150,
)

wells, masks, params = detect_wells(video0, params)
wells = expand_well_radius(wells, 1.1)

_, frame = video0.read()
H, W = frame.shape[:2]
frame_size = (W // 2, H // 2)
fps = 2
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(str(output_video_path), fourcc, fps, frame_size)

video0.release()


# ---- TRACKER (persistent across videos) ----
tracker = WormTracker(max_dist=500, max_missed=100)

# ---- GLOBAL STORAGE ----
all_records = []
global_frame_idx = 0
video_count = 0

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(str(output_video_path), fourcc, 30, frame_size)

print("Tracking across videos...")

for video_path in video_paths:
    print(f"Processing: {video_path.name}")
    video = open_dataset(str(video_path))

    while True:
        ret, img = video.read()
        if not ret:
            break

        # ---- grayscale ----
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # ---- foreground ----
        fg, thresh = extract_foreground(gray, background, thresh_val=20)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        num_labels, labels, stats, centroids = find_components(thresh)

        worms = build_worms(
            labels,
            stats,
            centroids,
            wells,
            minimum_skeleton_area=100,
        )

        worms_to_keep = filter_worms(worms, min_area=30)
        worm_tracks = tracker.update(worms_to_keep, frame_idx=global_frame_idx)

        # ---- VISUALIZATION ----
        visual = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # draw wells
        for well in wells:
            x, y, r = map(int, well)
            cv2.circle(visual, (x, y), r, (255, 0, 0), 1)

        # draw tracks
        for t in worm_tracks:
            cx, cy = map(int, t.last_centroid)

            head = None
            tail = None

            if t.head is not None:
                hx, hy = t.head
                head = (int(hx), int(hy))

            if t.tail is not None:
                tx, ty = t.tail
                tail = (int(tx), int(ty))
                cv2.circle(visual, tail, 3, (0, 0, 255), -1)

            # centroid
            cv2.circle(visual, (cx, cy), 3, (0, 255, 0), -1)

            # head / tail
            if head is not None:
                cv2.rectangle(
                    visual,
                    (head[0] - 4, head[1] - 4),
                    (head[0] + 4, head[1] + 4),
                    (255, 0, 0),
                    -1,
                )

            if tail is not None:
                cv2.circle(visual, tail, 3, (0, 0, 255), -1)

            # ID label
            cv2.putText(
                visual,
                f"ID:{t.id}",
                (cx + 5, cy - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

            current_sec = global_frame_idx / fps
            minutes, seconds = divmod(int(current_sec), 60)
            hours, minutes = divmod(minutes, 60)
            time_str = f"Time: {hours:03d}:{minutes:02d}:{seconds:02d}"

        cv2.putText(
            visual, time_str, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )

        # video counter
        cv2.putText(
            visual,
            f"Video #{video_count}",
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        # frame counter
        cv2.putText(
            visual,
            f"Frame: {global_frame_idx}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        # resize for speed/display
        display = cv2.resize(
            visual,
            (visual.shape[1] // 2, visual.shape[0] // 2),
            interpolation=cv2.INTER_AREA,
        )

        cv2.imshow("Tracking", display)
        if global_frame_idx % save_every == 0:
            out.write(display)

        # ---- RECORD DATA ----
        for t in worm_tracks:
            cx, cy = t.last_centroid
            e1 = t.head
            e2 = t.tail

            all_records.append(
                (
                    global_frame_idx,
                    t.id,
                    t.well_id,
                    int(cx),
                    int(cy),
                    e1[0] if e1 is not None else -1,
                    e1[1] if e1 is not None else -1,
                    e2[0] if e2 is not None else -1,
                    e2[1] if e2 is not None else -1,
                )
            )

        # ---- EXIT KEY ----
        if cv2.waitKey(1) & 0xFF == 27:
            break

        global_frame_idx += 1

    video.release()
    video_count += 1

out.release()
cv2.destroyAllWindows()

# ---- SAVE HDF5 ----
print("Saving...")

data = np.array(all_records, dtype=np.int32)

with h5py.File(output_h5, "w") as f:
    f.create_dataset("tracks", data=data, compression="gzip")
    f.create_dataset("wells", data=np.array(wells))

print("Done:", output_h5)
