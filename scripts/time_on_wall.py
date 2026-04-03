import sys
from pathlib import Path
import numpy as np
import h5py
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

# ---- SETTINGS ----
h5_path = "D:/output_tracks_week_1.h5"
worm_ids_to_plot = list(range(7))  # worm IDs 0-6
well_radius_threshold = 0.95       # fraction of well radius to consider "inside"

# ---- LOAD DATA ----
with h5py.File(h5_path, "r") as f:
    tracks = np.array(f["tracks"])  # (frame_idx, id, well_id, cx, cy, head_x, head_y, tail_x, tail_y)
    wells = np.array(f["wells"])    # (x, y, r) for each well

# ---- PREPARE FIGURE ----
fig, axes = plt.subplots(len(worm_ids_to_plot), 1, figsize=(12, 2*len(worm_ids_to_plot)), sharex=True)

if len(worm_ids_to_plot) == 1:
    axes = [axes]

max_frame = tracks[:, 0].max() + 1

for i, worm_id in enumerate(worm_ids_to_plot):
    ax = axes[i]

    worm_mask = tracks[:, 1] == worm_id
    worm_data = tracks[worm_mask]

    if len(worm_data) == 0:
        # Entire worm never tracked
        ax.broken_barh([(0, max_frame)], (0, 1), facecolors='red')
        ax.set_ylabel(f"Worm {worm_id}")
        ax.set_yticks([])
        continue

    well_id = worm_data[0, 2]
    well_x, well_y, well_r = wells[well_id]

    tracked_frames = worm_data[:, 0]
    cx = worm_data[:, 3]
    cy = worm_data[:, 4]

    # distance from well center
    dist_from_center = np.sqrt((cx - well_x)**2 + (cy - well_y)**2)
    inside_mask = dist_from_center <= well_r * well_radius_threshold

    # ---- identify untracked periods ----
    full_frames = np.arange(max_frame)
    untracked_mask = np.ones(max_frame, dtype=bool)
    untracked_mask[tracked_frames] = False

    # ---- identify outside periods ----
    outside_mask = np.ones(max_frame, dtype=bool)
    outside_mask[tracked_frames] = ~inside_mask

    # ---- combine periods for plotting ----
    # red = not tracked, orange = outside
    # plot as horizontal bars using broken_barh
    # untracked
    start = None
    for frame_idx in full_frames:
        if untracked_mask[frame_idx]:
            if start is None:
                start = frame_idx
        else:
            if start is not None:
                ax.broken_barh([(start, frame_idx - start)], (0, 1), facecolors='red')
                start = None
    if start is not None:
        ax.broken_barh([(start, max_frame - start)], (0, 1), facecolors='red')

    # outside
    start = None
    for idx, val in enumerate(outside_mask):
        if val:
            if start is None:
                start = idx
        else:
            if start is not None:
                ax.broken_barh([(start, idx - start)], (0, 1), facecolors='orange')
                start = None
    if start is not None:
        ax.broken_barh([(start, max_frame - start)], (0, 1), facecolors='orange')

    ax.set_ylabel(f"Worm {worm_id}")
    ax.set_yticks([])
    ax.set_xlim(0, max_frame)

axes[-1].set_xlabel("Frame index")
fig.suptitle("Worm Tracking Status: Orange = On or near well wall")
plt.tight_layout()
plt.show()