import sys
from pathlib import Path
import numpy as np
import h5py
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


# ---- SETTINGS ----
h5_path = "D:/output_tracks_week_1.h5"
well_to_plot = 0          # which well to plot
worm_ids_to_plot = [4]    # one worm per well
step = 5                 # plot every 30th point

# ---- LOAD DATA ----
with h5py.File(h5_path, "r") as f:
    tracks = np.array(f["tracks"])  # (frame_idx, id, well_id, cx, cy, head_x, head_y, tail_x, tail_y)

# ---- FILTER BY WELL ----
well_mask = tracks[:, 2] == well_to_plot
well_tracks = tracks[well_mask]

fig, ax = plt.subplots(figsize=(8, 6))

sc = None  # initialize for colorbar

for worm_id in worm_ids_to_plot:
    worm_mask = well_tracks[:, 1] == worm_id
    worm_data = well_tracks[worm_mask]

    if len(worm_data) == 0:
        continue

    # downsample
    worm_data = worm_data[::step]
    cx = worm_data[:, 3]
    cy = worm_data[:, 4]

    # scale color by point index in the track
    colors = np.linspace(0, 1, len(cx))

    # --- OPTION 1: scatter colored points ---
    #sc = ax.scatter(cx, cy, c=colors, cmap="viridis", s=20)

    # --- OPTION 2: line connecting points ---
    for i in range(1, len(cx)):
        ax.plot(cx[i-1:i+1], cy[i-1:i+1], color=plt.cm.viridis(colors[i]), alpha=0.8)

ax.set_title(f"Centroid Path - Well {well_to_plot}")
ax.set_xlabel("X position (pixels)")
ax.set_ylabel("Y position (pixels)")
ax.invert_yaxis()  # match image coordinates

if sc is not None:
    cbar = fig.colorbar(sc, ax=ax, label="Track progress (0=start, 1=end)")

plt.tight_layout()
plt.show()