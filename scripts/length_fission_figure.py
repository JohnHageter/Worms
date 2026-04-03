import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


import h5py
import numpy as np
import matplotlib.pyplot as plt

h5_path = "D:/output_tracks_week_1.h5"

with h5py.File(h5_path, "r") as f:
    data = f["tracks"][:]

FRAME = 0
TRACK_ID = 1
WELL = 2
CX = 3
CY = 4
H1X = 5
H1Y = 6
H2X = 7
H2Y = 8

fps = 2

# ---- compute length (0 if missing head/tail) ----
has_head = data[:, H1X] >= 0
has_tail = data[:, H2X] >= 0
valid = has_head & has_tail

head = data[:, [H1X, H1Y]]
center = data[:, [CX, CY]]
tail = data[:, [H2X, H2Y]]

d1 = np.linalg.norm(head - center, axis=1)
d2 = np.linalg.norm(center - tail, axis=1)

length = np.where(valid, d1 + d2, 0)

frames = data[:, FRAME]
track_ids = data[:, TRACK_ID]
wells = data[:, WELL]

# ---- setup plots for wells 0–5 ----
target_ids = np.arange(6)

fig, axes = plt.subplots(6, 1, figsize=(8, 18), sharex=True)

# ---- compute track lengths (number of frames per ID) ----
unique_ids, counts = np.unique(track_ids, return_counts=True)
track_lengths = dict(zip(unique_ids, counts))

# ---- find first occurrence of each track ----
first_occurrence = {}
for tid, frame, well in zip(track_ids, frames, wells):
    if tid not in first_occurrence:
        first_occurrence[tid] = (frame, well)

# ---- filter valid fission events ----
MIN_TRACK_LENGTH = 5000

valid_fission_events = []
for tid, (f_frame, f_well) in first_occurrence.items():
    if tid in target_ids:
        continue  # skip original worms

    if track_lengths.get(tid, 0) > MIN_TRACK_LENGTH:
        valid_fission_events.append((tid, f_frame, f_well))

# ---- plotting ----
for i, ax in enumerate(axes):
    tid = target_ids[i]
    
    mask = track_ids == tid

    if np.sum(mask) == 0:
        ax.set_title(f"Well {i} (ID {tid}) - no data")
        continue

    f = frames[mask]
    l = length[mask]

    # sort by frame
    order = np.argsort(f)
    f = f[order]
    l = l[order]

    time_sec = f / fps

    ax.plot(time_sec, l, label=f"ID {tid}")

    # ---- fission events (new IDs in this well) ----
    for other_id, (f_frame, f_well) in first_occurrence.items():
        if other_id in target_ids:
            continue  # skip original worms

        if f_well == i:
            t = f_frame / fps
            ax.scatter(t, np.max(l) if len(l) > 0 else 0, color="red", s=10)

    ax.set_ylabel("Length (px)")
    ax.set_title(f"Well {i} (ID {tid})")

axes[-1].set_xlabel("Time (seconds)")
plt.tight_layout()
plt.show()