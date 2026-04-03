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
X = 3
Y = 4

# sort by track then frame
data = data[np.lexsort((data[:, FRAME], data[:, TRACK_ID]))]

# compute displacements
same_track = data[1:, TRACK_ID] == data[:-1, TRACK_ID]

dx = data[1:, X] - data[:-1, X]
dy = data[1:, Y] - data[:-1, Y]
disp = np.sqrt(dx**2 + dy**2)

frames = data[1:, FRAME][same_track]
wells = data[1:, WELL][same_track]
disp = disp[same_track]

# binning
bin_size = 120
bin_ids = frames // bin_size

# unique wells
unique_wells = np.unique(wells)

fps = 2

fig, axes = plt.subplots(len(unique_wells), 1, figsize=(8, 3*len(unique_wells)), sharex=True)

if len(unique_wells) == 1:
    axes = [axes]  # make iterable

for ax, w in zip(axes, unique_wells):
    mask = wells == w

    b = bin_ids[mask]
    d = disp[mask]

    # fast aggregation
    sum_disp = np.bincount(b, weights=d)
    counts = np.bincount(b)

    valid = counts > 0
    mean_disp = sum_disp[valid] / counts[valid]
    bins = np.nonzero(valid)[0]

    time_sec = (bins * bin_size) / fps

    ax.plot(time_sec, mean_disp)
    ax.set_title(f"Well {int(w)}")
    ax.set_ylabel("Displacement (px)")

axes[-1].set_xlabel("Time (seconds)")
plt.tight_layout()
plt.show()