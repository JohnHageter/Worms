from pathlib import Path
import h5py
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from matplotlib.collections import LineCollection


BASE_DIR = Path("D:/Tracking_Data/Sachi/T3/week_1_out/")
WELLS = range(6)

FPS = 2
OUT_DIR = Path(BASE_DIR)
OUT_DIR.mkdir(exist_ok=True)


def load(path):
    with h5py.File(path, "r") as f:
        return f["main"][:]


def compute_time(vid, frame):
    max_vid = vid.max() + 1
    lengths = np.zeros(max_vid, dtype=int)

    for v in range(max_vid):
        m = vid == v
        if np.any(m):
            lengths[v] = frame[m].max() + 1

    offsets = np.cumsum(np.concatenate([[0], lengths[:-1]]))
    return offsets[vid] + frame


def smooth(x, y, t, sigma=4):
    valid = np.isfinite(x) & np.isfinite(y)
    x, y, t = x[valid], y[valid], t[valid]

    order = np.argsort(t)
    x, y, t = x[order], y[order], t[order]

    x = gaussian_filter1d(x, sigma=sigma, mode="nearest")
    y = gaussian_filter1d(y, sigma=sigma, mode="nearest")

    return x, y, t


def colored_line(ax, x, y, c):
    pts = np.array([x, y]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)

    lc = LineCollection(segs, cmap="viridis", array=c, linewidth=1.2)
    ax.add_collection(lc)
    ax.autoscale()
    return lc


for well in WELLS:

    path = BASE_DIR / f"well_{well}.h5"

    data = load(path)

    vid = data[:, 0].astype(int)
    frame = data[:, 1].astype(int)

    t = compute_time(vid, frame) / FPS  

    cx = data[:, 4]
    cy = data[:, 5]

    x, y, t = smooth(cx, cy, t, sigma=4)

    fig, ax = plt.subplots(figsize=(6, 6))

    lc = colored_line(ax, x, y, t/60/60)
    cbar = fig.colorbar(lc, ax=ax)
    cbar.set_label("Time (h)")

    ax.set_title(f"Well {well}")
    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout()

    out_file = OUT_DIR / f"wellc_{well}.png"
    #plt.show()
    plt.savefig(out_file, format="png", bbox_inches="tight")
    plt.close()

    print(f"Saved {out_file}")
