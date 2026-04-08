import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from collections import Counter
from matplotlib.lines import Line2D

# ---------------- CONFIG ----------------
DATA_DIR = Path("D:/Sachi/T3/week_1_out_v1")
WELL_FILES = sorted(DATA_DIR.glob("well_*.h5"))

WELL_DIAMETER_MM = 35.0
BIN_SIZE = 1000  # frames per bin
SMOOTH_SIGMA = 3
FPS = 2.0  # frames per second

REGION_CORE = 1
REGION_WALL = 2

REGION_COLORS = {
    REGION_CORE: "green",
    REGION_WALL: "orange",
}
# --------------------------------------


def print_h5_info(h5, label=""):
    print(f"\n=== HDF5 INFO {label} ===")
    for name, obj in h5.items():
        print(f"Dataset '{name}': shape={obj.shape}, dtype={obj.dtype}")
    print("========================")


def compute_length_px(main):
    # NEW COLUMN INDICES
    cx, cy = main[:, 4], main[:, 5]
    hx, hy = main[:, 6], main[:, 7]
    tx, ty = main[:, 8], main[:, 9]
    region = main[:, 10]
    is_primary = main[:, 12]

    # Confidence = primary worm + valid head & tail
    confident = (is_primary == 1) & (hx >= 0) & (tx >= 0)

    length_px = np.full(len(main), np.nan)

    d_hc = np.sqrt((hx - cx) ** 2 + (hy - cy) ** 2)
    d_ct = np.sqrt((cx - tx) ** 2 + (cy - ty) ** 2)

    length_px[confident] = d_hc[confident] + d_ct[confident]
    return length_px, confident, region


def robust_well_diameter_px(cx, cy):
    return max(
        np.nanmax(cx) - np.nanmin(cx),
        np.nanmax(cy) - np.nanmin(cy),
    )


def bin_trace(x, y, region, bin_size):
    xb, yb, rb = [], [], []
    n_bins = len(x) // bin_size

    for i in range(n_bins):
        sl = slice(i * bin_size, (i + 1) * bin_size)
        yseg = y[sl]
        rseg = region[sl]

        valid = ~np.isnan(yseg)
        if valid.sum() < bin_size * 0.3:
            continue

        xb.append(np.nanmean(x[sl]))
        yb.append(np.nanmean(yseg[valid]))

        r_mode = Counter(rseg[valid].astype(int)).most_common(1)[0][0]
        rb.append(r_mode)

    return np.array(xb), np.array(yb), np.array(rb)


# ---------------- MAIN ----------------
for path in WELL_FILES:
    with h5py.File(path, "r") as h5:
        print_h5_info(h5, label=path.name)
        main = h5["main"][:]
        fission = h5["fission"][:] if "fission" in h5 else None

    # --- compute length ---
    length_px, confident, region = compute_length_px(main)

    cx, cy = main[:, 4], main[:, 5]
    well_diam_px = robust_well_diameter_px(cx[confident], cy[confident])
    px_to_mm = WELL_DIAMETER_MM / well_diam_px
    length_mm = length_px * px_to_mm

    # --- time axis ---
    frame_idx = main[:, 1]
    time_hours = frame_idx / FPS / 3600.0

    # --- bin + smooth ---
    bx, by, br = bin_trace(time_hours, length_mm, region, BIN_SIZE)
    by_smooth = gaussian_filter1d(by, sigma=SMOOTH_SIGMA)

    # --- plot ---
    plt.figure(figsize=(12, 4))

    for i in range(len(bx) - 1):
        color = REGION_COLORS.get(br[i], "gray")
        plt.plot(bx[i : i + 2], by_smooth[i : i + 2], lw=2, color=color)

    # --- fission markers ---
    print("Number of fission events:", len(fission) if fission is not None else 0)
    if fission is not None and len(fission) > 0:
        fission_hours = fission[:, 1] / FPS / 3600.0
        y_marker = np.nanmax(by_smooth) * 1.05
        plt.scatter(
            fission_hours,
            np.full_like(fission_hours, y_marker),
            color="red",
            s=30,
            zorder=5,
        )

    # --- 24h ticks ---
    max_hours = time_hours[-1]
    day_ticks = np.arange(0, max_hours + 24, 24)
    plt.xticks(day_ticks)

    plt.xlabel("Time (hours)")
    plt.ylabel("Head–Centroid–Tail length (mm)")
    plt.title(f"{path.stem} — Main worm length")

    # --- legend ---
    legend_elements = [
        Line2D([0], [0], color="green", lw=2, label="Core"),
        Line2D([0], [0], color="orange", lw=2, label="Wall"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="red",
            linestyle="None",
            markersize=6,
            label="Fission event",
        ),
    ]

    plt.legend(handles=legend_elements, frameon=False)
    plt.tight_layout()
    plt.show()
