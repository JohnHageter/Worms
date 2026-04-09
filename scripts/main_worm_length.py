import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from collections import Counter
from matplotlib.lines import Line2D

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = Path("D:/Tracking_Data/Sachi/T3/week_1_out")
WELL_FILES = sorted(DATA_DIR.glob("well_*.h5"))

WELL_DIAMETER_MM = 35.0
FPS = 2.0  # frames per second
BIN_FRAMES = 1000  # bin width in frames
SMOOTH_SIGMA = 3  # smoothing after binning
FRAMES_PER_VIDEO = int(FPS * 3530)

REGION_CORE = 1
REGION_WALL = 2

REGION_COLORS = {
    REGION_CORE: "green",
    REGION_WALL: "orange",
}

# ============================================================
# HELPERS
# ============================================================


def print_h5_info(h5, label=""):
    print(f"\n=== HDF5 INFO {label} ===")
    for name, obj in h5.items():
        print(f"Dataset '{name}': shape={obj.shape}, dtype={obj.dtype}")
    print("========================")


def compute_length_px(main):
    """
    Compute head-centroid-tail length in pixels for the PRIMARY worm only.

    main columns:
    4  cx
    5  cy
    6  hx
    7  hy
    8  tx
    9  ty
    10 region
    12 is_primary
    """

    cx, cy = main[:, 4], main[:, 5]
    hx, hy = main[:, 6], main[:, 7]
    tx, ty = main[:, 8], main[:, 9]
    region = main[:, 10]
    is_primary = main[:, 12]

    # Confident frames only
    confident = (is_primary == 1) & (hx >= 0) & (tx >= 0)

    length_px = np.full(len(main), np.nan)

    d_hc = np.sqrt((hx - cx) ** 2 + (hy - cy) ** 2)
    d_ct = np.sqrt((cx - tx) ** 2 + (cy - ty) ** 2)

    length_px[confident] = d_hc[confident] + d_ct[confident]
    return length_px, confident, region


def robust_well_diameter_px(cx, cy):
    """
    For circular ROIs, estimate diameter from centroid span.
    """
    return max(
        np.nanmax(cx) - np.nanmin(cx),
        np.nanmax(cy) - np.nanmin(cy),
    )


def bin_trace_by_time(time_h, length_mm, region, bin_hours):
    """
    Time-based binning with region voting restricted to valid frames.
    """
    xb, yb, rb = [], [], []

    edges = np.arange(time_h.min(), time_h.max() + bin_hours, bin_hours)

    for t0, t1 in zip(edges[:-1], edges[1:]):
        mask = (time_h >= t0) & (time_h < t1)

        if mask.sum() < 10:
            continue

        xb.append(0.5 * (t0 + t1))
        yb.append(np.nanmean(length_mm[mask]))

        r_mode = Counter(region[mask].astype(int)).most_common(1)[0][0]
        rb.append(r_mode)

    return np.asarray(xb), np.asarray(yb), np.asarray(rb)


# ============================================================
# MAIN
# ============================================================

for path in WELL_FILES:
    with h5py.File(path, "r") as h5:
        print_h5_info(h5, label=path.name)
        main = h5["main"][:]
        fission = h5["fission"][:] if "fission" in h5 else None

    # --------------------------------------------------------
    # Extract confident primary frames
    # --------------------------------------------------------
    length_px, confident, region = compute_length_px(main)

    length_px = length_px[confident]
    region = region[confident]
    frame_idx = main[:, 1][confident]
    video_idx = main[:, 0][confident]

    # --------------------------------------------------------
    # Time axis (hours)
    # --------------------------------------------------------
    global_frame_idx = video_idx * FRAMES_PER_VIDEO + frame_idx
    time_hours = global_frame_idx / FPS / 3600.0

    # --------------------------------------------------------
    # Pixel → mm conversion (circular well)
    # --------------------------------------------------------
    cx = main[:, 4][confident]
    cy = main[:, 5][confident]

    well_diam_px = robust_well_diameter_px(cx, cy)
    px_to_mm = WELL_DIAMETER_MM / well_diam_px
    length_mm = length_px * px_to_mm

    # --------------------------------------------------------
    # Bin + smooth (TIME-based)
    # --------------------------------------------------------
    BIN_HOURS = (BIN_FRAMES / FPS) / 3600.0

    bx, by, br = bin_trace_by_time(
        time_hours,
        length_mm,
        region,
        BIN_HOURS,
    )

    by_smooth = gaussian_filter1d(by, sigma=SMOOTH_SIGMA)

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------
    plt.figure(figsize=(12, 4))

    for i in range(len(bx) - 1):
        color = REGION_COLORS.get(br[i], "gray")
        plt.plot(bx[i : i + 2], by_smooth[i : i + 2], lw=2, color=color)

    # --------------------------------------------------------
    # Fission events (if present)
    # --------------------------------------------------------
    if fission is not None and len(fission) > 0:
        print("Number of fission events:", len(fission))

        # frame_idx is column 1
        fission_hours = fission[:, 1] / FPS / 3600.0
        y_marker = np.nanmax(by_smooth) * 1.05

        plt.scatter(
            fission_hours,
            np.full_like(fission_hours, y_marker),
            color="red",
            s=30,
            zorder=5,
        )
    else:
        print("Number of fission events: 0")

    # --------------------------------------------------------
    # Axes, ticks, legend
    # --------------------------------------------------------
    max_hours = time_hours.max()
    day_ticks = np.arange(0, max_hours + 24, 24)

    plt.xticks(day_ticks)
    plt.xlabel("Time (hours)")
    plt.ylabel("Head–Centroid–Tail length (mm)")
    plt.title(f"{path.stem} — Main worm length")

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
