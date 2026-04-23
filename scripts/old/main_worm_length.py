import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from collections import Counter
from matplotlib.lines import Line2D


DATA_DIR = Path("D:/Tracking_Data/Sachi/T3/week_1_out")
WELL_FILES = sorted(DATA_DIR.glob("well_*.h5"))

WELL_DIAMETER_MM = 35.0
FPS = 2.0  
BIN_FRAMES = 1000  
SMOOTH_SIGMA = 3  
FRAMES_PER_VIDEO = int(FPS * 3530)

REGION_CORE = 1
REGION_WALL = 2

REGION_COLORS = {
    REGION_CORE: "green",
    REGION_WALL: "orange",
}


events = {
    5: {
        "green": [(7, 5000), (24, 2500), (125, 1700)],
        "red": [(82, 1000), (113, 5200), (120, 400)],
    },
    4: {
        "red": [(25, 1300), (28, 0)],
        "green": [(36, 6100), (22, 4200), (65, 1300), (103, 1500)],
    },
    3: {"red": [(69, 4200)]},
    2: {"red": [(43, 3500)], "green": [(66, 600), (102, 4400)]},
    1: {"green": [(80, 200)]},
    0: {"green": [(80, 800), (109, 4400)]},
}


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


for path in WELL_FILES:
    with h5py.File(path, "r") as h5:
        print_h5_info(h5, label=path.name)
        main = h5["main"][:]  # type: ignore
        fission = h5["fission"][:] if "fission" in h5 else None  # type: ignore

    length_px, confident, region = compute_length_px(main)
    well = int(path.stem.split("_")[1])
    length_px = length_px[confident]
    region = region[confident] #type: ignore
    frame_idx = main[:, 1][confident]  # type: ignore
    video_idx = main[:, 0][confident]  # type: ignore

    global_frame_idx = video_idx * FRAMES_PER_VIDEO + frame_idx  # type: ignore
    time_hours = global_frame_idx / FPS / 3600.0

    cx = main[:, 4][confident]  # type: ignore
    cy = main[:, 5][confident]  # type: ignore

    well_diam_px = robust_well_diameter_px(cx, cy)
    px_to_mm = WELL_DIAMETER_MM / well_diam_px
    length_mm = length_px * px_to_mm

    BIN_HOURS = (BIN_FRAMES / FPS) / 3600.0

    bx, by, br = bin_trace_by_time(
        time_hours,
        length_mm,
        region,
        BIN_HOURS,
    )

    by_smooth = gaussian_filter1d(by, sigma=SMOOTH_SIGMA)

    plt.figure(figsize=(12, 4))

    for i in range(len(bx) - 1):
        color = REGION_COLORS.get(br[i], "gray")
        plt.plot(bx[i : i + 2], by_smooth[i : i + 2], lw=2, color=color)

    if fission is not None and len(fission) > 0:  # type: ignore
        print("Number of fission events:", len(fission))  # type: ignore

        fission_hours = fission[:, 1] / FPS / 3600.0  # type: ignore
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

    if well in events:

        y_event = np.nanmax(by_smooth) * 1.02

        for color, pts in events[well].items():
            for v, f in pts:

                global_frame = v * FRAMES_PER_VIDEO + f
                t = global_frame / FPS / 3600.0

                if t < bx.min() or t > bx.max():
                    continue

                idx = np.argmin(np.abs(bx - t))
                y = by_smooth[idx]

                plt.scatter(
                    t,
                    y + 0.2,
                    c=color,
                    s=30,
                    zorder=6,
                )

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
