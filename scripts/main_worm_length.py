import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from collections import Counter
import re

# -----------------------------
# Inputs
# -----------------------------
DATA_DIR = Path("D:/Tracking_Data/Sachi/T3/week_1_out_v3")
WELL_FILES = sorted(DATA_DIR.glob("well_*.h5"))

# If available, this makes pixel->mm calibration much more accurate:
WELLS_NPY = Path("wells.npy")  # contains circle ROIs like [cx, cy, r] or rect [x,y,w,h]

# -----------------------------
# Constants / settings
# -----------------------------
WELL_DIAMETER_MM = 35.0
FPS = 2.0
BIN_FRAMES = 1000
SMOOTH_SIGMA = 3

# If your videos all have the same frame count, use this for continuous time across videos:
FRAMES_PER_VIDEO = int(FPS * 3530)  # adjust if needed

REGION_CORE = 1
REGION_WALL = 2
REGION_COLORS = {REGION_CORE: "green", REGION_WALL: "orange", 0: "gray"}
VIDEO_IDX = 1

OUT_DIR = DATA_DIR / "length_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Utilities
# -----------------------------
def print_h5_info(h5, label=""):
    print(f"\n=== HDF5 INFO {label} ===")
    for name, obj in h5.items():
        print(f"Dataset '{name}': shape={obj.shape}, dtype={obj.dtype}")
    print("========================")


def parse_well_id(path: Path) -> int:
    m = re.search(r"well_(\d+)\.h5$", path.name)
    if not m:
        raise ValueError(f"Could not parse well id from: {path.name}")
    return int(m.group(1))


def well_diameter_px_from_wells_npy(wells: np.ndarray, well_id: int) -> float:
    """
    wells[well_id] is either [cx, cy, r] or [x, y, w, h]
    Your crop_well uses a 2r x 2r crop for circle ROIs => diameter_px = 2r
    """
    roi = wells[well_id].astype(int).tolist()
    if len(roi) == 3:
        _, _, r = roi
        return float(2 * r)
    if len(roi) == 4:
        _, _, w, h = roi
        return float(max(w, h))
    raise ValueError(f"Unsupported ROI format for well {well_id}: {roi}")


def robust_well_diameter_px(cx: np.ndarray, cy: np.ndarray) -> float:
    """
    Fallback estimator if you don't have wells.npy.
    Uses robust range (95th - 5th percentile) so a worm that doesn't explore the whole well
    won't explode the estimate too badly.
    """
    cx = cx[np.isfinite(cx)]
    cy = cy[np.isfinite(cy)]
    if cx.size < 10 or cy.size < 10:
        return np.nan
    dx = np.percentile(cx, 95) - np.percentile(cx, 5)
    dy = np.percentile(cy, 95) - np.percentile(cy, 5)
    return float(max(dx, dy))


def compute_length_px(tr: np.ndarray):
    """
    tr is structured array with fields cx,cy,hx,hy,tx,ty,region.
    Length computed only when head and tail exist (>=0).
    """
    cx = tr["cx"].astype(float)
    cy = tr["cy"].astype(float)
    hx = tr["hx"].astype(float)
    hy = tr["hy"].astype(float)
    tx = tr["tx"].astype(float)
    ty = tr["ty"].astype(float)
    region = tr["region"].astype(int)

    valid = (hx >= 0) & (hy >= 0) & (tx >= 0) & (ty >= 0)

    length_px = np.full(cx.shape, np.nan, dtype=float)
    d_hc = np.sqrt((hx - cx) ** 2 + (hy - cy) ** 2)
    d_ct = np.sqrt((cx - tx) ** 2 + (cy - ty) ** 2)
    length_px[valid] = d_hc[valid] + d_ct[valid]

    return length_px, valid, region


def time_hours(video_idx: np.ndarray, frame_idx: np.ndarray) -> np.ndarray:
    """
    Continuous time across videos.
    """
    global_frame = video_idx.astype(int) * FRAMES_PER_VIDEO + frame_idx.astype(int)
    return global_frame / FPS / 3600.0


def bin_trace_by_time(time_h, length_mm, region, bin_hours):
    """
    Time-based binning with region voting restricted to valid length frames.
    """
    xb, yb, rb = [], [], []
    edges = np.arange(time_h.min(), time_h.max() + bin_hours, bin_hours)

    for t0, t1 in zip(edges[:-1], edges[1:]):
        mask = (time_h >= t0) & (time_h < t1) & np.isfinite(length_mm)
        if mask.sum() < 10:
            continue

        xb.append(0.5 * (t0 + t1))
        yb.append(np.nanmean(length_mm[mask]))

        r_mode = Counter(region[mask].astype(int)).most_common(1)[0][0]
        rb.append(r_mode)

    return np.asarray(xb), np.asarray(yb), np.asarray(rb)


def smooth_nan_safe(y: np.ndarray, sigma: float) -> np.ndarray:
    """
    Smooth y while respecting NaNs by smoothing data and weights separately.
    """
    y = y.astype(float)
    mask = np.isfinite(y).astype(float)
    y0 = np.nan_to_num(y, nan=0.0)

    ys = gaussian_filter1d(y0, sigma=sigma, mode="nearest")
    ms = gaussian_filter1d(mask, sigma=sigma, mode="nearest")

    with np.errstate(invalid="ignore", divide="ignore"):
        out = ys / ms
    out[ms < 1e-6] = np.nan
    return out


# -----------------------------
# Load wells.npy if available
# -----------------------------
WELLS = None
if WELLS_NPY.exists():
    WELLS = np.load(WELLS_NPY)
    print(f"Loaded wells from {WELLS_NPY} (shape={WELLS.shape})")
else:
    print(
        "wells.npy not found; will estimate well diameter from centroid span (less accurate)."
    )

# -----------------------------
# Main plotting loop
# -----------------------------
BIN_HOURS = BIN_FRAMES / FPS / 3600.0

for path in WELL_FILES:
    well_id = parse_well_id(path)

    with h5py.File(path, "r") as h5:
        print_h5_info(h5, label=path.name)
        tr = h5["tracking"][()]  # structured rows

    if tr.shape[0] == 0:
        print(f"Well {well_id}: no tracking data, skipping.")
        continue

    # Extract time
    t_h = time_hours(tr["video_idx"], tr["frame_idx"])

    # Compute length in pixels
    length_px, valid, region = compute_length_px(tr)

    # Determine well diameter in pixels for px->mm conversion
    if WELLS is not None and well_id < len(WELLS):
        diam_px = well_diameter_px_from_wells_npy(WELLS, well_id)
    else:
        diam_px = robust_well_diameter_px(
            tr["cx"].astype(float), tr["cy"].astype(float)
        )

    if not np.isfinite(diam_px) or diam_px <= 0:
        print(f"Well {well_id}: could not determine well diameter in px, skipping.")
        continue

    px_to_mm = WELL_DIAMETER_MM / diam_px
    length_mm = length_px * px_to_mm

    # Bin and smooth
    xb, yb, rb = bin_trace_by_time(t_h, length_mm, region, BIN_HOURS)
    ys = smooth_nan_safe(yb, sigma=SMOOTH_SIGMA) if len(yb) > 3 else yb

    # Plot
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_title(f"Well {well_id} — Worm length (mm) vs time")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Length (mm)")

    # scatter binned points colored by region mode
    for reg_code in np.unique(rb) if rb.size else []:
        m = rb == reg_code
        ax.scatter(
            xb[m],
            yb[m],
            s=20,
            alpha=0.8,
            color=REGION_COLORS.get(int(reg_code), "gray"),
            label=f"Region {int(reg_code)}",
        )

    # smoothed line
    ax.plot(xb, ys, color="black", linewidth=2, label="Smoothed")

    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    out_png = OUT_DIR / f"well_{well_id:02d}_length_video{VIDEO_IDX:04d}.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    print(f"Saved: {out_png}")
