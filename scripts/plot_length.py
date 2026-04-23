from pathlib import Path
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from Module.dataset.Drawer import ROIDrawer


def interpolate_keypoints(frames, values):
    """
    Linear interpolation using nearest valid neighbors.
    values: 1D array (e.g., hx)
    frames: corresponding frame indices
    """
    valid = values != -1

    if valid.sum() < 2:
        return values

    return np.interp(frames, frames[valid], values[valid])


BASE = Path("D:/Tracking_Data/Sachi/T3/week_1_out/")
WELLS = [0, 1, 2, 3, 4, 5]

FPS = 2
ROI_MM = 20

FRAME_COL = 1
HX, HY = 2, 3
CX, CY = 4, 5
TX, TY = 6, 7
VALID = -1


rois = ROIDrawer.load("wells.npy")


def mm_per_px(well):
    r = rois[well][2]
    return ROI_MM / (2 * r)


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


def compute_offsets(data):
    vids = data[:, 0].astype(int)
    frames = data[:, 1].astype(int)

    max_vid = vids.max() + 1
    lengths = np.zeros(max_vid, dtype=int)

    for v in range(max_vid):
        mask = vids == v
        if np.any(mask):
            lengths[v] = frames[mask].max() + 1

    offsets = np.cumsum(np.concatenate([[0], lengths[:-1]]))
    return offsets


max_events = max(
    sum(len(v) for v in events[w].values()) if w in events else 1 for w in WELLS
)

fig, axes = plt.subplots(
    len(WELLS), max_events, figsize=(4 * max_events, 2.5 * len(WELLS)), sharey=True
)

if len(WELLS) == 1:
    axes = axes[np.newaxis, :]

for row, well in enumerate(WELLS):

    path = BASE / f"well_{well}.h5"

    with h5py.File(path, "r") as f:
        data = f["main"][:]  # type: ignore

    valid = data[:, VALID] == 1  # type: ignore
    data = data[valid]  # type: ignore

    vids = data[:, 0].astype(int)  # type: ignore
    frames_local = data[:, FRAME_COL].astype(int)  # type: ignore

    offsets = compute_offsets(data)
    frames = frames_local + offsets[vids]

    hx = data[:, HX].copy()  # type: ignore
    hy = data[:, HY].copy()  # type: ignore
    tx = data[:, TX].copy()  # type: ignore
    ty = data[:, TY].copy()  # type: ignore

    for v in np.unique(vids):

        mask = vids == v
        f_local = frames_local[mask]

        hx[mask] = interpolate_keypoints(f_local, hx[mask])
        hy[mask] = interpolate_keypoints(f_local, hy[mask])
        tx[mask] = interpolate_keypoints(f_local, tx[mask])
        ty[mask] = interpolate_keypoints(f_local, ty[mask])

    valid_kp = (hx != -1) & (hy != -1) & (tx != -1) & (ty != -1)

    frames = frames[valid_kp]
    length = np.sqrt(
        (hx[valid_kp] - tx[valid_kp]) ** 2 + (hy[valid_kp] - ty[valid_kp]) ** 2
    )

    length *= mm_per_px(well)

    max_f = frames.max()

    sum_len = np.bincount(frames, weights=length, minlength=max_f + 1)
    count = np.bincount(frames, minlength=max_f + 1)

    mean_len = np.divide(sum_len, count, out=np.zeros_like(sum_len), where=count > 0)

    window = 10
    kernel = np.ones(window) / window
    smooth = np.convolve(mean_len, kernel, mode="same")
    smooth = smooth[10:-10]

    time_hours = np.arange(len(smooth)) / FPS / 3600

    event_list = []

    if well in events:
        for color, pts in events[well].items():
            for v, f in pts:

                mask = vids == v
                if not np.any(mask):
                    continue

                video_frames = frames_local[mask]
                idx_local = np.argmin(np.abs(video_frames - f))
                frame_local = int(video_frames[idx_local])

                global_frame = frame_local + offsets[v]

                if global_frame >= len(smooth):
                    continue

                t = global_frame / FPS / 3600
                event_list.append((t, color))

    for col in range(max_events):

        ax = axes[row, col]

        if col >= len(event_list):
            ax.axis("off")
            continue

        t_event, color = event_list[col]

        t_min = t_event - 2
        t_max = t_event + 2

        mask = (time_hours >= t_min) & (time_hours <= t_max)

        t_window = time_hours[mask] - t_event
        y_window = smooth[mask]

        ax.plot(t_window, y_window)

        idx_event = np.argmin(np.abs(time_hours - t_event))
        y_event = smooth[idx_event]

        ax.scatter(0, y_event + 0.2, c=color, s=30, zorder=5)

        ax.set_title(f"W{well}", fontsize=8)

        if col == 0:
            ax.set_ylabel(f"Length (mm)")

        if row == len(WELLS) - 1:
            ax.set_xlabel("Time (hours)")


plt.tight_layout()
plt.show()
