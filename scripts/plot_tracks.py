from pathlib import Path
from pprint import pprint
import numpy as np
import h5py
import cv2

# ----------------------------
# User settings
# ----------------------------
H5_DIR = Path(
    r"D:/Tracking_Data/Sachi/T3/week_1_out_v3"
)  # folder containing well_0.h5, well_1.h5, ...
WELLS_NPY = Path("wells.npy")  # your ROI definitions (cx,cy,r) or (x,y,w,h)
OUT_DIR = H5_DIR / "track_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VIDEO_IDX = 0  # which video_idx to plot (change as needed)
DRAW_POINTS = False  # True: draw dots on trajectory points
LINE_THICKNESS = 2
POINT_RADIUS = 2

# If your H5 "tracking" is numeric NxM instead of compound dtype, set this mapping:
NUMERIC_COLUMNS = None
# Example:
# NUMERIC_COLUMNS = {
#     "video_idx": 0, "frame_idx": 1, "track_id": 2,
#     "cx": 3, "cy": 4, "hx": 5, "hy": 6, "tx": 7, "ty": 8,
#     "region": 9, "ht_conf": 10
# }


# ----------------------------
# Helpers
# ----------------------------
def roi_size_and_origin(roi):
    """
    Matches your crop_well():
      - circle roi: (cx, cy, r) => origin=(cx-r,cy-r), size=(2r,2r)
      - rect roi:   (x, y, w, h) => origin=(x,y), size=(w,h)
    """
    roi = np.asarray(roi).astype(int).tolist()
    if len(roi) == 3:
        cx, cy, r = roi
        return (2 * r, 2 * r), (cx - r, cy - r)
    elif len(roi) == 4:
        x, y, w, h = roi
        return (w, h), (x, y)
    else:
        raise ValueError(f"Unsupported ROI format: {roi}")


def id_to_bgr(track_id: int):
    """
    Stable bright color per track_id using HSV -> BGR.
    """
    h = (track_id * 37) % 180  # 0..179
    hsv = np.uint8([[[h, 220, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].tolist()
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def load_tracking_rows(h5_path: Path, video_idx: int):
    """
    Returns a structured dict-like view:
      arrays for frame_idx, track_id, cx, cy
    Works for compound dtype or numeric NxM.
    """
    with h5py.File(h5_path, "r") as f:
        if "tracking" not in f:
            raise KeyError(f"'tracking' dataset not found in {h5_path}")
        ds = f["tracking"]
        pprint(ds[:20])
        arr = ds[()]

    if arr.shape[0] == 0:
        return None

    # Compound dtype
    if arr.dtype.names is not None:
        required = {"video_idx", "frame_idx", "track_id", "cx", "cy"}
        missing = required.difference(arr.dtype.names)
        if missing:
            raise ValueError(f"{h5_path} missing required fields: {missing}")

        arr = arr[arr["video_idx"] == int(video_idx)]
        if arr.shape[0] == 0:
            return None

        # sort by frame
        arr = np.sort(arr, order="frame_idx")
        return {
            "frame_idx": arr["frame_idx"].astype(int),
            "track_id": arr["track_id"].astype(int),
            "cx": arr["cx"].astype(float),
            "cy": arr["cy"].astype(float),
        }

    # Numeric NxM
    if NUMERIC_COLUMNS is None:
        raise ValueError(
            f"{h5_path} tracking dataset is numeric; set NUMERIC_COLUMNS mapping at top of script."
        )

    c = NUMERIC_COLUMNS
    arr = arr[arr[:, c["video_idx"]].astype(int) == int(video_idx)]
    if arr.shape[0] == 0:
        return None

    order = np.argsort(arr[:, c["frame_idx"]].astype(int))
    arr = arr[order]

    return {
        "frame_idx": arr[:, c["frame_idx"]].astype(int),
        "track_id": arr[:, c["track_id"]].astype(int),
        "cx": arr[:, c["cx"]].astype(float),
        "cy": arr[:, c["cy"]].astype(float),
    }


def draw_tracks_on_well_canvas(well_roi, rows_dict):
    """
    Produces an image (BGR) of size equal to the well crop,
    with full trajectories drawn colored by ID.
    """
    (w, h), (ox, oy) = roi_size_and_origin(well_roi)

    # Canvas is just black; if you want, you can draw circle boundary later
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    if rows_dict is None:
        return canvas

    frame_idx = rows_dict["frame_idx"]
    track_id = rows_dict["track_id"]
    cx = rows_dict["cx"]
    cy = rows_dict["cy"]

    # Group by track_id
    unique_ids = np.unique(track_id)

    for tid in unique_ids:
        mask = track_id == tid
        xs = cx[mask] - ox
        ys = cy[mask] - oy

        # Keep points inside canvas
        pts = np.column_stack([xs, ys]).astype(np.float32)
        pts = pts[np.isfinite(pts).all(axis=1)]
        if pts.shape[0] < 2:
            continue

        # Convert to int pixel coords
        pts_i = np.round(pts).astype(np.int32)

        # Clip
        pts_i[:, 0] = np.clip(pts_i[:, 0], 0, w - 1)
        pts_i[:, 1] = np.clip(pts_i[:, 1], 0, h - 1)

        color = id_to_bgr(int(tid))

        # Draw polyline
        cv2.polylines(
            canvas,
            [pts_i.reshape(-1, 1, 2)],
            isClosed=False,
            color=color,
            thickness=LINE_THICKNESS,
        )

        # Optional: draw points
        if DRAW_POINTS:
            for px, py in pts_i:
                cv2.circle(canvas, (int(px), int(py)), POINT_RADIUS, color, -1)

        # Label at last point
        lx, ly = int(pts_i[-1, 0]), int(pts_i[-1, 1])
        cv2.putText(
            canvas,
            str(int(tid)),
            (lx + 3, ly - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )

    # If circular ROI, draw boundary for context
    roi = np.asarray(well_roi).astype(int).tolist()
    if len(roi) == 3:
        _, _, r = roi
        cv2.circle(canvas, (r, r), r, (80, 80, 80), 1)

    return canvas


def stitch_grid(images):
    """
    Simple square-ish grid stitch (BGR).
    """
    n = len(images)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    h, w = images[0].shape[:2]
    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, img in enumerate(images):
        r = i // cols
        c = i % cols
        grid[r * h : (r + 1) * h, c * w : (c + 1) * w] = img
    return grid


# ----------------------------
# Main
# ----------------------------
def main():
    wells = np.load(WELLS_NPY)
    per_well_imgs = []

    for well_id in range(len(wells)):
        h5_path = H5_DIR / f"well_{well_id}.h5"
        rows = load_tracking_rows(h5_path, VIDEO_IDX)

        canvas = draw_tracks_on_well_canvas(wells[well_id], rows)
        out_path = OUT_DIR / f"well_{well_id:02d}_tracks_video{VIDEO_IDX:04d}.png"
        cv2.imwrite(str(out_path), canvas)
        per_well_imgs.append(canvas)

        print(f"Saved: {out_path}")

    # Optional stitched overview
    stitched = stitch_grid(per_well_imgs)
    stitched_path = OUT_DIR / f"all_wells_tracks_video{VIDEO_IDX:04d}.png"
    cv2.imwrite(str(stitched_path), stitched)
    print(f"Saved: {stitched_path}")

    # Optional preview
    cv2.imshow("Tracks (all wells)", stitched)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
