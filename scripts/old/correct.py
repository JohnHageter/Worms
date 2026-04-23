"""

Doesn't really work

"""



from pathlib import Path
import sys
import cv2
import h5py
import numpy as np
import copy
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from Module.dataset.Drawer import ROIDrawer
from Module.utils import crop_well


VIDEO_DIR = Path("D:/Tracking_Data/Sachi/T4_4233A_Rig_AgaroseWell/first/")
H5_PATH = "D:/Tracking_Data/Sachi/T4_4233A_Rig_AgaroseWell/out_v2/well_0.h5"
CORRECTIONS_PATH = Path("corrections_log.json")

SCRUB_STEP = 200

roi = [391, 636, 370]

wells = ROIDrawer.load("AgarWell.npy")

video_paths = sorted(VIDEO_DIR.glob("*.mp4"))
caps = [cv2.VideoCapture(str(p)) for p in video_paths]

video_lengths = [int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in caps]
offsets = np.cumsum([0] + video_lengths[:-1])
total_frames = int(offsets[-1] + video_lengths[-1])


h5 = h5py.File(H5_PATH, "a")
tracking = h5["tracking"][:]


frame_map = {}

n_vids = len(offsets)

for r in tracking:
    vid = int(r["video_idx"])
    local_frame = int(r["frame_idx"])

    if vid < 0 or vid >= n_vids:
        continue

    gf = int(offsets[vid] + local_frame)
    frame_map.setdefault(gf, []).append(r)

current_frame = 0
selected_ids = []
corrections = []


def to_dict(r):
    return {name: r[name] for name in r.dtype.names}


def color_from_id(tid):
    tid = int(tid)
    return (
        int((37 * tid) % 255),
        int((91 * tid) % 255),
        int((181 * tid) % 255),
    )


def apply_all(rows, corrections, frame):
    rows = copy.deepcopy(rows)

    for e in corrections:
        if e["start"] > frame:
            continue  

        if e["type"] == "swap":
            a, b = e["a"], e["b"]

            for r in rows:
                if r["track_id"] == a:
                    r["track_id"] = -b
                elif r["track_id"] == b:
                    r["track_id"] = -a

            for r in rows:
                if r["track_id"] == -a:
                    r["track_id"] = a
                elif r["track_id"] == -b:
                    r["track_id"] = b

        elif e["type"] == "merge":
            src, dst = e["src"], e["dst"]

            for r in rows:
                if r["track_id"] == src:
                    r["track_id"] = dst

        elif e["type"] == "delete":
            tid = e["id"]

            for r in rows:
                if r["track_id"] == tid:
                    r["track_id"] = -1

        elif e["type"] == "relabel":
            src, dst = e["src"], e["dst"]
            for r in rows:
                if r["track_id"] == src:
                    r["track_id"] = dst

    return rows


def get_video_frame(gf):
    if gf < 0:
        return None, -1, -1

    gf = min(gf, total_frames - 1)

    vid = np.searchsorted(offsets, gf, side="right") - 1

    # FIX: clamp correctly
    vid = max(0, min(int(vid), len(caps) - 1))

    local = gf - offsets[vid]

    if local < 0 or local >= video_lengths[vid]:
        return None, vid, local

    cap = caps[vid]
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(local))

    ret, frame = cap.read()
    if not ret or frame is None:
        return None, vid, local

    frame = crop_well(frame, roi)

    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    return frame, vid, local



def mouse_cb(event, x, y, flags, param):
    global selected_ids

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    best_id = None
    best_d = 1e9

    for r in current_rows:
        cx, cy = int(r["cx"]), int(r["cy"])
        d = (cx - x) ** 2 + (cy - y) ** 2

        if d < best_d and d < 300:
            best_d = d
            best_id = int(r["track_id"])

    if best_id is not None:
        if best_id in selected_ids:
            selected_ids.remove(best_id)
        elif len(selected_ids) < 2:
            selected_ids.append(best_id)


cv2.namedWindow("viewer")
cv2.setMouseCallback("viewer", mouse_cb)


while True:

    frame, vid, fid = get_video_frame(current_frame)

    if frame is None:
        current_frame = min(current_frame + SCRUB_STEP, total_frames - 1)
        continue


    base_rows = frame_map.get(current_frame, [])
    base_rows = [to_dict(r) for r in base_rows]

    current_rows = apply_all(base_rows, corrections, current_frame)

    globals()["current_rows"] = current_rows


    for r in current_rows:
        x, y = int(r["cx"]), int(r["cy"])
        hx, hy = int(r["hx"]), int(r["hy"])
        tx, ty = int(r["tx"]), int(r["ty"])
        tid = int(r["track_id"])

        color = color_from_id(tid)

        if tid in selected_ids:
            cv2.circle(frame, (x, y), 8, (0, 255, 255), 2)

        cv2.circle(frame, (x, y), 3, color, -1)
        cv2.putText(
            frame, str(tid), (x + 4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
        )

        if hx >= 0 and hy >= 0:
            cv2.circle(frame, (int(hx), int(hy)), 3, (0, 0, 255), -1)

        if tx >= 0 and ty >= 0:
            cv2.circle(frame, (int(tx), int(ty)), 3, (255, 0, 0), -1)


    cv2.putText(
        frame,
        f"Frame {current_frame} | selected {selected_ids}",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        "A/D scrub | S swap | M merge | X delete | ESC save",
        (10, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
    )

    cv2.imshow("viewer", frame)

    key = cv2.waitKey(1) & 0xFF


    if key == ord("d"):
        current_frame = min(current_frame + SCRUB_STEP, total_frames - 1)

    elif key == ord("a"):
        current_frame = max(current_frame - SCRUB_STEP, 0)


    elif key == ord("s") and len(selected_ids) == 2:
        a, b = selected_ids
        corrections.append({"type": "swap", "a": a, "b": b, "start": current_frame})
        selected_ids.clear()


    elif key == ord("m") and len(selected_ids) == 2:
        src, dst = selected_ids
        corrections.append(
            {"type": "merge", "src": src, "dst": dst, "start": current_frame}
        )
        selected_ids.clear()


    elif key == ord("x") and len(selected_ids) == 1:
        corrections.append({"type": "delete", "id": tid, "start": current_frame})
        selected_ids.clear()


    elif key == 27:
        with open(CORRECTIONS_PATH, "w") as f:
            json.dump(corrections, f, indent=2)

        if "corrections" not in h5:
            h5.create_dataset(
                "corrections", shape=(0,), maxshape=(None,), dtype=h5py.string_dtype()
            )

        ds = h5["corrections"]
        start = ds.shape[0]
        ds.resize((start + len(corrections),))

        for i, c in enumerate(corrections):
            ds[start + i] = json.dumps(c)

        h5.flush()

        print(f"Saved {len(corrections)} corrections")
        break

cv2.destroyAllWindows()
h5.close()
