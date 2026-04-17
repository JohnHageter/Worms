from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Any, Dict, Literal

import numpy as np
import cv2
import h5py

from Module.utils import crop_well  
from Module.dataset.track_corrections import (
    CorrectionPlan,
    MergeOp,
    SwapOp,
    apply_corrections,
    load_plan_from_h5,
    save_plan_to_h5,
    commit_corrections_inplace,
)


def roi_origin_xy(roi: Any) -> Tuple[int, int]:
    """
    Matches crop_well():
      circle roi: (cx, cy, r) => crop origin = (cx-r, cy-r)
      rect roi:   (x, y, w, h) => crop origin = (x, y)
    """
    if isinstance(roi, np.ndarray):
        roi = roi.tolist()
    roi = list(map(int, roi))
    if len(roi) == 3:
        cx, cy, r = roi
        return cx - r, cy - r
    if len(roi) == 4:
        x, y, w, h = roi
        return x, y
    raise ValueError(f"Unsupported ROI format: {roi!r}")


def id_color(track_id: int) -> Tuple[int, int, int]:
    """Stable vivid color per ID (HSV -> BGR)."""
    h = (track_id * 37) % 180
    hsv = np.uint8([[[h, 220, 255]]]) #type: ignore
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0] #type: ignore
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _to_int_pt(x: float, y: float) -> Tuple[int, int]:
    return int(round(float(x))), int(round(float(y)))


@dataclass
class Selection:
    well_id: int = 0
    track_id: Optional[int] = None
    frame_idx: int = 0


@dataclass
class WellState:
    well_id: int
    h5_path: str
    roi: Any

    plan: CorrectionPlan
    undo_stack: List[Tuple[Literal["merge", "swap"], object]]

    base_rows: np.ndarray          # /raw if exists else /tracking
    tracking_rows: np.ndarray      # /tracking if exists else empty



class MultiWellTrackReviewer:
    """
    Multi-well reviewer for videos where all wells are visible in each frame.

    - Loads per-well H5 files (well_0.h5, well_1.h5, ...)
    - Shows a stitched grid of well crops
    - Click to select a track in a specific well
    - Merge/swap operations apply to the currently selected well only
    - Save plan into that well's H5 (/corrections/json)
    - Commit in-place into that well's H5 (/raw preserved, /tracking rewritten)

    Keybinds
    --------
    SPACE : play/pause
    A/D   : step -1/+1 frame
    J/L   : jump -50/+50 frames
    [ / ] : jump -500/+500 frames

    TAB   : cycle selected well (0 -> 1 -> 2 -> ...)
    R     : toggle RAW vs PREVIEW for selected well
    T     : toggle COMMITTED /tracking view for selected well (if exists)

    Left click : select nearest track at that click position (in the clicked well)

    M     : merge selected track -> dst_id (prompt)
    S     : swap selected track with other_id (prompt) starting now
    E     : end last merge/swap at current frame (end_frame)
    U     : undo last op (selected well only)

    K     : save plan into selected well H5 (/corrections/json)
    P     : commit plan into selected well H5 (creates /raw if needed, rewrites /tracking)

    Q/ESC : quit

    Coordinate mode
    ---------------
    coords_are_crop=True: H5 coords are crop-relative (your current case)
    coords_are_crop=False: H5 coords are full-frame (subtract ROI origin)
    """

    def __init__(
        self,
        *,
        video_path: str | Path,
        h5_dir: str | Path,
        wells: Any,
        video_idx: int = 0,
        coords_are_crop: bool = True,
        window_name: str = "Multi-Well Track Reviewer",
        grid_cols: Optional[int] = None,
        step_play: int = 1,
    ):
        self.video_path = str(video_path)
        self.h5_dir = Path(h5_dir)
        self.wells = wells
        self.video_idx = int(video_idx)
        self.coords_are_crop = bool(coords_are_crop)

        self.window_name = window_name
        self.step_play = int(max(1, step_play))

        self.sel = Selection(well_id=0, track_id=None, frame_idx=0)
        self.playing = False

        # view mode per well (raw/preview/tracking)
        self.view_mode: Dict[int, Literal["raw", "preview", "tracking"]] = {}

        # open video
        self._cap = cv2.VideoCapture(self.video_path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.video_path}")

        self._frame_max = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1

        # load per-well h5 states
        self.well_states: List[WellState] = self._load_all_wells()

        # default view modes
        for ws in self.well_states:
            self.view_mode[ws.well_id] = "preview"

        # grid layout
        self.n_wells = len(self.well_states)
        self.cols = grid_cols if grid_cols is not None else int(np.ceil(np.sqrt(self.n_wells)))
        self.rows = int(np.ceil(self.n_wells / self.cols))

        # cell size (computed from first frame)
        self.cell_w = None
        self.cell_h = None

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._on_mouse)



    def _load_rows_from_dataset(self, h5_path: str, name: str) -> np.ndarray:
        with h5py.File(h5_path, "r") as f:
            if name not in f:
                return np.zeros((0,), dtype=[("video_idx","i4"),("frame_idx","i4"),("track_id","i4"),
                                             ("cx","f4"),("cy","f4"),("hx","f4"),("hy","f4"),
                                             ("tx","f4"),("ty","f4"),("region","i2"),("ht_conf","f4")])
            arr = f[name][()] #type: ignore
        if arr.shape[0] == 0: #type: ignore
            return arr #type: ignore
        if arr.dtype.names is None: #type: ignore
            raise ValueError(f"Dataset {name} in {h5_path} is not compound dtype.")
        arr = arr[arr["video_idx"] == self.video_idx] #type: ignore
        if arr.shape[0] == 0: #type: ignore
            return arr #type: ignore
        return np.sort(arr, order="frame_idx") #type: ignore

    def _load_best_base(self, h5_path: str) -> np.ndarray:
        with h5py.File(h5_path, "r") as f:
            base = "raw" if "raw" in f else "tracking"
        return self._load_rows_from_dataset(h5_path, base)

    def _load_all_wells(self) -> List[WellState]:
        states: List[WellState] = []
        for well_id, roi in enumerate(self.wells):
            h5_path = self.h5_dir / f"well_{well_id}.h5"
            if not h5_path.exists():
                raise FileNotFoundError(f"Missing well H5: {h5_path}")

            plan = load_plan_from_h5(str(h5_path))
            base_rows = self._load_best_base(str(h5_path))
            tracking_rows = self._load_rows_from_dataset(str(h5_path), "tracking")

            states.append(
                WellState(
                    well_id=well_id,
                    h5_path=str(h5_path),
                    roi=roi,
                    plan=plan,
                    undo_stack=[],
                    base_rows=base_rows,
                    tracking_rows=tracking_rows,
                )
            )
        return states

    def _rows_at_frame(self, rows: np.ndarray, frame_idx: int) -> np.ndarray:
        if rows.shape[0] == 0:
            return rows[:0]
        fi = rows["frame_idx"].astype(int)
        left = np.searchsorted(fi, frame_idx, side="left")
        right = np.searchsorted(fi, frame_idx, side="right")
        return rows[left:right]


    def _read_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = self._cap.read()
        if not ok:
            return None
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame


    def _get_display_rows_for_well(self, ws: WellState, frame_idx: int) -> np.ndarray:
        mode = self.view_mode[ws.well_id]

        if mode == "tracking":
            return self._rows_at_frame(ws.tracking_rows, frame_idx)

        base = self._rows_at_frame(ws.base_rows, frame_idx)

        if mode == "raw":
            return base

        # preview: apply corrections in-memory
        if base.shape[0] == 0 or (not ws.plan.merges and not ws.plan.swaps):
            return base
        return apply_corrections(base, ws.plan)


    def _compose_grid(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Pads each crop to (cell_h, cell_w) and stitches into rows/cols grid.
        """
        assert self.cell_w is not None and self.cell_h is not None

        pad_images = []
        for img in images:
            h, w = img.shape[:2]
            canvas = np.zeros((self.cell_h, self.cell_w, 3), dtype=np.uint8)
            canvas[:h, :w] = img
            pad_images.append(canvas)

        grid = np.zeros((self.rows * self.cell_h, self.cols * self.cell_w, 3), dtype=np.uint8)
        for i, img in enumerate(pad_images):
            rr = i // self.cols
            cc = i % self.cols
            y0 = rr * self.cell_h
            x0 = cc * self.cell_w
            grid[y0:y0 + self.cell_h, x0:x0 + self.cell_w] = img
        return grid

    def _draw_well(self, frame_gray: np.ndarray, ws: WellState, frame_idx: int) -> np.ndarray:
        crop = crop_well(frame_gray, ws.roi)
        crop_bgr = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)

        ox, oy = roi_origin_xy(ws.roi)
        rows = self._get_display_rows_for_well(ws, frame_idx)

        for r in rows:
            tid = int(r["track_id"])
            color = id_color(tid)

            if self.coords_are_crop:
                cx, cy = float(r["cx"]), float(r["cy"])
                hx, hy = float(r["hx"]), float(r["hy"])
                tx, ty = float(r["tx"]), float(r["ty"])
            else:
                cx, cy = float(r["cx"]) - ox, float(r["cy"]) - oy
                hx, hy = float(r["hx"]) - ox, float(r["hy"]) - oy
                tx, ty = float(r["tx"]) - ox, float(r["ty"]) - oy

            cxi, cyi = _to_int_pt(cx, cy)
            cv2.circle(crop_bgr, (cxi, cyi), 2, color, -1)
            cv2.putText(crop_bgr, str(tid), (cxi + 4, cyi - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

            if hx >= 0 and hy >= 0:
                cv2.circle(crop_bgr, _to_int_pt(hx, hy), 3, (255, 0, 0), -1)
            if tx >= 0 and ty >= 0:
                cv2.circle(crop_bgr, _to_int_pt(tx, ty), 3, (0, 0, 255), -1)

        # header
        mode = self.view_mode[ws.well_id].upper()
        header = f"WELL {ws.well_id} | {mode}"
        cv2.putText(crop_bgr, header, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # highlight selected well
        if ws.well_id == self.sel.well_id:
            cv2.rectangle(crop_bgr, (1, 1), (crop_bgr.shape[1]-2, crop_bgr.shape[0]-2), (0, 255, 0), 2)
            if self.sel.track_id is not None:
                cv2.putText(crop_bgr, f"Selected ID: {self.sel.track_id}", (8, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        return crop_bgr

    def _render(self, frame_gray: np.ndarray, frame_idx: int) -> np.ndarray:
        well_imgs = [self._draw_well(frame_gray, ws, frame_idx) for ws in self.well_states]

        if self.cell_w is None or self.cell_h is None:
            # compute max cell dims
            self.cell_h = max(img.shape[0] for img in well_imgs)
            self.cell_w = max(img.shape[1] for img in well_imgs)

        canvas = self._compose_grid(well_imgs)

        # global footer
        cv2.putText(canvas, f"Frame {frame_idx}/{self._frame_max}", (10, canvas.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return canvas


    def _on_mouse(self, event, x, y, flags, userdata):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self.cell_w is None or self.cell_h is None:
            return

        # determine clicked well cell
        col = x // self.cell_w
        row = y // self.cell_h
        well_id = row * self.cols + col
        if well_id < 0 or well_id >= self.n_wells:
            return

        self.sel.well_id = int(well_id)

        # coordinates inside crop
        local_x = x - col * self.cell_w
        local_y = y - row * self.cell_h

        ws = self.well_states[self.sel.well_id]
        rows = self._get_display_rows_for_well(ws, self.sel.frame_idx)
        if rows.shape[0] == 0:
            self.sel.track_id = None
            return

        # pick nearest centroid
        pts = []
        ids = []
        ox, oy = roi_origin_xy(ws.roi)
        for r in rows:
            if self.coords_are_crop:
                pts.append([float(r["cx"]), float(r["cy"])])
            else:
                pts.append([float(r["cx"]) - ox, float(r["cy"]) - oy])
            ids.append(int(r["track_id"]))

        pts = np.asarray(pts, dtype=float)
        d = np.linalg.norm(pts - np.array([local_x, local_y], dtype=float), axis=1)
        idx = int(np.argmin(d))
        self.sel.track_id = ids[idx]


    def _prompt_int(self, prompt: str) -> int:
        return int(input(prompt).strip())

    def _ws(self) -> WellState:
        return self.well_states[self.sel.well_id]

    def merge_selected_to(self, dst_id: int):
        if self.sel.track_id is None:
            print("No selected track.")
            return
        ws = self._ws()
        src = int(self.sel.track_id)
        if src == dst_id:
            print("Source and destination are same.")
            return
        op = MergeOp(src_id=src, dst_id=dst_id, start_frame=self.sel.frame_idx, end_frame=2**31 - 1)
        ws.plan.merges.append(op)
        ws.undo_stack.append(("merge", op))
        print(f"[well {ws.well_id}] MERGE: {src} -> {dst_id} from frame {self.sel.frame_idx}")

    def swap_selected_with(self, other_id: int):
        if self.sel.track_id is None:
            print("No selected track.")
            return
        ws = self._ws()
        a = int(self.sel.track_id)
        b = int(other_id)
        if a == b:
            print("Same ID.")
            return
        op = SwapOp(id_a=a, id_b=b, start_frame=self.sel.frame_idx, end_frame=2**31 - 1)
        ws.plan.swaps.append(op)
        ws.undo_stack.append(("swap", op))
        print(f"[well {ws.well_id}] SWAP: {a} <-> {b} from frame {self.sel.frame_idx}")

    def end_last_op_here(self):
        ws = self._ws()
        if not ws.undo_stack:
            print("No ops to end.")
            return
        kind, op = ws.undo_stack[-1]
        op.end_frame = self.sel.frame_idx #type: ignore
        print(f"[well {ws.well_id}] Set end_frame={self.sel.frame_idx} for last {kind}")

    def undo(self):
        ws = self._ws()
        if not ws.undo_stack:
            return
        kind, op = ws.undo_stack.pop()
        if kind == "merge":
            ws.plan.merges.remove(op) #type: ignore
        else:
            ws.plan.swaps.remove(op) #type: ignore
        print(f"[well {ws.well_id}] Undid {kind}")

    def save_plan(self):
        ws = self._ws()
        save_plan_to_h5(ws.h5_path, ws.plan)
        print(f"[well {ws.well_id}] Saved plan to H5: /corrections/json")

    def commit_plan(self, user: str = "reviewer"):
        ws = self._ws()
        commit_corrections_inplace(ws.h5_path, plan=ws.plan, user=user, note="ui commit")
        print(f"[well {ws.well_id}] Committed in-place: /raw preserved, /tracking updated")

        # reload well datasets after commit
        ws.base_rows = self._load_best_base(ws.h5_path)
        ws.tracking_rows = self._load_rows_from_dataset(ws.h5_path, "tracking")


    def run(self):
        print("MultiWellTrackReviewer started.")
        print("Click to select track. TAB cycles wells.")
        print("SPACE play/pause | A/D step | J/L jump 50 | [/] jump 500")
        print("R raw/preview | T committed tracking | M merge | S swap | E end op | U undo | K save | P commit | Q quit")

        while True:
            if self.playing:
                self.sel.frame_idx = min(self.sel.frame_idx + self.step_play, self._frame_max)

            frame_gray = self._read_frame(self.sel.frame_idx)
            if frame_gray is None:
                break

            vis = self._render(frame_gray, self.sel.frame_idx)
            cv2.imshow(self.window_name, vis)

            key = cv2.waitKey(20 if self.playing else 0) & 0xFF

            if key in (27, ord("q")):
                break
            elif key == ord(" "):
                self.playing = not self.playing

            elif key == ord("a"):
                self.sel.frame_idx = max(0, self.sel.frame_idx - 1)
            elif key == ord("d"):
                self.sel.frame_idx = min(self._frame_max, self.sel.frame_idx + 1)

            elif key == ord("j"):
                self.sel.frame_idx = max(0, self.sel.frame_idx - 50)
            elif key == ord("l"):
                self.sel.frame_idx = min(self._frame_max, self.sel.frame_idx + 50)

            elif key == ord("["):
                self.sel.frame_idx = max(0, self.sel.frame_idx - 500)
            elif key == ord("]"):
                self.sel.frame_idx = min(self._frame_max, self.sel.frame_idx + 500)

            elif key == 9:  # TAB
                self.sel.well_id = (self.sel.well_id + 1) % self.n_wells
                self.sel.track_id = None

            elif key == ord("r"):
                ws = self._ws()
                self.view_mode[ws.well_id] = "raw" if self.view_mode[ws.well_id] != "raw" else "preview"

            elif key == ord("t"):
                ws = self._ws()
                if ws.tracking_rows.shape[0] > 0:
                    self.view_mode[ws.well_id] = "tracking" if self.view_mode[ws.well_id] != "tracking" else "preview"
                else:
                    print(f"[well {ws.well_id}] No committed /tracking dataset yet.")

            elif key == ord("m"):
                dst = self._prompt_int("Merge selected -> dst_id: ")
                self.merge_selected_to(dst)
            elif key == ord("s"):
                other = self._prompt_int("Swap selected with id: ")
                self.swap_selected_with(other)
            elif key == ord("e"):
                self.end_last_op_here()
            elif key == ord("u"):
                self.undo()

            elif key == ord("k"):
                self.save_plan()
            elif key == ord("p"):
                self.commit_plan(user="reviewer")

        self.close()

    def close(self):
        self._cap.release()
        cv2.destroyWindow(self.window_name)
