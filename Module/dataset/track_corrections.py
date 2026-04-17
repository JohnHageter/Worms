from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Optional, Union
import json
import time

import h5py
import numpy as np


@dataclass
class MergeOp:
    src_id: int
    dst_id: int
    start_frame: int = 0
    end_frame: int = 2**31 - 1  

@dataclass
class SwapOp:
    id_a: int
    id_b: int
    start_frame: int
    end_frame: int = 2**31 - 1  


@dataclass
class CorrectionPlan:
    merges: List[MergeOp]
    swaps: List[SwapOp]

    def to_json(self) -> str:
        return json.dumps(
            {
                "merges": [asdict(m) for m in self.merges],
                "swaps": [asdict(s) for s in self.swaps],
            },
            indent=2,
        )

    @staticmethod
    def from_json(text: str) -> "CorrectionPlan":
        d = json.loads(text)
        return CorrectionPlan(
            merges=[MergeOp(**m) for m in d.get("merges", [])],
            swaps=[SwapOp(**s) for s in d.get("swaps", [])],
        )


def apply_corrections(rows: np.ndarray, plan: CorrectionPlan) -> np.ndarray:
    """
    rows: structured array with at least fields ['frame_idx', 'track_id'] (and others).
    Applies merges first, then swaps, in the order recorded.
    Returns a new structured array (copy).
    """
    out = rows.copy()
    f = out["frame_idx"].astype(int)

    # merges: replace src_id -> dst_id within frame interval
    for op in plan.merges:
        mask = (
            (f >= op.start_frame) & (f <= op.end_frame) & (out["track_id"] == op.src_id)
        )
        out["track_id"][mask] = op.dst_id

    # swaps: swap id_a <-> id_b within interval
    for op in plan.swaps:
        mask = (f >= op.start_frame) & (f <= op.end_frame)
        a = op.id_a
        b = op.id_b
        mask_a = mask & (out["track_id"] == a)
        mask_b = mask & (out["track_id"] == b)
        out["track_id"][mask_a] = b
        out["track_id"][mask_b] = a

    return out


def _ensure_group(f: h5py.File, name: str) -> h5py.Group:
    if name in f:
        obj = f[name]
        if isinstance(obj, h5py.Group):
            return obj
        raise TypeError(f"{name} exists but is not a Group")
    return f.create_group(name)


def _write_string_dataset(f: h5py.File, path: str, text: str) -> None:
    """
    Create or overwrite a string dataset at path with a single string element.
    """
    if path in f:
        del f[path]
    f.create_dataset(path, data=np.array([text], dtype=h5py.string_dtype()))


def _read_string_dataset(f: h5py.File, path: str) -> Optional[str]:
    if path not in f:
        return None
    ds = f[path][()]  # type: ignore
    if isinstance(ds, np.ndarray):
        return str(ds[0])
    return str(ds)


def _replace_dataset_atomic(f: h5py.File, name: str, data: np.ndarray) -> None:
    """
    Overwrite dataset `name` with `data` using a temp dataset to avoid partial writes.
    """
    tmp = f"{name}__tmp"
    if tmp in f:
        del f[tmp]
    f.create_dataset(tmp, data=data, maxshape=(None,), chunks=True)
    if name in f:
        del f[name]
    f.move(tmp, name)


def load_plan_from_h5(h5_path: str) -> CorrectionPlan:
    """
    Load the stored correction plan from /corrections/json if present.
    If not present, returns empty plan.
    """
    with h5py.File(h5_path, "r") as f:
        text = _read_string_dataset(f, "corrections/json")
    if not text:
        return CorrectionPlan(merges=[], swaps=[])
    return CorrectionPlan.from_json(text)


def save_plan_to_h5(h5_path: str, plan: CorrectionPlan) -> None:
    """
    Save plan into /corrections/json (overwrites existing).
    Does NOT modify /tracking or /raw.
    """
    plan_json = plan.to_json()
    with h5py.File(h5_path, "r+") as f:
        _ensure_group(f, "corrections")
        _write_string_dataset(f, "corrections/json", plan_json)


def rebuild_tracking_from_raw(h5_path: str) -> None:
    """
    Rebuild /tracking from /raw plus stored /corrections/json.
    Requires /raw to exist. Does not change /raw.
    """
    with h5py.File(h5_path, "r+") as f:
        if "raw" not in f:
            raise KeyError("Cannot rebuild: /raw does not exist.")
        raw = f["raw"][()]  # type: ignore

        plan_text = _read_string_dataset(f, "corrections/json")
        plan = (
            CorrectionPlan.from_json(plan_text) if plan_text else CorrectionPlan([], [])
        )

        corrected = apply_corrections(raw, plan)  # type: ignore
        _replace_dataset_atomic(f, "tracking", corrected)


def commit_corrections_inplace(
    h5_path: str,
    *,
    plan: Union[CorrectionPlan, str],
    user: str = "unknown",
    note: str = "commit",
) -> None:
    """
    Implements the requested behavior:

      - If /raw does not exist, move current /tracking -> /raw
      - Apply correction plan to /raw
      - Write corrected to /tracking (authoritative)
      - Store plan in /corrections/json
      - Append log entry to /corrections/history

    No new H5 file is created.
    """

    plan_obj = (
        plan if isinstance(plan, CorrectionPlan) else CorrectionPlan.from_json(plan)
    )
    plan_json = plan_obj.to_json()

    with h5py.File(h5_path, "r+") as f:
        if "tracking" not in f and "raw" not in f:
            raise KeyError(
                "Neither '/tracking' nor '/raw' exists in H5. Nothing to correct."
            )

        # 1) Ensure /raw exists (preserve original)
        if "raw" not in f:
            if "tracking" not in f:
                raise KeyError(
                    "'/raw' missing and '/tracking' missing — cannot initialize raw."
                )
            f.move("tracking", "raw")

        raw = f["raw"][()]  # type: ignore
        if raw.shape[0] == 0:  # type: ignore
            _ensure_group(f, "corrections")
            _write_string_dataset(f, "corrections/json", plan_json)
            _append_history(f, plan_json, user=user, note=note)
            return

        # 2) Apply corrections to raw
        corrected = apply_corrections(raw, plan_obj)  # type: ignore

        # 3) Overwrite /tracking with corrected
        _replace_dataset_atomic(f, "tracking", corrected)

        # 4) Save plan JSON in-file
        _ensure_group(f, "corrections")
        _write_string_dataset(f, "corrections/json", plan_json)

        # 5) Append history log
        _append_history(f, plan_json, user=user, note=note)


def _append_history(f: h5py.File, plan_json: str, *, user: str, note: str) -> None:
    """
    Append one JSON entry to /corrections/history.
    """
    corr_grp = _ensure_group(f, "corrections")

    if "history" not in corr_grp:
        dt = h5py.string_dtype()
        corr_grp.create_dataset(
            "history", shape=(0,), maxshape=(None,), dtype=dt, chunks=True
        )

    hist = corr_grp["history"]
    entry = {
        "timestamp": time.time(),
        "user": user,
        "note": note,
        "plan": json.loads(plan_json),
    }

    n = hist.shape[0]  # type: ignore
    hist.resize((n + 1,)) # type: ignore
    hist[n] = json.dumps(entry) # type: ignore
