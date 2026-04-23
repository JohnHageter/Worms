from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Optional, Union
import json
import time

import h5py
import numpy as np


# ============================================================
# Correction operations
# ============================================================


@dataclass
class MergeOp:
    """
    Replace src_id -> dst_id over a frame interval.
    Used when one track should be absorbed into another.
    """

    src_id: int
    dst_id: int
    start_frame: int = 0
    end_frame: int = 2**31 - 1


@dataclass
class SwapOp:
    """
    Swap two IDs over a frame interval.
    Used when identities cross or flip.
    """

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


# ============================================================
# Core correction logic
# ============================================================


def apply_corrections(rows: np.ndarray, plan: CorrectionPlan) -> np.ndarray:
    """
    Apply a CorrectionPlan to a structured tracking array.

    Required fields:
        - frame_idx
        - track_id
    """
    if "frame_idx" not in rows.dtype.names or "track_id" not in rows.dtype.names:
        raise ValueError("Dataset missing required fields: frame_idx, track_id")

    out = rows.copy()
    frames = out["frame_idx"].astype(int)

    # --- merges ---
    for op in plan.merges:
        mask = (
            (frames >= op.start_frame)
            & (frames <= op.end_frame)
            & (out["track_id"] == op.src_id)
        )
        out["track_id"][mask] = op.dst_id

    # --- swaps ---
    for op in plan.swaps:
        mask = (frames >= op.start_frame) & (frames <= op.end_frame)
        a, b = op.id_a, op.id_b
        mask_a = mask & (out["track_id"] == a)
        mask_b = mask & (out["track_id"] == b)

        out["track_id"][mask_a] = b
        out["track_id"][mask_b] = a

    return out


# ============================================================
# HDF5 helpers
# ============================================================


def _ensure_group(f: h5py.File, name: str) -> h5py.Group:
    if name in f:
        obj = f[name]
        if isinstance(obj, h5py.Group):
            return obj
        raise TypeError(f"{name} exists but is not a Group")
    return f.create_group(name)


def _write_string_dataset(f: h5py.File, path: str, text: str) -> None:
    if path in f:
        del f[path]
    f.create_dataset(path, data=np.array([text], dtype=h5py.string_dtype()))


def _read_string_dataset(f: h5py.File, path: str) -> Optional[str]:
    if path not in f:
        return None
    ds = f[path][()]
    return str(ds[0]) if isinstance(ds, np.ndarray) else str(ds)


def _replace_dataset_atomic(f: h5py.File, name: str, data: np.ndarray) -> None:
    tmp = f"{name}__tmp"
    if tmp in f:
        del f[tmp]
    f.create_dataset(tmp, data=data, maxshape=(None,), chunks=True)
    if name in f:
        del f[name]
    f.move(tmp, name)


# ============================================================
# Public API
# ============================================================


def load_plan_from_h5(h5_path: str) -> CorrectionPlan:
    with h5py.File(h5_path, "r") as f:
        text = _read_string_dataset(f, "corrections/json")
    return (
        CorrectionPlan.from_json(text) if text else CorrectionPlan(merges=[], swaps=[])
    )


def commit_corrections_inplace(
    h5_path: str,
    *,
    plan: Union[CorrectionPlan, str],
    user: str = "unknown",
    note: str = "commit",
) -> None:
    """
    Apply corrections safely and persist them in-place.

    Behavior:
      - Preserve original tracking as /raw (first commit only)
      - Apply corrections to /raw
      - Write corrected /tracking
      - Log plan + history
    """
    plan_obj = (
        plan if isinstance(plan, CorrectionPlan) else CorrectionPlan.from_json(plan)
    )
    plan_json = plan_obj.to_json()

    with h5py.File(h5_path, "r+") as f:
        if "tracking" not in f and "raw" not in f:
            raise KeyError("No /tracking or /raw dataset found")

        if "raw" not in f:
            f.move("tracking", "raw")

        raw = f["raw"][()]
        corrected = apply_corrections(raw, plan_obj)

        _replace_dataset_atomic(f, "tracking", corrected)

        corr_grp = _ensure_group(f, "corrections")
        _write_string_dataset(f, "corrections/json", plan_json)

        _append_history(corr_grp, plan_json, user=user, note=note)


def _append_history(
    corr_grp: h5py.Group, plan_json: str, *, user: str, note: str
) -> None:
    if "history" not in corr_grp:
        corr_grp.create_dataset(
            "history",
            shape=(0,),
            maxshape=(None,),
            dtype=h5py.string_dtype(),
            chunks=True,
        )

    hist = corr_grp["history"]
    entry = {
        "timestamp": time.time(),
        "user": user,
        "note": note,
        "plan": json.loads(plan_json),
    }

    n = hist.shape[0]
    hist.resize((n + 1,))
    hist[n] = json.dumps(entry)
