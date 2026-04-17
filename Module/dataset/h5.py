from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import h5py


@dataclass
class TrackRow:
    """
    One row of tracking output for a single track at a single frame.
    Coordinates are expected to be in full-frame coordinates, unless you stored crop-relative.
    """

    track_id: int
    cx: float
    cy: float
    hx: float
    hy: float
    tx: float
    ty: float
    region: int
    conf: float


class H5WellReader:
    """
    Efficient reader for per-well H5 tracking outputs.

    Expects:
      - dataset: 'tracking'
      - optionally dataset: 'metadata'

    Supports 'tracking' stored as:
      A) compound dtype (fields)  
      B) numeric matrix (NxM)     
    """

    def __init__(
        self,
        h5_path: Union[str, Path],
        *,
        video_idx: int,
        columns: Optional[Dict[str, int]] = None,
    ):
        self.h5_path = str(h5_path)
        self.video_idx = int(video_idx)
        self.columns = columns  # only used for numeric tracking arrays

        self._f = h5py.File(self.h5_path, "r")
        if "tracking" not in self._f:
            raise KeyError(f"'tracking' dataset not found in {self.h5_path}")

        self._ds = self._f["tracking"]
        self._rows: np.ndarray = self._load_video_rows(
            self._ds, self.video_idx, self.columns
        )

        self._ptr = 0  # streaming pointer (assumes rows sorted by frame_idx)

    def close(self) -> None:
        self._f.close()

    def metadata(self) -> Optional[str]:
        """
        Returns metadata JSON string if present, else None.
        """
        if "metadata" not in self._f:
            return None
        md = self._f["metadata"][()] # type: ignore
        if isinstance(md, np.ndarray):
            return str(md[0])
        return str(md)

    def rows_for_frame(self, frame_idx: int) -> List[TrackRow]:
        """
        Return all rows for the given frame_idx, advancing internal pointer.
        """
        out: List[TrackRow] = []
        f = int(frame_idx)

        while (
            self._ptr < len(self._rows) and self._get_frame(self._rows[self._ptr]) < f
        ):
            self._ptr += 1

        while (
            self._ptr < len(self._rows) and self._get_frame(self._rows[self._ptr]) == f
        ):
            out.append(self._parse_row(self._rows[self._ptr]))
            self._ptr += 1

        return out



    @staticmethod
    def _load_video_rows(
        ds, video_idx: int, columns: Optional[Dict[str, int]]
    ) -> np.ndarray:
        arr = ds[()]
        if arr.shape[0] == 0:
            return arr

        if arr.dtype.names is not None:
            required = {
                "video_idx",
                "frame_idx",
                "track_id",
                "cx",
                "cy",
                "hx",
                "hy",
                "tx",
                "ty",
                "region",
            }
            missing = required.difference(arr.dtype.names)
            if missing:
                raise ValueError(
                    f"Compound tracking dtype missing fields {missing} in dataset {ds.name}"
                )

            arr = arr[arr["video_idx"] == int(video_idx)]
            if arr.shape[0] == 0:
                return arr
            arr = np.sort(arr, order="frame_idx")
            return arr

        if columns is None:
            raise ValueError(
                "Numeric 'tracking' dataset detected, but no columns mapping provided.\n"
                "Provide columns like:\n"
                "columns={'video_idx':0,'frame_idx':1,'track_id':2,'cx':3,'cy':4,'hx':5,'hy':6,'tx':7,'ty':8,'region':9,'ht_conf':10}"
            )

        vid_col = columns["video_idx"]
        frm_col = columns["frame_idx"]

        arr = arr[arr[:, vid_col].astype(int) == int(video_idx)]
        if arr.shape[0] == 0:
            return arr

        arr = arr[np.argsort(arr[:, frm_col].astype(int))]
        return arr

    def _get_frame(self, row) -> int:
        if row.dtype.names is not None:
            return int(row["frame_idx"])
        return int(row[self.columns["frame_idx"]])  # type: ignore[index]

    def _parse_row(self, row) -> TrackRow:
        if row.dtype.names is not None:
            conf = float(row["ht_conf"]) if "ht_conf" in row.dtype.names else 0.0
            return TrackRow(
                track_id=int(row["track_id"]),
                cx=float(row["cx"]),
                cy=float(row["cy"]),
                hx=float(row["hx"]),
                hy=float(row["hy"]),
                tx=float(row["tx"]),
                ty=float(row["ty"]),
                region=int(row["region"]),
                conf=conf,
            )


        c = self.columns  # type: ignore[assignment]
        conf = float(row[c["ht_conf"]]) if "ht_conf" in c else 0.0  # type: ignore[operator]

        return TrackRow(
            track_id=int(row[c["track_id"]]),  # type: ignore
            cx=float(row[c["cx"]]),  # type: ignore
            cy=float(row[c["cy"]]),  # type: ignore
            hx=float(row[c["hx"]]),  # type: ignore
            hy=float(row[c["hy"]]),  # type: ignore
            tx=float(row[c["tx"]]),  # type: ignore
            ty=float(row[c["ty"]]),  # type: ignore
            region=int(row[c["region"]]),  # type: ignore
            conf=conf,
        )
