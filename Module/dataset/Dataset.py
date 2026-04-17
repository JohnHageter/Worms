from __future__ import annotations
import h5py
import numpy as np
from typing import Optional
from h5py import Dataset
from Module.Tracking.Configuration import TrackingParameters

TRACK_DTYPE = np.dtype(
    [
        ("video_idx", "i4"),
        ("frame_idx", "i4"),
        ("track_id", "i4"),
        ("cx", "f4"),
        ("cy", "f4"),
        ("hx", "f4"),
        ("hy", "f4"),
        ("tx", "f4"),
        ("ty", "f4"),
        ("region", "i2"),
        ("ht_conf", "f4"),
    ]
)


class TrackingDataset:
    def __init__(self, path: str, params: TrackingParameters):
        self.h5 = h5py.File(path, "w")

        # metadata: single JSON blob
        meta = params.to_json()
        self.h5.create_dataset(
            "metadata", data=np.array([meta], dtype=h5py.string_dtype())
        )

        # tracking: append-only compound dtype
        self.h5.create_dataset(
            "tracking", shape=(0,), maxshape=(None,), dtype=TRACK_DTYPE, chunks=(8192,)
        )

    def append_track_row(
        self,
        video_idx: int,
        frame_idx: int,
        track_id: int,
        centroid: np.ndarray,
        region: int,
        head: Optional[np.ndarray],
        tail: Optional[np.ndarray],
        ht_conf: float = 0.0,
    ):
        tr = self.h5["tracking"]
        assert isinstance(tr, Dataset)

        hx = float(head[0]) if head is not None else -1.0
        hy = float(head[1]) if head is not None else -1.0
        tx = float(tail[0]) if tail is not None else -1.0
        ty = float(tail[1]) if tail is not None else -1.0

        row = np.array(
            (
                int(video_idx),
                int(frame_idx),
                int(track_id),
                float(centroid[0]),
                float(centroid[1]),
                hx,
                hy,
                tx,
                ty,
                int(region),
                float(ht_conf),
            ),
            dtype=TRACK_DTYPE,
        )

        n = tr.shape[0]
        tr.resize((n + 1,))
        tr[n] = row

    def close(self):
        self.h5.close()
