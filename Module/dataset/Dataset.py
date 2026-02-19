from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, TypedDict, cast

import h5py


H5Mode = Literal["r", "r+", "w", "w-", "x", "a"]


class ExperimentData(TypedDict):
    acquisition_parameters: Dict[str, object]
    tracking_parameters: Dict[str, object]
    video_paths: List[Path]
    csv_paths: List[Path]


@dataclass(frozen=True)
class ExperimentPaths:
    root: Path
    metadata_file: Path
    videos_dir: Path
    csv_dir: Path


class ExperimentDataset:

    def __init__(self, root_dir: str | Path, mode: H5Mode = "a") -> None:
        self._paths = self._build_paths(Path(root_dir))

        if mode in ("w", "w-", "x"):
            self._create_structure()

        self._h5: h5py.File = h5py.File(self._paths.metadata_file, mode)

        if mode in ("w", "w-", "x"):
            self._initialize_file()

        self._acquisition_ds: h5py.Dataset = cast(
            h5py.Dataset, self._h5["acquisition_parameters"]
        )
        self._tracking_ds: h5py.Dataset = cast(
            h5py.Dataset, self._h5["tracking_parameters"]
        )
        self._videos_ds: h5py.Dataset = cast(h5py.Dataset, self._h5["videos"])
        self._csv_ds: h5py.Dataset = cast(h5py.Dataset, self._h5["csv_files"])


    @staticmethod
    def _build_paths(root: Path) -> ExperimentPaths:
        return ExperimentPaths(
            root=root,
            metadata_file=root / "metadata.h5",
            videos_dir=root / "videos",
            csv_dir=root / "csv",
        )

    def _create_structure(self) -> None:
        self._paths.root.mkdir(parents=True, exist_ok=True)
        self._paths.videos_dir.mkdir(exist_ok=True)
        self._paths.csv_dir.mkdir(exist_ok=True)


    def _initialize_file(self) -> None:
        string_dt = h5py.string_dtype(encoding="utf-8")

        self._h5.attrs["created"] = datetime.utcnow().isoformat()
        self._h5.attrs["version"] = "1.0"

        self._h5.create_dataset(
            "acquisition_parameters",
            data=json.dumps({}),
            dtype=string_dt,
        )

        self._h5.create_dataset(
            "tracking_parameters",
            data=json.dumps({}),
            dtype=string_dt,
        )

        self._h5.create_dataset(
            "videos",
            shape=(0,),
            maxshape=(None,),
            dtype=string_dt,
        )

        self._h5.create_dataset(
            "csv_files",
            shape=(0,),
            maxshape=(None,),
            dtype=string_dt,
        )

        self._h5.flush()


    def set_acquisition_parameters(self, parameters: Dict[str, object]) -> None:
        self._acquisition_ds[...] = json.dumps(parameters)
        self._h5.flush()

    def set_tracking_parameters(self, parameters: Dict[str, object]) -> None:
        self._tracking_ds[...] = json.dumps(parameters)
        self._h5.flush()


    def add_video(self, video_path: str | Path) -> Path:
        src = Path(video_path)
        if not src.exists():
            raise FileNotFoundError(src)

        dest = self._paths.videos_dir / src.name
        shutil.copy2(src, dest)

        self._append_string(self._videos_ds, dest.name)
        return dest


    def add_csv(self, csv_path: str | Path) -> Path:
        src = Path(csv_path)
        if not src.exists():
            raise FileNotFoundError(src)

        dest = self._paths.csv_dir / src.name
        shutil.copy2(src, dest)

        self._append_string(self._csv_ds, dest.name)
        return dest


    def load(self) -> ExperimentData:
        acquisition_raw = self._acquisition_ds[()]
        tracking_raw = self._tracking_ds[()]

        acquisition = json.loads(acquisition_raw.decode())
        tracking = json.loads(tracking_raw.decode())

        video_names = self._read_string_array(self._videos_ds)
        csv_names = self._read_string_array(self._csv_ds)

        video_paths = [self._paths.videos_dir / name for name in video_names]
        csv_paths = [self._paths.csv_dir / name for name in csv_names]

        return {
            "acquisition_parameters": acquisition,
            "tracking_parameters": tracking,
            "video_paths": video_paths,
            "csv_paths": csv_paths,
        }


    def _append_string(self, dataset: h5py.Dataset, value: str) -> None:
        current_size = dataset.shape[0]
        dataset.resize((current_size + 1,))
        dataset[current_size] = value
        self._h5.flush()

    @staticmethod
    def _read_string_array(dataset: h5py.Dataset) -> List[str]:
        return [item.decode() for item in dataset[()]]


    def close(self) -> None:
        self._h5.close()

    def __enter__(self) -> "ExperimentDataset":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
