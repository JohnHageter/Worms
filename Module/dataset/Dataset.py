from abc import ABC
import datetime
from pathlib import Path
from typing import Optional, Any

import h5py


class Dataset(ABC):
    def __init__(self, root: Path = Path(".")):
        self.root = root
        self.metadata = self.root / "metadata.h5"
        self.capture = self.root / "acquisition"
        self.tracking = self.root / "tracks"

        self._h5: Optional[h5py.File] = None

    def create(self, overwrite: bool = False) -> None:
        if self.metadata.exists() and not overwrite:
            raise FileExistsError(f"{self.metadata} already exists.")

        self.root.mkdir(parents=True, exist_ok=True)
        self.capture.mkdir(exist_ok=True)
        self.tracking.mkdir(exist_ok=True)

        self._h5 = h5py.File(self.metadata, "w")

        self._h5.create_group("camera")
        self._h5.create_group("acquisition")

        timestamp = datetime.datetime.now().isoformat(timespec="seconds")
        self._h5.attrs["created"] = timestamp

        self._h5.flush()

    def write_attr(self, key: str, value: Any, group: Optional[str] = None):
        if self._h5 is None:
            raise RuntimeError("HDF5 file not created.")

        target = self._h5 if group is None else self._h5.require_group(group)
        target.attrs[key] = value
        self._h5.flush()


if __name__ == "__main__":
    d = Dataset(root=Path("dataout"))

    print(d.metadata.absolute())
    print(d.capture.absolute())

    d.create()

    timestamp = d._h5.attrs["created"]
