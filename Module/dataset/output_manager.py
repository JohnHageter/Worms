import cv2
import h5py

from Module.utils import append_row

class OutputManager:
    def __init__(self, out_dir, wells, fps):
        self.out_dir = out_dir
        self.fps = fps
        self.wells = wells
        self.h5_files = {}
        self.video = None
        self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        self._init_h5()

    def _init_h5(self):
        for well_id in range(len(self.wells)):
            h5 = h5py.File(self.out_dir / f"well_{well_id}.h5", "w")

            h5.create_dataset(
                "main", (0, 13), maxshape=(None, 13), dtype="i4", chunks=(1024, 13)
            )
            h5.create_dataset(
                "fission", (0, 9), maxshape=(None, 9), dtype="i4", chunks=(1024, 9)
            )

            self.h5_files[well_id] = h5

    def write_main(self, well_id, row):
        append_row(self.h5_files[well_id]["main"], row)

    def write_fission(self, well_id, row):
        append_row(self.h5_files[well_id]["fission"], row)

    def write_video(self, stitched_frame):
        if self.video is None:
            h, w, _ = stitched_frame.shape
            self.video = cv2.VideoWriter(
                str(self.out_dir / "Annotated.mp4"),
                self.fourcc,
                self.fps,
                (w, h),
            )
        self.video.write(stitched_frame)

    def close(self):
        for h5 in self.h5_files.values():
            h5.close()
        if self.video is not None:
            self.video.release()
