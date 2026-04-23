from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from pathlib import Path
import cv2

from Module.dataset.video import open_dataset
from Module.dataset.h5 import H5WellReader
from Module.dataset.Annotater import MultiWellAnnotator, H5TrackProvider
from Module.utils import (
    crop_well,
    stitch_wells,
    annotate_frame_metadata,
)


def annotate_all_videos(
    *,
    video_dir: Path,
    h5_dir: Path,
    wells,
    output_path: Path,
    every: int = 1,
    capture_fps: int = 2,
):
    """
    Annotate all videos in a directory into one continuous annotated video.

    Parameters
    ----------
    video_dir : Path
        Directory containing original videos (mp4).
    h5_dir : Path
        Directory containing per-well HDF5 files (well_0.h5, ...).
    wells : list
        List of ROIs exactly as used during tracking.
    output_path : Path
        Path to output annotated video.
    every : int
        Annotate every Nth frame (e.g. 50, 100).
    capture_fps : int
        FPS used during tracking (for annotation timing).
    """

    video_paths = sorted(video_dir.glob("*.mp4"))
    if not video_paths:
        raise RuntimeError(f"No videos found in {video_dir}")


    h5_paths = sorted(h5_dir.glob("well_*.h5"))
    readers = [
        H5WellReader(h5, video_idx=0) for h5 in h5_paths  
        
    track_provider = H5TrackProvider(readers)

    annotator = MultiWellAnnotator(
        wells=wells,
        crop_well_fn=crop_well,
        stitch_wells_fn=stitch_wells,
        annotate_frame_fn=lambda frame, fi, vi, n, fps: annotate_frame_metadata(
            frame,
            frame_idx=fi,
            video_idx=vi,
            num_worms=n,
            fps=fps,
        ),
        capture_fps=capture_fps,
        draw_ids=True,
        draw_head_tail=True,
        coords_are_full_frame=False,
    )

    writer = None

    try:
        for video_idx, video_path in enumerate(video_paths):
            print(f"Annotating video {video_idx}: {video_path.name}")

            for r in readers:
                r.video_idx = video_idx
                r._rows = r._load_video_rows(r._ds, video_idx, r.columns)
                r._ptr = 0

            cap = open_dataset(video_path)
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % every != 0:
                    frame_idx += 1
                    continue

                if frame.ndim == 3:
                    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    frame_gray = frame

                tracks_by_well = track_provider.tracks_by_well(frame_idx)

                annotated = annotator.annotate_frame_multiwell(
                    frame_gray,
                    tracks_by_well,
                    video_idx=video_idx,
                    frame_idx=frame_idx,
                )

                if writer is None:
                    h, w = annotated.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(
                        str(output_path),
                        fourcc,
                        capture_fps,
                        (w, h),
                    )

                writer.write(annotated)
                frame_idx += 1

            cap.release()

    finally:
        for r in readers:
            r.close()
        if writer is not None:
            writer.release()

    print(f"Annotated video written to: {output_path}")


if __name__ == "__main__":
    from pathlib import Path
    import numpy as np

    WELLS = np.load("AgarWell.npy")

    annotate_all_videos(
        video_dir=Path("D:/Tracking_Data/Sachi/T4_4233A_Rig_AgaroseWell/first"),
        h5_dir=Path("D:/Tracking_Data/Sachi/T4_4233A_Rig_AgaroseWell/out_v3/"),
        wells=WELLS,  
        output_path=Path("D:/Tracking_Data/Sachi/T4_4233A_Rig_AgaroseWell/out_v3/post_annotation.mp4"),
        every=100, 
        capture_fps=2,
    )
