from pathlib import Path
import cv2
import numpy as np
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


from Module.dataset.Drawer import ROIDrawer
from Module.utils import (
    crop_well,
    stitch_wells,
    annotate_frame_metadata,
    annotate_region_boundaries,
)
from Module.dataset.h5 import H5WellReader
from Module.dataset.Annotater import MultiWellAnnotator, H5TrackProvider


def annotate_folder_from_h5(
    video_dir: Path,
    h5_dir: Path,
    wells_npy: Path,
    out_path: Path,
    *,
    single_output_mp4: bool = True,
    annotate_every_n_frames: int = 1,  # e.g. 50 or 100
    write_fps: int = 30,
    show_preview: bool = False,
    quit_key: str = "q",
):
    video_paths = sorted(video_dir.glob("*.mp4"))
    if not video_paths:
        raise RuntimeError(f"No videos found in: {video_dir}")

    wells = ROIDrawer.load(str(wells_npy))

    annotator = MultiWellAnnotator(
        wells=wells,
        crop_well_fn=crop_well,
        stitch_wells_fn=stitch_wells,
        annotate_frame_fn=annotate_frame_metadata,
        annotate_region_boundaries_fn=annotate_region_boundaries,
        wall_width_draw=8,
        region_boundary_thickness=1,
        capture_fps=2,  # used only for overlay text
        draw_ids=True,
        draw_head_tail=True,
        coords_are_full_frame=False,  # IMPORTANT: your H5 coords are crop-relative
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video = None
    out_size = None

    def ensure_writer(frame_bgr):
        nonlocal out_video, out_size
        h, w = frame_bgr.shape[:2]
        if out_video is None:
            out_size = (w, h)
            out_video = cv2.VideoWriter(str(out_path), fourcc, write_fps, out_size)
        else:
            if (w, h) != out_size:
                raise RuntimeError(
                    f"Output resolution changed from {out_size} to {(w, h)}. "
                    f"All stitched frames must have the same size for a single MP4."
                )

    try:
        if not single_output_mp4:
            # If not single output, we'll create one output per input video.
            # We'll override out_path per video.
            pass

        for video_idx, video_path in enumerate(video_paths):
            print(f"\nAnnotating video {video_idx}: {video_path.name}")

            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video: {video_path}")

            # Build readers for this video_idx (one per well)
            readers = [
                H5WellReader(h5_dir / f"well_{well_id}.h5", video_idx=video_idx)
                for well_id in range(len(wells))
            ]
            provider = H5TrackProvider(readers)

            # Per-video output mode
            if not single_output_mp4:
                # close previous writer if any
                if out_video is not None:
                    out_video.release()
                    out_video = None
                    out_size = None

                per_video_out = (
                    out_path.parent
                    / f"{out_path.stem}_{video_path.stem}{out_path.suffix}"
                )
                print(f"Writing: {per_video_out}")
                current_out_path = per_video_out
            else:
                current_out_path = out_path

            frame_idx = 0
            written = 0

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                # Only annotate every N frames
                if (
                    annotate_every_n_frames > 1
                    and (frame_idx % annotate_every_n_frames) != 0
                ):
                    frame_idx += 1
                    continue

                if frame.ndim == 3:
                    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    frame_gray = frame

                tracks_by_well = provider.tracks_by_well(frame_idx)

                stitched = annotator.annotate_frame_multiwell(
                    frame_gray=frame_gray,
                    tracks_by_well=tracks_by_well,
                    video_idx=video_idx,
                    frame_idx=frame_idx,
                )

                # Create writer (single output) or per video
                if not single_output_mp4:
                    # Ensure per-video writer is created with correct path
                    if out_video is None:
                        h, w = stitched.shape[:2]
                        out_size = (w, h)
                        out_video = cv2.VideoWriter(
                            str(current_out_path), fourcc, write_fps, out_size
                        )
                    else:
                        if (
                            stitched.shape[1] != out_size[0]
                            or stitched.shape[0] != out_size[1]
                        ):
                            raise RuntimeError(
                                "Per-video output resolution changed unexpectedly."
                            )
                else:
                    ensure_writer(stitched)

                out_video.write(stitched)
                written += 1

                if show_preview:
                    cv2.imshow("Annotated", stitched)
                    if cv2.waitKey(1) & 0xFF == ord(quit_key):
                        raise KeyboardInterrupt

                frame_idx += 1

            cap.release()
            for r in readers:
                r.close()

            print(f"  wrote {written} frames from {video_path.name}")

        print(f"\nDONE. Output written to: {out_path}")

    finally:
        if out_video is not None:
            out_video.release()
        if show_preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    VIDEO_DIR = Path(r"D:/Tracking_Data/Sachi/T3/week_1/")
    H5_DIR = Path(r"D:/Tracking_Data/Sachi/T3/week_1_out_v3/")
    WELLS_NPY = Path("wells.npy")
    OUT_PATH = H5_DIR / "Annotated_ALL_VIDEOS.mp4"

    annotate_folder_from_h5(
        video_dir=VIDEO_DIR,
        h5_dir=H5_DIR,
        wells_npy=WELLS_NPY,
        out_path=OUT_PATH,
        single_output_mp4=True,  
        annotate_every_n_frames=50,  
        write_fps=30,
        show_preview=True,
    )
