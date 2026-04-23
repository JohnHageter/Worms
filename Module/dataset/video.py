import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from Module.utils import create_writer


def count_total_frames(video_paths, frame_step=0):
    total = 0
    for vp in video_paths:
        v = open_dataset(str(vp))
        total += int(v.get(cv2.CAP_PROP_FRAME_COUNT))
        v.release()
    return total // frame_step


def open_dataset(video_path) -> cv2.VideoCapture:
    if not Path(video_path).exists():
        raise ValueError(f"Video file {video_path} does not exist.")

    cap = cv2.VideoCapture(str(video_path), )
    if not cap.isOpened():
        raise ValueError(f"Could not open video file {video_path}.")

    return cap

def generate_dataset_from_timelapse(
    folder_path: str | Path,
    output_file_name: str = "dataset.mp4",
    frame_rate: float = 10.0,
):
    folder_path = Path(folder_path)

    if not folder_path.exists():
        raise IOError("Dataset folder does not exist.")
    if not output_file_name.endswith(".mp4"):
        output_file_name += ".mp4"

    image_paths = sorted(
        p
        for p in folder_path.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    )

    if not image_paths:
        raise IOError("No images found in dataset folder.")

    first = cv2.imread(str(image_paths[0]))
    if first is None:
        raise IOError(f"Failed to read {image_paths[0]}")

    h, w = first.shape[:2]

    writer = create_writer(
        file=str(folder_path / output_file_name),
        width=w,
        height=h,
        frame_rate=frame_rate,
    )

    try:
        for path in tqdm(image_paths, desc="Writing video", unit="frame"):
            image = cv2.imread(str(path))
            if image is None:
                raise IOError(f"Failed to read {path}")

            if image.shape[:2] != (h, w):
                raise ValueError(f"Image size mismatch: {path}")

            writer.write(image)

    finally:
        writer.release()



def expand_well_radius(wells, scale=1.2):
    expanded = []
    for x, y, r in wells:
        r_new = r * scale
        expanded.append((x, y, r_new))
    return expanded
