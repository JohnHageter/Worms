import cv2
from datetime import datetime
import re
from pathlib import Path

import numpy as np

def play_video(video, framerate, num_frames=10, shrink_factor=0.5):
    if isinstance(video, cv2.VideoCapture):
        for _ in range(num_frames):
            ret, frame = video.read()
            if not ret:
                break
            frame = cv2.resize(frame, (0,0), fx=shrink_factor, fy=shrink_factor)
            cv2.imshow("Video", frame)
            if cv2.waitKey(int(1000 / framerate)) & 0xFF == ord('q'):
                break
        video.release()
    else:
        for i in range(min(num_frames, video.shape[0])):
            frame = video[i]
            frame = cv2.resize(frame, (0,0), fx=shrink_factor, fy=shrink_factor)
            cv2.imshow("Video", frame)
            if cv2.waitKey(int(1000 / framerate)) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()

def plot_well_masks(image, wells, parameters=None):
    img_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    for  idx, (x,y,r) in enumerate(wells):
        cv2.circle(img_color, (int(x), int(y)), int(r), (255, 0, 0), 2)
        cv2.putText(img_color, f'well {idx+1}', 
                    (int(x)-int(r/2), int(y)-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 0, 255), 2)
    
    if parameters is not None:
        text_y = img_color.shape[0] - 20 * (len(parameters) + 1)
        for key, value in parameters.items():
            cv2.putText(img_color, f'{key}: {value}', 
                        (10, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (0, 255, 0), 2)
            text_y += 30

    img_color = cv2.resize(img_color, (600,512), interpolation=cv2.INTER_LINEAR)
    cv2.imshow("Detected Wells", img_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def read_timestamp(path):
    """
    Extract timestamp from filename like:
    image_20251222_121900.png
    """
    m = re.search(r'(\d{8})_(\d{6})', path)
    if not m:
        raise ValueError(f"No timestamp found in {path}")

    date_part, time_part = m.groups()
    return datetime.strptime(date_part + time_part, "%Y%m%d%H%M%S")

def find_endpoints(skel):
    kernel = np.array([[1,1,1],
                       [1,0,1],
                       [1,1,1]], np.uint8)

    neighbors = cv2.filter2D(skel, -1, kernel)
    ys, xs = np.where((skel > 0) & (neighbors == 255))
    return list(zip(xs, ys))

def trace_path(skel, start):
    h, w = skel.shape
    path = [start]
    visited = set([start])
    cur = start

    while True:
        x, y = cur
        next_pixel = None

        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    if skel[ny, nx] and (nx, ny) not in visited:
                        next_pixel = (nx, ny)
                        break
            if next_pixel is not None:
                break

        if next_pixel is None:
            break

        visited.add(next_pixel)
        path.append(next_pixel)
        cur = next_pixel

    return path

def create_writer(
    file: str, width: int, height: int, frame_rate: float = 10.0, is_color: bool = True
) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        file, fourcc, frame_rate, (width, height), isColor=is_color
    )

    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {file}")

    return writer
