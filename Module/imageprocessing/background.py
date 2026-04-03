import cv2
import numpy as np
from Module.dataset.video import load_frame


def snap_background(image, window=11, smooth_iters=3):
    bg = cv2.medianBlur(image, window)

    for _ in range(smooth_iters):
        bg = cv2.GaussianBlur(bg, (201, 201), sigmaX=3)

    return bg


def sample_background_from_images(image_data, n_frames=300, seed=None) -> np.ndarray:
    """
    Sample background from a list of images
    """

    if seed is not None:
        np.random.seed(seed)

    total_frames = len(image_data)
    n_frames = min(n_frames, total_frames)
    indicies = np.random.choice(total_frames, size=n_frames, replace=False)

    frames = []
    for idx in indicies:
        frame = load_frame(image_data[idx]).astype(np.float32)
        frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-12)
        frames.append(frame)

    stack = np.stack(frames, axis=0)
    bg = np.median(stack, axis=0)
    bg = (bg * 255).astype(np.uint8)

    return bg


def sample_background(video, n_frames=300, seed=None) -> np.ndarray:
    """
    Sample background from a cv2.VideoCapture object
    """

    if seed is not None:
        np.random.seed(seed)

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    n_frames = min(n_frames, total_frames)

    indices = np.random.choice(total_frames, size=n_frames, replace=False)

    frames = []

    for idx in indices:
        video.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = video.read()

        if not ret:
            continue

        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frame = frame.astype(np.float32)

        frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-12)

        frames.append(frame)

    if len(frames) == 0:
        raise RuntimeError("No frames could be read for background sampling.")

    stack = np.stack(frames, axis=0)
    bg = np.median(stack, axis=0)

    bg = (bg * 255).astype(np.uint8)

    video.set(cv2.CAP_PROP_POS_FRAMES, 0)

    return bg
