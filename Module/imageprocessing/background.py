import cv2
import numpy as np
from Module.dataset.Dataset import load_frame


def snap_background(image, window=11, smooth_iters=3):
    bg = cv2.medianBlur(image, window)

    for _ in range(smooth_iters):
        bg = cv2.GaussianBlur(bg, (201, 201), sigmaX=3)

    return bg


def sample_background_im(image_data, n_frames=300, seed=None):
    if seed is not None:
        np.random.seed(seed)

    total_frames = len(image_data)
    n_frames = min(n_frames, total_frames)
    indicies = np.random.choice(total_frames, size = n_frames, replace = False)

    frames = []
    for idx in indicies:
        frame = load_frame(image_data[idx]).astype(np.float32)
        frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-12) 
        frames.append(frame)

    stack = np.stack(frames, axis=0)
    bg = np.median(stack, axis=0)
    bg = (bg * 255).astype(np.uint8)

    return bg


def sample_background(cap: cv2.VideoCapture, n_frames=300, seed=None):
    if seed is not None:
        np.random.seed(seed)

    if not cap.isOpened():
        raise ValueError("VideoCapture is not opened")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        raise ValueError("Could not determine total frame count")

    n_frames = min(n_frames, total_frames)
    indices = np.random.choice(total_frames, size=n_frames, replace=False)
    original_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frame = frame.astype(np.float32)
        frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-12)
        frames.append(frame)

    cap.set(cv2.CAP_PROP_POS_FRAMES, original_pos)

    if len(frames) == 0:
        raise ValueError("No frames could be read from VideoCapture")

    stack = np.stack(frames, axis=0)
    bg = np.median(stack, axis=0)
    bg = (bg * 255).astype(np.uint8)

    return bg
