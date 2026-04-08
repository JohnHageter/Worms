import cv2
import numpy as np
from Module.dataset.video import load_frame
from Module.utils import crop_well
import random


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


def sample_per_well_backgrounds(
    cap,
    wells,
    n_frames=50,
):
    """
    Sample per-well backgrounds from random frames of an opened VideoCapture.
    Returns: dict {well_id: background_image}
    """
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = sorted(
        random.sample(range(total_frames), min(n_frames, total_frames))
    )

    # Accumulate crops per well
    accum = {i: [] for i in range(len(wells))}

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for well_id, well in enumerate(wells):
            crop = crop_well(gray, well)
            accum[well_id].append(crop)

    # Median background per well
    backgrounds = {}
    for well_id, crops in accum.items():
        if len(crops) == 0:
            raise RuntimeError(f"No background frames for well {well_id}")
        backgrounds[well_id] = np.median(np.stack(crops, axis=0), axis=0).astype(
            np.uint8
        )

    return backgrounds


def sample_per_well_backgrounds_masked(
    cap,
    wells,
    n_frames=2000,
    motion_thresh=8,
):
    """
    Sample per-well backgrounds using a masked median so that
    pixels which ever show motion are never absorbed into background.

    Parameters
    ----------
    cap : cv2.VideoCapture
        Open video capture (already opened)
    wells : list of (x, y, r)
        Well definitions
    n_frames : int
        Number of frames to sample from the video
    motion_thresh : int
        Intensity difference threshold for detecting motion

    Returns
    -------
    backgrounds : dict
        {well_id: background_image}
    """

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_indices = sorted(
        random.sample(
            range(total_frames),
            min(n_frames, total_frames),
        )
    )

    # Per-well accumulators
    stacks = {i: [] for i in range(len(wells))}
    exclusion_masks = {}

    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for i, well in enumerate(wells):
            crop = crop_well(gray, well)

            if i not in exclusion_masks:
                exclusion_masks[i] = np.zeros(crop.shape, dtype=bool)

            # Use a conservative running reference
            if stacks[i]:
                reference = np.median(
                    np.stack(stacks[i]),
                    axis=0,
                ).astype(np.uint8)
                diff = cv2.absdiff(crop, reference)
                moving = diff > motion_thresh
                exclusion_masks[i] |= moving

            stacks[i].append(crop)

    # Build masked median backgrounds
    backgrounds = {}

    for i, frames in stacks.items():
        stack = np.stack(frames).astype(float)

        # Exclude foreground-like pixels
        stack[:, exclusion_masks[i]] = np.nan

        bg = np.nanmedian(stack, axis=0)

        # Fill any remaining NaNs safely
        nan_mask = np.isnan(bg)
        if np.any(nan_mask):
            bg[nan_mask] = np.nanmedian(bg)

        backgrounds[i] = bg.astype(np.uint8)

    return backgrounds
