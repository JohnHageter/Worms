import cv2
import numpy as np
from Module.utils import crop_well
import random


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
    frame_step=5,  # NEW: compare frames 5 apart
):
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Take every `frame_step`-th frame to spread out samples
    sample_indices = list(range(0, total_frames, max(1, total_frames // n_frames)))[
        :n_frames
    ]

    stacks = {i: [] for i in range(len(wells))}
    exclusion_masks = {i: None for i in range(len(wells))}
    prev_crops = {i: None for i in range(len(wells))}

    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for i, well in enumerate(wells):
            crop = crop_well(gray, well).astype(np.uint8)

            # Initialize exclusion mask
            if exclusion_masks[i] is None:
                exclusion_masks[i] = np.zeros(crop.shape, dtype=bool)

            # Compare with previous crop (if exists, and spaced by frame_step)
            if prev_crops[i] is not None:
                diff = cv2.absdiff(crop, prev_crops[i])
                moving = diff > motion_thresh
                exclusion_masks[i] |= moving

            prev_crops[i] = crop
            stacks[i].append(crop)

    # Build backgrounds using masked median
    backgrounds = {}
    for i, frames in stacks.items():
        stack = np.stack(frames).astype(np.float32)
        # Apply exclusion mask: set moving pixels to NaN
        mask = exclusion_masks[i]
        stack[:, mask] = np.nan
        bg = np.nanmedian(stack, axis=0)
        # Fill remaining NaNs with global median
        if np.any(np.isnan(bg)):
            global_med = np.nanmedian(bg)
            bg[np.isnan(bg)] = global_med
        backgrounds[i] = bg.astype(np.uint8)

    return backgrounds


class RollingMedianBackground:
    def __init__(self, history=300, update_every=10, decay=0.95):
        """
        history: number of past frames to keep for median
        update_every: recompute median every N frames (for speed)
        decay: weight for old frames (exponential forgetting)
        """
        self.history = history
        self.update_every = update_every
        self.decay = decay
        self.frame_buffer = []
        self.frame_count = 0
        self.background = None

    def update(self, frame_gray):
        """Call for every frame. Returns current background image."""
        self.frame_buffer.append(frame_gray.astype(np.float32))
        if len(self.frame_buffer) > self.history:
            self.frame_buffer.pop(0)

        self.frame_count += 1
        if self.frame_count % self.update_every == 0:
            # Weighted median – newer frames have higher weight?
            # Simpler: compute median of buffer
            stack = np.stack(self.frame_buffer, axis=0)
            self.background = np.median(stack, axis=0).astype(np.uint8)

        if self.background is None:
            # Not enough frames yet – return current frame as rough bg
            return frame_gray
        return self.background

    def get_foreground(self, frame_gray, threshold=25):
        """Return binary mask of moving objects."""
        if self.background is None:
            return np.zeros_like(frame_gray, dtype=np.uint8)
        diff = cv2.absdiff(frame_gray, self.background)
        _, fg = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        return fg

