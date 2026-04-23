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
    bg = np.max(stack, axis=0)

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

    backgrounds = {}
    for well_id, crops in accum.items():
        if len(crops) == 0:
            raise RuntimeError(f"No background frames for well {well_id}")
        backgrounds[well_id] = np.max(np.stack(crops, axis=0), axis=0).astype(
            np.uint8
        )

    return backgrounds


def sample_per_well_backgrounds_masked(
    cap,
    wells,
    n_frames=2000,
    motion_thresh=8,
    frame_step=5,  
):
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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

            if exclusion_masks[i] is None:
                exclusion_masks[i] = np.zeros(crop.shape, dtype=bool)

            if prev_crops[i] is not None:
                diff = cv2.absdiff(crop, prev_crops[i])
                moving = diff > motion_thresh
                exclusion_masks[i] |= moving

            prev_crops[i] = crop
            stacks[i].append(crop)

    backgrounds = {}
    for i, frames in stacks.items():
        stack = np.stack(frames).astype(np.float32)
        mask = exclusion_masks[i]
        stack[:, mask] = np.nan
        bg = np.nanmedian(stack, axis=0)
        if np.any(np.isnan(bg)):
            global_med = np.nanmedian(bg)
            bg[np.isnan(bg)] = global_med
        backgrounds[i] = bg.astype(np.uint8)

    return backgrounds


class RollingMedianBackground:
    def __init__(self, history=300, update_every=100):
        self.history = history
        self.update_every = update_every
        self.frame_buffer = []
        self.background = None
        self.frame_count = 0

    def update(self, frame_gray, fg_mask=None):
        """
        frame_gray : current well crop
        fg_mask    : binary foreground mask (255 = foreground)
        """
        self.frame_count += 1

        frame = frame_gray.copy()

        if fg_mask is not None:
            frame = frame.astype(np.float32)
            frame[fg_mask > 0] = np.nan

        self.frame_buffer.append(frame)

        if len(self.frame_buffer) > self.history:
            self.frame_buffer.pop(0)

    def recompute(self):
        if not self.frame_buffer:
            return
        stack = np.stack(self.frame_buffer, axis=0)
        self.background = np.median(stack, axis=0).astype(np.uint8)


    def get_foreground(self, frame_gray, rel_thresh=0.02, abs_thresh=8):
        """
        Detect dark objects on a bright background using
        BOTH relative contrast and absolute difference.
        """
        if self.background is None:
            return np.zeros_like(frame_gray, dtype=np.uint8)

        bg = self.background.astype(np.float32)
        im = frame_gray.astype(np.float32)

        diff = bg - im
        rel = diff / (bg + 1e-6)

        fg = np.zeros_like(frame_gray, dtype=np.uint8)

        fg[(rel > rel_thresh) | (diff > abs_thresh)] = 255

        return fg
