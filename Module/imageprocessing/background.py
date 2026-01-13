import cv2
import numpy as np
from Module.dataset.Dataset import load_frame


def snap_background(image, window=11, smooth_iters=3):
    bg = cv2.medianBlur(image, window)

    for _ in range(smooth_iters):
        bg = cv2.GaussianBlur(bg, (201, 201), sigmaX=3)

    return bg


def sample_background(image_data, n_frames=300, seed=None):
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


def filter_homomorphic(image, cutoff=30, gamma_h=2.0, gamma_l=0.5):
    img_log = np.log1p(image)

    dft = np.fft.fft2(img_log)
    dft_shift = np.fft.fftshift(dft)

    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    u = np.arange(-crow, crow)
    v = np.arange(-ccol, ccol)
    U, V = np.meshgrid(u, v, indexing='ij')
    D = np.sqrt(U**2 + V**2)
    H = (gamma_h - gamma_l)*(1 - np.exp(-(D**2)/(2*(cutoff**2)))) + gamma_l

    dft_shift_filtered = dft_shift * H

    dft_ifft = np.fft.ifftshift(dft_shift_filtered)
    img_filtered = np.fft.ifft2(dft_ifft)
    img_filtered = np.real(img_filtered)

    img_out = np.expm1(img_filtered)

    img_out = cv2.normalize(img_out, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return img_out