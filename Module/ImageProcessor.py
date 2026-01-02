import numpy as np
from Module.Dataset import load_frame
import cv2
from skimage.morphology import skeletonize
from collections import defaultdict, deque

def snap_background(image, window=11, smooth_iters=3):
    bg = cv2.medianBlur(image, window)

    for _ in range(smooth_iters):
        bg = cv2.GaussianBlur(bg, (201, 201), sigmaX=3)

    return bg

def blob_metrics(binary):
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    metrics = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 10:
            continue

        length = None
        thickness = None

        if len(cnt) >= 5:
            (_, _), (MA, ma), _ = cv2.fitEllipse(cnt)
            length = max(MA, ma)
            thickness = min(MA, ma)

        metrics.append({
            "contour": cnt,
            "area": area,
            "length": length,
            "thickness": thickness
        })

    return metrics

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

def extract_foreground(img, background, thresh_val=35):
    fg = cv2.subtract(background, img)
    _, binary = cv2.threshold(fg, thresh_val, 255, cv2.THRESH_BINARY)
    # binary = cv2.adaptiveThreshold(fg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                                    cv2.THRESH_BINARY, 15, -5)
    return fg, binary

def find_components(binary):
    return cv2.connectedComponentsWithStats(binary, connectivity=8)

def centroid_in_any_well(cx, cy, wells):
    for well in wells:
        if centroid_within_well(cx, cy, well):
            return True
    return False

def centroid_within_well(x,y, well):
    wx, wy, r = well
    return (x - wx)**2 + (y - wy)**2 <= r**2

def extract_curve(mask, bbox, area, min_area=50):
    if area <= min_area:
        return None, None, None, 0

    skel = skeletonize(mask.astype(bool)).astype(np.uint8)
    path = longest_skeleton_path(skel)

    if len(path) < 2:
        return None, None, None, 0

    path = np.array(path)
    x, y, _, _ = bbox

    curve_centroid = path.mean(axis=0)
    end1 = path[0]
    end2 = path[-1]

    curve_centroid = (curve_centroid[0] + x, curve_centroid[1] + y)
    end1 = (end1[0] + x, end1[1] + y)
    end2 = (end2[0] + x, end2[1] + y)

    return curve_centroid, end1, end2, len(path)

def longest_skeleton_path(skel):
    ys, xs = np.where(skel > 0)
    if len(xs) < 2:
        return []

    points = list(zip(xs, ys))
    graph = defaultdict(list)

    neighbors = [(-1,-1), (-1,0), (-1,1),
                 ( 0,-1),         ( 0,1),
                 ( 1,-1), ( 1,0), ( 1,1)]

    for x, y in points:
        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            if (nx, ny) in set(points):
                graph[(x,y)].append((nx,ny))

    endpoints = [p for p in graph if len(graph[p]) == 1]
    if not endpoints:
        return []

    def bfs(start):
        prev = {start: None}
        q = deque([start])
        while q:
            u = q.popleft()
            for v in graph[u]:
                if v not in prev:
                    prev[v] = u
                    q.append(v)
        end = max(prev, key=lambda p: (p[0]-start[0])**2 + (p[1]-start[1])**2)
        path = []
        cur = end
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        return path[::-1]

    longest = []
    for ep in endpoints:
        p = bfs(ep)
        if len(p) > len(longest):
            longest = p

    return longest

def build_worms(labels, stats, centroids, wells, minimum_skeleton_area=10):
    worms = []

    for label in range(1, stats.shape[0]):
        x, y, w, h, area = stats[label]
        cx, cy = centroids[label]

        well_id = None
        for i, well in enumerate(wells):
            if centroid_within_well(cx, cy, well):
                well_id = i
                break

        if well_id is None:
            continue

        mask = (labels[y:y+h, x:x+w] == label).astype(np.uint8) * 255

        curve_centroid, end1, end2, curve_length = extract_curve(
            mask, (x, y, w, h), area, min_area=minimum_skeleton_area
        )

        worms.append({
            "well_id": well_id,
            "mask": mask,
            "bbox": (x, y, w, h),
            "area": area,
            "centroid": (cx, cy),
            "curve_centroid": curve_centroid,
            "end1": end1,
            "end2": end2,
            "curve_length": curve_length
        })

    return worms

def filter_worms(worms, min_area=10, max_area=500, min_thickness=1):
    kept = []

    for worm in worms:
        area = worm["area"]
        if area < min_area or area > max_area:
            continue

        dist_map = cv2.distanceTransform(worm["mask"], cv2.DIST_L2, 5)
        if dist_map.max() < min_thickness:
            continue

        kept.append(worm)

    return kept
