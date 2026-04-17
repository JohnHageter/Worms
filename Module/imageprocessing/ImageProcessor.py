import numpy as np
import cv2
from skimage.morphology import skeletonize
from collections import defaultdict, deque

def mask_to_well(binary, well):
    mask = np.zeros_like(binary, dtype=np.uint8)
    x, y, r = map(int, well)
    cv2.circle(mask, (x, y), r, 255, -1)
    return cv2.bitwise_and(binary, mask)


def filter_worms(worms, min_area=10, max_area=500):
    return [w for w in worms if min_area <= w.area <= max_area]

def find_components(binary):
    return cv2.connectedComponentsWithStats(binary, connectivity=8)


def extract_curve(mask, bbox, area, min_area=50):
    """
    Extract skeleton curve endpoints and centroid from a worm mask.

    Parameters
    ----------
    mask : 2D np.ndarray
        Binary mask of a worm (1s for worm pixels)
    bbox : tuple
        Bounding box (x, y, w, h) of the mask in the original frame
    area : int
        Area of the blob
    min_area : int
        Minimum area to consider

    Returns
    -------
    curve_centroid : tuple
        Centroid of skeleton path (x, y)
    end1 : tuple
        Endpoint 1 of skeleton (x, y)
    end2 : tuple
        Endpoint 2 of skeleton (x, y)
    path_length : int
        Number of pixels in the skeleton path
    """
    if area < min_area:
        return None, None, None, 0

    skel = skeletonize(mask.astype(bool)).astype(np.uint8)
    path = longest_skeleton_path(skel)
    
    if len(path) < 5:  # too short to assign head/tail
        return None, None, None, len(path)

    path = np.array(path)
    x0, y0, _, _ = bbox
    end1 = (path[0][0] + x0, path[0][1] + y0)
    end2 = (path[-1][0] + x0, path[-1][1] + y0)
    curve_centroid = (path[:,0].mean() + x0, path[:,1].mean() + y0)

    return curve_centroid, end1, end2, len(path)

def longest_skeleton_path(skel):
    """
    Find the longest connected path through a skeletonized binary image.
    Handles broken skeletons and small gaps robustly.
    
    Parameters
    ----------
    skel : 2D np.ndarray (binary)
        Skeletonized worm mask (1-pixel wide line)
    
    Returns
    -------
    path : list of (x, y)
        Coordinates along the longest skeleton path
    """
    ys, xs = np.where(skel > 0)
    if len(xs) < 2:
        return []

    points = set(zip(xs, ys))
    graph = defaultdict(list)
    neighbors = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    for x, y in points:
        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            if (nx, ny) in points:
                graph[(x, y)].append((nx, ny))

    endpoints = [p for p in graph if len(graph[p]) == 1]
    if not endpoints:
        endpoints = list(points)  # fallback if skeleton looped

    def bfs_longest(start):
        prev = {start: None}
        q = deque([start])
        visited = set([start])
        while q:
            u = q.popleft()
            for v in graph[u]:
                if v not in visited:
                    prev[v] = u
                    visited.add(v)
                    q.append(v)
        max_dist = -1
        farthest = start
        for p in prev:
            dx = p[0] - start[0]
            dy = p[1] - start[1]
            d = dx*dx + dy*dy
            if d > max_dist:
                max_dist = d
                farthest = p
        path = []
        cur = farthest
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        return path[::-1]

    longest = []
    for ep in endpoints:
        path = bfs_longest(ep)
        if len(path) > len(longest):
            longest = path

    return longest

