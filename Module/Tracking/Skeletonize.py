from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
from collections import deque
from skimage.morphology import skeletonize

# 8-connected neighborhood
_NEI = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _build_graph(skel: np.ndarray):
    """
    Build adjacency list for skeleton pixels.
    Returns:
      coords: (N,2) array of (y,x) skeleton pixel coords
      adj: list[list[int]] adjacency by index into coords
      idx_map: dict[(y,x)] -> idx
    """
    ys, xs = np.where(skel)
    coords = np.column_stack([ys, xs]).astype(np.int32)
    if coords.shape[0] == 0:
        return coords, [], {}

    idx_map = {(int(y), int(x)): i for i, (y, x) in enumerate(coords)}
    adj = [[] for _ in range(coords.shape[0])]

    for i, (y, x) in enumerate(coords):
        for dy, dx in _NEI:
            j = idx_map.get((int(y + dy), int(x + dx)))
            if j is not None:
                adj[i].append(j)

    return coords, adj, idx_map


def _bfs_farthest(
    start: int, adj: list[list[int]]
) -> Tuple[int, dict[int, int], dict[int, float]]:
    """
    BFS on unweighted graph, but we keep predecessor and hop distance.
    Returns farthest node, predecessor map, distance map.
    """
    prev = {start: -1}
    dist = {start: 0.0}
    q = deque([start])

    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1.0
                prev[v] = u
                q.append(v)

    farthest = max(dist, key=dist.get) # type: ignore
    return farthest, prev, dist


def _recover_path(end: int, prev: dict[int, int]) -> list[int]:
    path = []
    cur = end
    while cur != -1:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def skeleton_longest_path(
    mask: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
    """
    Returns:
      endpoints: (2,2) float array in mask coordinates: [[x1,y1],[x2,y2]] or None
      path_xy: (M,2) float array in mask coordinates (x,y) ordered, or None
      length: float (approx pixels along path)
    """
    skel = skeletonize(mask > 0)
    coords, adj, _ = _build_graph(skel)

    if coords.shape[0] < 2:
        return None, None, 0.0

    # Two-pass BFS gives a good approximation of graph diameter
    a, _, _ = _bfs_farthest(0, adj)
    b, prev, _ = _bfs_farthest(a, adj)

    path_idx = _recover_path(b, prev)
    path_yx = coords[path_idx]  # (y,x)

    # Compute Euclidean length along the ordered path
    # Convert to (x,y) for downstream use
    path_xy = np.column_stack([path_yx[:, 1], path_yx[:, 0]]).astype(np.float32)
    if path_xy.shape[0] < 2:
        return None, None, 0.0

    diffs = np.diff(path_xy, axis=0)
    length = float(np.sum(np.linalg.norm(diffs, axis=1)))

    endpoints = np.vstack([path_xy[0], path_xy[-1]]).astype(np.float32)  # [[x,y],[x,y]]
    return endpoints, path_xy, length
