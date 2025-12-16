import numpy as np
import cv2
from scipy.spatial.distance import cdist
import uuid
import numpy as np
import cv2


def estimate_head(prev_centroid, current_centroid):
    """Return the direction vector from previous centroid to current."""
    if prev_centroid is None:
        return np.array([0.0, 0.0])
    return (current_centroid - prev_centroid)


def assign_detections_to_tracks(track_centroids, det_centroids, max_dist=25):
    """
    nearest-neighbor assignment.
    Returns (matches, unassigned_tracks, unassigned_detections)
    """

    track_centroids_clean = []
    track_index_map = []   

    for i, c in enumerate(track_centroids):
        if c is not None and len(c) == 2 and not np.any(np.isnan(c)):
            track_centroids_clean.append(c)
            track_index_map.append(i)

    det_centroids_clean = []
    det_index_map = []     

    for j, c in enumerate(det_centroids):
        if c is not None and len(c) == 2 and not np.any(np.isnan(c)):
            det_centroids_clean.append(c)
            det_index_map.append(j)

    track_arr = np.array(track_centroids_clean, dtype=float).reshape(-1, 2)
    det_arr = np.array(det_centroids_clean, dtype=float).reshape(-1, 2)

    if len(track_arr) == 0 and len(det_arr) == 0:
        return [], [], []
    if len(track_arr) == 0:
        return [], [], det_index_map
    if len(det_arr) == 0:
        return [], track_index_map, []

    dist_matrix = cdist(track_arr, det_arr)

    matches = []
    used_track = set()
    used_det = set()

    for _ in range(min(len(track_arr), len(det_arr))):
        i, j = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
        if dist_matrix[i, j] > max_dist:
            break

        matches.append((track_index_map[i], det_index_map[j]))
        used_track.add(i)
        used_det.add(j)

        dist_matrix[i, :] = np.inf
        dist_matrix[:, j] = np.inf

    unassigned_tracks = [track_index_map[i] for i in range(len(track_arr)) if i not in used_track]
    unassigned_dets = [det_index_map[j] for j in range(len(det_arr)) if j not in used_det]

    return matches, unassigned_tracks, unassigned_dets



def extract_blob_keypoints(mask):
    """
    mask: binary mask of a single blob
    Returns: centroid, head, tail, area
    """
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None, None, 0

    cnt = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(cnt)

    M = cv2.moments(cnt)
    if M["m00"] == 0:
        return None, None, None, area
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    centroid = np.array([cx, cy])

    # PCA to estimate head/tail orientation
    pts = cnt.reshape(-1, 2).astype(np.float32)
    mean = np.mean(pts, axis=0)
    pts_centered = pts - mean
    cov = np.cov(pts_centered.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    direction = eigvecs[:, np.argmax(eigvals)]  # principal axis

    head = centroid + direction * 10
    tail = centroid - direction * 10

    return centroid, head, tail, area


def track_worms(video, well_masks, fg_masks,
                min_area_for_head_tail=150,
                max_tracking_dist=20,
                max_still_frames=7):
    """
    video: (frames, H, W)
    well_masks: list of masks for each well
    fg_masks: (frames, H, W) foreground-only masks
    """

    num_frames = video.shape[0]
    num_wells = len(well_masks)
    wells_tracks = []

    for w in range(num_wells):
        wells_tracks.append({})  # track_id → track_data

    def new_id():
        return str(uuid.uuid4())[:8]

    lineage_parent = {}  # child_id → parent_id

    for f in range(num_frames):
        frame_fg = fg_masks[f]

        for w, well_mask in enumerate(well_masks):
            tracks = wells_tracks[w]
            masked_fg = (frame_fg * well_mask).astype(np.uint8)
            num_labels, labels = cv2.connectedComponents(masked_fg)

            det_centroids = []
            det_keypoints = []
            det_areas = []

            for lab in range(1, num_labels):
                blob_mask = (labels == lab).astype(np.uint8)
                centroid, head, tail, area = extract_blob_keypoints(blob_mask)

                if centroid is None:
                    continue

                if area < min_area_for_head_tail:
                    head = None
                    tail = None

                det_centroids.append(centroid)
                det_keypoints.append((centroid, head, tail))
                det_areas.append(area)

            track_ids = list(tracks.keys())
            track_centroids = [tracks[t]["centroid"] for t in track_ids]

            matches, unassigned_tracks, unassigned_dets = assign_detections_to_tracks(track_centroids, det_centroids, 
                                                                                      max_dist=max_tracking_dist)


            for ti, di in matches:
                tid = track_ids[ti]
                centroid, head, tail = det_keypoints[di]

                prev_cent = tracks[tid]["centroid"]
                direction = estimate_head(prev_cent, centroid)


                if np.linalg.norm(direction) < 1.0:
                    centroid = prev_cent
                    head = tracks[tid]["head"]
                    tail = tracks[tid]["tail"]
                else:
                    if head is None:
                        head = centroid + direction / np.linalg.norm(direction) * 10
                    if tail is None:
                        tail = centroid - direction / np.linalg.norm(direction) * 10

                tracks[tid]["centroid"] = centroid
                tracks[tid]["head"] = head
                tracks[tid]["tail"] = tail
                tracks[tid]["last_moved"] = f
                tracks[tid]["history"].append((f, centroid, head, tail))


            for ti in unassigned_tracks:
                # Keep tracks indefinitely if no movement detected
                tid = track_ids[ti]
                tracks[tid]["still_frames"] += 1
                cx, cy = tracks[tid]["centroid"]
                hx, hy = tracks[tid]["head"], tracks[tid]["head"]
                tx, ty = tracks[tid]["tail"], tracks[tid]["tail"]

                tracks[tid]["history"].append((f, tracks[tid]["centroid"],
                                                tracks[tid]["head"],
                                                tracks[tid]["tail"]))                


            for di in unassigned_dets:
                centroid, head, tail = det_keypoints[di]

                new_tid = new_id()

                # Fission detection: if very close to an existing track and this is extra blob
                parent = None
                if len(matches) > 0:
                    used_det_idxs = [md[1] for md in matches]
                    if di not in used_det_idxs:
                        # find nearest existing track
                        dists = [np.linalg.norm(centroid - tc) for tc in track_centroids]
                        nearest_track_i = int(np.argmin(dists)) if len(dists) else None
                        if nearest_track_i is not None and dists[nearest_track_i] < (max_tracking_dist * 1.2):
                            parent = track_ids[nearest_track_i]

                tracks[new_tid] = {
                    "id": new_tid,
                    "centroid": centroid,
                    "head": head,
                    "tail": tail,
                    "last_moved": f,
                    "still_frames": 0,
                    "history": [(f,centroid,head,tail)]
                }

                if parent:
                    lineage_parent[new_tid] = parent

    return wells_tracks, lineage_parent


def visualize_tracking(video, wells_tracks, output_path=None, fps=30):
    """
    OpenCV visualization of tracking.
    
    video: numpy array of shape (frames, H, W)
    wells_tracks: list of track dicts (each containing "history": [(f, centroid, head, tail), ...])
    output_path: path to save mp4 (or None for display only)
    fps: video write FPS
    """

    num_frames = video.shape[0]
    H, W = video.shape[1], video.shape[2]

    if output_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H), isColor=True)

    for f in range(num_frames):
        frame_gray = video[f]
        frame_bgr = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)

        # Draw keypoints for only this frame
        for well_id, tracks in enumerate(wells_tracks):
            for tid, tr in tracks.items():
                # find this frame's entry
                for (tf, centroid, head, tail) in tr["history"]:
                    if tf != f:
                        continue

                    cx, cy = centroid
                    cx, cy = int(cx), int(cy)

                    # Draw centroid
                    cv2.circle(frame_bgr, (cx, cy), 3, (0, 255, 0), -1)

                    # Construct point set tail → centroid → head
                    pts = []

                    if tail is not None:
                        tx, ty = map(int, tail)
                        pts.append((tx, ty))

                    pts.append((cx, cy))

                    if head is not None:
                        hx, hy = map(int, head)
                        pts.append((hx, hy))

                    # Draw a polyline
                    if len(pts) > 1:
                        cv2.polylines(
                            frame_bgr,
                            [np.array(pts, dtype=np.int32)],
                            isClosed=False,
                            color=(0, 255, 0),
                            thickness=2
                        )

        # Show or save
        if output_path is None:
            cv2.imshow("Tracking", frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            writer.write(frame_bgr)

    if output_path is not None:
        writer.release()
    cv2.destroyAllWindows()