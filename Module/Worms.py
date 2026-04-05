import numpy as np


class WormDetection:
    def __init__(
        self,
        centroid,
        curve_centroid=None,
        end1=None,
        end2=None,
        area=0,
        mask=None,
        bbox=None,
        curve_length=0,
        well_id=None,
    ):
        self.centroid = np.array(centroid, dtype=float)
        self.curve_centroid = (
            np.array(curve_centroid, dtype=float)
            if curve_centroid is not None
            else None
        )
        self.end1 = np.array(end1, dtype=float) if end1 is not None else None
        self.end2 = np.array(end2, dtype=float) if end2 is not None else None
        self.area = area
        self.mask = mask
        self.bbox = bbox
        self.curve_length = curve_length
        self.well_id = well_id


class WormTrack:
    def __init__(self, worm_id, well_id, centroid=None):
        self.id = worm_id
        self.well_id = well_id
        self.centroid = (
            np.array([np.nan, np.nan], dtype=float)
            if centroid is None
            else np.array(centroid, dtype=float)
        )
        self.head = None
        self.tail = None
        self._last_centroid = None
        self.velocity = np.array([0.0, 0.0], dtype=float)
        self.missed = 0

    def update(self, centroid, skeleton=None):
        """Update track with new detection and stabilize head/tail."""
        if centroid is not None:
            centroid = np.array(centroid, dtype=float)

            # Update velocity
            if self._last_centroid is not None:
                self.velocity = centroid - self._last_centroid
            else:
                self.velocity = np.array([0.0, 0.0])

            self.centroid = centroid
            self._last_centroid = centroid

            # --- Head/tail assignment ---
            if skeleton is not None:
                # Clean skeleton points (skip None)
                skeleton_pts = np.array([pt for pt in skeleton if pt is not None], dtype=float)
                if skeleton_pts.size > 0:
                    if self.head is None or self.tail is None:
                        # First frame: assign using farthest from previous centroid
                        diffs = skeleton_pts - (centroid - self.velocity)
                        dists = np.linalg.norm(diffs, axis=1)
                        self.head = tuple(skeleton_pts[np.argmax(dists)])
                        self.tail = tuple(skeleton_pts[np.argmin(dists)])
                    else:
                        # Assign new endpoints based on previous head/tail
                        d_head = np.linalg.norm(skeleton_pts - self.head, axis=1)
                        d_tail = np.linalg.norm(skeleton_pts - self.tail, axis=1)
                        new_head = skeleton_pts[np.argmin(d_head)]
                        new_tail = skeleton_pts[np.argmin(d_tail)]

                        # Optional: enforce minimum separation to avoid flip
                        if np.linalg.norm(new_head - new_tail) > 1e-3:
                            self.head = tuple(new_head)
                            self.tail = tuple(new_tail)

                        # If worm is moving, bias head toward motion direction
                        if np.linalg.norm(self.velocity) > 1.0:
                            motion_point = centroid + self.velocity
                            d_motion = np.linalg.norm(skeleton_pts - motion_point, axis=1)
                            motion_head = skeleton_pts[np.argmin(d_motion)]
                            # Keep previous head if distance is small, otherwise update
                            if np.linalg.norm(np.array(self.head) - motion_head) > 1.0:
                                self.head = tuple(motion_head)
                                # Assign tail as opposite endpoint
                                other_idx = np.argmin(np.linalg.norm(skeleton_pts - motion_head, axis=1))
                                self.tail = tuple(skeleton_pts[other_idx])

            self.missed = 0
        else:
            self.mark_missed()


    def mark_missed(self):
        """Called when the worm is not detected in the current frame."""
        self.centroid = np.array([np.nan, np.nan], dtype=float)
        self.head = None
        self.tail = None
        self.velocity = np.array([0.0, 0.0], dtype=float)
        self.missed += 1

    def is_lost(self, max_missed):
        """Check if the worm should be removed."""
        return self.missed > max_missed


class WormTracker:
    def __init__(self, max_dist=50, max_missed=5):
        self.tracks = []
        self.next_id = 1
        self.max_dist = max_dist
        self.max_missed = max_missed

    def update(self, detected_worms):
        """
        Update all tracks with new detections.
        detected_worms: list of WormDetection objects
        Returns: list of WormTrack objects
        """
        matched_detections = set()

        # --- Update existing tracks ---
        for track in self.tracks:
            # Only consider detections in the same well
            candidates = [
                w
                for w in detected_worms
                if w.well_id == track.well_id and id(w) not in matched_detections
            ]

            best_match = None
            best_dist = self.max_dist + 1

            for w in candidates:
                last_centroid = (
                    track._last_centroid
                    if track._last_centroid is not None
                    else w.centroid
                )
                dist = np.linalg.norm(last_centroid - w.centroid)
                if dist < best_dist:
                    best_dist = dist
                    best_match = w

            if best_match is not None:
                track.update(
                    best_match.centroid, skeleton=[best_match.end1, best_match.end2]
                )
                matched_detections.add(id(best_match))
            else:
                track.mark_missed()

        # --- Create new tracks for unmatched detections ---
        for w in detected_worms:
            if id(w) not in matched_detections:
                new_track = WormTrack(self.next_id, w.well_id, centroid=w.centroid)
                new_track.update(w.centroid, skeleton=[w.end1, w.end2])
                self.tracks.append(new_track)
                self.next_id += 1

        # --- Remove lost tracks ---
        self.tracks = [t for t in self.tracks if not t.is_lost(self.max_missed)]

        return self.tracks
