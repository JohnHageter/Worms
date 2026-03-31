import numpy as np


class WormTrack:
    def __init__(self, track_id, worm, frame_idx):
        self.id = track_id
        self.last_centroid = np.array(worm["centroid"], dtype=float)

        self.head = worm["end1"]
        self.tail = worm["end2"]

        self.last_frame = frame_idx
        self.age = 1
        self.missed = 0
        self.well_id = worm["well_id"]

        self.velocity = np.zeros(2)


class WormTracker:
    def __init__(self, max_dist=25, max_missed=5):
        self.max_dist = max_dist
        self.max_missed = max_missed
        self.tracks = []
        self.next_id = 0

    def _match_tracks(self, tracks, worms):
        matches = []
        used_tracks = set()
        used_worms = set()

        for ti, t in enumerate(tracks):
            best = None
            best_dist = self.max_dist

            for wi, w in enumerate(worms):
                if wi in used_worms:
                    continue

                d = np.linalg.norm(t.last_centroid - w["centroid"])
                if d < best_dist:
                    best_dist = d
                    best = wi

            if best is not None:
                matches.append((ti, best))
                used_tracks.add(ti)
                used_worms.add(best)

        return matches, used_tracks, used_worms

    def order_endpoints(self, end1, end2, prev_end1, prev_end2):
        if end1 is None or end2 is None:
            return end1, end2
        if prev_end1 is None or prev_end2 is None:
            return end1, end2

        # Assign end1 to whichever is closer to previous end1
        if np.hypot(end2[0] - prev_end1[0], end2[1] - prev_end1[1]) < np.hypot(
            end1[0] - prev_end1[0], end1[1] - prev_end1[1]
        ):
            # swap
            return end2, end1
        return end1, end2


    def assign_head_tail(self, t, centroid, endA, endB):
        if endA is None or endB is None:
            return t.head, t.tail

        centroid = np.array(centroid, dtype=float)
        prev_centroid = t.last_centroid

        # --- Update smoothed velocity ---
        step = centroid - prev_centroid
        t.velocity = 0.7 * t.velocity + 0.3 * step

        if np.linalg.norm(t.velocity) < 1e-3:
            # Too little motion → keep previous assignment
            return t.head, t.tail

        # --- Candidate vectors ---
        vA = np.array(endA) - centroid
        vB = np.array(endB) - centroid

        # --- Alignment with motion ---
        dotA = np.dot(vA, t.velocity)
        dotB = np.dot(vB, t.velocity)

        # --- Choose head based on motion direction ---
        if dotA > dotB:
            new_head, new_tail = endA, endB
        else:
            new_head, new_tail = endB, endA

        # --- Anti-flip stabilization ---
        # Only switch if clearly better than previous head
        if t.head is not None:
            prev_vec = np.array(t.head) - centroid
            prev_score = np.dot(prev_vec, t.velocity)

            new_vec = np.array(new_head) - centroid
            new_score = np.dot(new_vec, t.velocity)

            # Require margin to flip
            if new_score < prev_score * 1.2:
                return t.head, t.tail

        return new_head, new_tail

    def update(self, worms, frame_idx):
        by_well = {}
        for w in worms:
            by_well.setdefault(w["well_id"], []).append(w)

        for well_id, dets in by_well.items():
            well_tracks = [t for t in self.tracks if t.well_id == well_id]

            matches, used_tracks, used_dets = self._match_tracks(well_tracks, dets)

            for ti, di in matches:
                t = well_tracks[ti]
                w = dets[di]

                head, tail = self.assign_head_tail(
                    t,
                    w["centroid"],
                    w["end1"],
                    w["end2"]
                )

                t.head = head
                t.tail = tail
                t.last_centroid = np.array(w["centroid"], dtype=float)
                t.last_frame = frame_idx
                t.age += 1
                t.missed = 0


            for t in well_tracks:
                if t not in [well_tracks[ti] for ti in used_tracks]:
                    t.missed += 1

            for di, w in enumerate(dets):
                if di not in used_dets:
                    self.tracks.append(WormTrack(self.next_id, w, frame_idx))
                    self.next_id += 1

        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]

        return self.tracks
