class WormAnalyzer:
    def __init__(
        self,
        min_motion=2.0,
        strong_update=0.5,
        flip_threshold=-3.0,
    ):
        self.min_motion = min_motion
        self.strong_update = strong_update
        self.flip_threshold = flip_threshold

    def update(self, track):

        if track.endpoints is None:
            return

        p1, p2 = track.endpoints
        if "head" not in track.features:
            track.features["head"] = p1.copy()
            track.features["tail"] = p2.copy()
            track.features["confidence"] = 0.0
            return
