import numpy as np


class Track:
    """Closed track: a centerline plus a constant width. Units are meters."""

    def __init__(self, centerline, width=14.0, name="Track"):
        self.name = name
        self.center = np.asarray(centerline, dtype=float)      # (N, 2)
        self.width = width
        n = len(self.center)

        # tangent and normal at each centerline point
        nxt = np.roll(self.center, -1, axis=0)
        prv = np.roll(self.center, 1, axis=0)
        tang = nxt - prv
        tang /= np.linalg.norm(tang, axis=1, keepdims=True)
        self.tangent = tang
        normal = np.stack([-tang[:, 1], tang[:, 0]], axis=1)   # points to the left
        self.normal = normal
        self.tangent_angles = np.arctan2(tang[:, 1], tang[:, 0])

        self.left = self.center + normal * width / 2
        self.right = self.center - normal * width / 2

        # all boundary segments as (M, 2, 2): [[ax, ay], [bx, by]]
        self.segments = np.concatenate([self._segs(self.left), self._segs(self.right)])
        self.seg_midpoints = (self.segments[:, 0] + self.segments[:, 1]) * 0.5

        # arc length along the centerline, used for progress
        step = np.linalg.norm(nxt - self.center, axis=1)
        self.cum_len = np.concatenate([[0], np.cumsum(step)[:-1]])
        self.length = step.sum()

    @staticmethod
    def _segs(poly):
        return np.stack([poly, np.roll(poly, -1, axis=0)], axis=1)

    @classmethod
    def random(cls, seed=None, n_points=300, base_radius=150.0, width=14.0):
        """Closed loop r(theta) = base + a few random harmonics.
        A polar curve cannot cross itself, so the track is always valid."""
        rng = np.random.default_rng(seed)
        theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        r = np.full_like(theta, base_radius)
        for k in (2, 3, 4, 5):
            amp = rng.uniform(0.0, 0.20 / (k - 1)) * base_radius
            r += amp * np.cos(k * theta + rng.uniform(0, 2 * np.pi))
        pts = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
        return cls(pts, width, name=f"Random #{seed}")

    @classmethod
    def random_extreme(cls, seed=None, base_radius=170.0, width=11.0, n_points=360):
        """Procedural track with extreme turns, hairpins, and rapid chicane complexes.

        Uses higher-frequency harmonics and adaptive validation to ensure the layout
        remains mathematically sound (no self-intersections or illegal pinch points)
        while creating aggressive, highly technical racing challenges.
        """
        from sim.tracks import validate, _resample, SPACING

        rng = np.random.default_rng(seed)
        for attempt in range(50):
            sub_seed = int(rng.integers(0, 1_000_000_000))
            sub_rng = np.random.default_rng(sub_seed)
            theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
            r = np.full_like(theta, base_radius)

            # Low harmonics: overall shape (peanut, oval, asymmetric lobe)
            for k in (2, 3):
                amp = sub_rng.uniform(0.12, 0.28) * base_radius
                r += amp * np.cos(k * theta + sub_rng.uniform(0, 2 * np.pi))

            # Medium harmonics: hairpins, deep braking pockets, stadium corners
            for k in (4, 5, 6, 7):
                amp = sub_rng.uniform(0.06, 0.16) * base_radius
                r += amp * np.cos(k * theta + sub_rng.uniform(0, 2 * np.pi))

            # High harmonics: rapid chicane flicking and esses
            for k in (8, 9, 10):
                amp = sub_rng.uniform(0.02, 0.08) * base_radius
                r += amp * np.cos(k * theta + sub_rng.uniform(0, 2 * np.pi))

            pts = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
            pts = _resample(pts, SPACING)
            track_name = f"Random Extreme #{seed if seed is not None else sub_seed}"
            track = cls(pts, width, name=track_name)
            track.country = "Procedural"
            track.real_km = track.length * 2.0 / 1000.0  # approximate scaled km
            track.note = "Procedurally generated extreme technical circuit with hairpins and chicanes."
            if not validate(track):
                return track

        # Fallback to smooth track if extreme generation could not settle within 50 attempts
        return cls.random(seed=seed, base_radius=base_radius, width=width)

    # ---- queries ----
    def nearest_index(self, pos):
        """Nearest centerline point index using fast squared distance."""
        dx = self.center[:, 0] - pos[0]
        dy = self.center[:, 1] - pos[1]
        return int(np.argmin(dx * dx + dy * dy))

    def contains(self, pos, nearest_i=None):
        """True if pos is on the track (within half-width of the centerline)."""
        if nearest_i is None:
            nearest_i = self.nearest_index(pos)
        dx = self.center[nearest_i, 0] - pos[0]
        dy = self.center[nearest_i, 1] - pos[1]
        return bool((dx * dx + dy * dy) <= (self.width * 0.5) ** 2)

    def progress(self, pos, nearest_i=None):
        """Distance along the centerline, in meters."""
        if nearest_i is None:
            nearest_i = self.nearest_index(pos)
        return float(self.cum_len[nearest_i])

    def query(self, pos):
        """Single-pass query returning (on_track: bool, progress: float, nearest_index: int)."""
        i = self.nearest_index(pos)
        dx = self.center[i, 0] - pos[0]
        dy = self.center[i, 1] - pos[1]
        on_track = bool((dx * dx + dy * dy) <= (self.width * 0.5) ** 2)
        return on_track, float(self.cum_len[i]), i

    def get_track_telemetry(self, pos, heading, nearest_i=None, lookahead_200=False):
        """Compute lateral position, heading error, and upcoming curvature.

        Returns:
            lat_pos: float in [-1.0, 1.0] (fraction of half-width: +left kerb, -right kerb)
            angle_to_tangent: float in [-1.0, 1.0] (relative heading to track direction)
            curve_50: float in [-1.0, 1.0] (curvature 50m ahead: +left, -right)
            curve_100: float in [-1.0, 1.0] (curvature 100m ahead: +left, -right)
            curve_200: float in [-1.0, 1.0] (only if lookahead_200=True: curvature 200m ahead)
        """
        if nearest_i is None:
            nearest_i = self.nearest_index(pos)
        i = nearest_i
        n = len(self.center)

        # 1. Lateral offset from centerline (+ left, - right)
        delta = pos - self.center[i]
        lat_offset = float(delta[0] * self.normal[i, 0] + delta[1] * self.normal[i, 1])
        lat_pos = float(np.clip(lat_offset / (self.width * 0.5), -1.5, 1.5))

        # 2. Heading alignment relative to track tangent
        tang_angle = self.tangent_angles[i]
        diff_angle = (heading - tang_angle + np.pi) % (2.0 * np.pi) - np.pi
        angle_to_tangent = float(diff_angle / np.pi)

        # 3. Upcoming curvature at 50m, 100m, and 200m
        step_len = self.length / max(1, n)
        k50 = max(1, int(round(50.0 / step_len)))
        k100 = max(2, int(round(100.0 / step_len)))

        idx_50 = (i + k50) % n
        idx_100 = (i + k100) % n

        d50 = (self.tangent_angles[idx_50] - tang_angle + np.pi) % (2.0 * np.pi) - np.pi
        d100 = (self.tangent_angles[idx_100] - tang_angle + np.pi) % (2.0 * np.pi) - np.pi

        curve_50 = float(d50 / np.pi)
        curve_100 = float(d100 / np.pi)

        if lookahead_200:
            k200 = max(4, int(round(200.0 / step_len)))
            idx_200 = (i + k200) % n
            d200 = (self.tangent_angles[idx_200] - tang_angle + np.pi) % (2.0 * np.pi) - np.pi
            curve_200 = float(d200 / np.pi)
            return lat_pos, angle_to_tangent, curve_50, curve_100, curve_200

        return lat_pos, angle_to_tangent, curve_50, curve_100
