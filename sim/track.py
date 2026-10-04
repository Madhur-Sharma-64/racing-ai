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

        self.left = self.center + normal * width / 2
        self.right = self.center - normal * width / 2

        # all boundary segments as (M, 2, 2): [[ax, ay], [bx, by]]
        self.segments = np.concatenate([self._segs(self.left), self._segs(self.right)])

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

    # ---- queries ----
    def nearest_index(self, pos):
        return int(np.argmin(np.sum((self.center - pos) ** 2, axis=1)))

    def contains(self, pos):
        """True if pos is on the track (within half-width of the centerline)."""
        i = self.nearest_index(pos)
        return np.linalg.norm(self.center[i] - pos) <= self.width / 2

    def progress(self, pos):
        """Distance along the centerline, in meters."""
        return self.cum_len[self.nearest_index(pos)]
