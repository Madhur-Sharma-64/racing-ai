"""Real-world-inspired circuits.

Each layout is a hand-made, simplified outline of the real circuit: the same
kind of corners in the same order, with the right proportions, but NOT survey
accurate. Control points are in arbitrary units; a smooth spline is fitted
through them and scaled to (real length x GAME_SCALE).
Control point 0 always sits on the start/finish straight.
"""
import numpy as np

from sim.track import Track

GAME_SCALE = 0.5      # game lap length = real lap length * GAME_SCALE
SPACING = 4.0         # meters between centerline points

CIRCUITS = {
    "Speedway Oval": dict(
        country="USA", km=4.0, width=16.0,
        note="Four fast corners. The easiest track: a good first test for the AI.",
        points=[(0, -4), (6, -4), (9.5, -2.8), (10.5, 0), (9.5, 2.8), (6, 4), (0, 4),
                (-6, 4), (-9.5, 2.8), (-10.5, 0), (-9.5, -2.8), (-6, -4)],
    ),
    "Monza": dict(
        country="Italy", km=5.79, width=13.0,
        note="Long straights, chicanes and a big final corner. All about braking late.",
        points=[(0, 26), (0, 18), (2.0, 15.0), (-0.3, 12.6), (1.8, 10.2),     # pit straight, 1st chicane
                (3.8, 6.8), (7.5, 4.4), (11.5, 3.8),                           # Curva Grande
                (14.0, 3.2), (15.4, 1.8), (17.0, 3.2),                         # 2nd chicane
                (19.0, 5.0), (19.8, 8.0), (19.2, 10.6),                        # the two Lesmo bends
                (16.5, 16.5), (13.5, 22.0),                                    # back straight
                (12.4, 24.3), (13.8, 25.8), (12.0, 27.4),                      # Ascari chicane
                (8.0, 30.8), (3.0, 31.0), (0.0, 28.8)],                        # final big corner
    ),
    "Silverstone": dict(
        country="UK", km=5.89, width=13.0,
        note="Fast sweeping corners and flowing esses, with a few slow bends in between.",
        points=[(0, 22), (7.0, 22.4), (11.0, 21.0), (13.2, 18.6), (12.0, 16.2),   # straight, fast right-left
                (13.4, 13.6), (16.2, 12.2), (17.6, 9.8), (15.8, 7.6), (12.6, 8.0), # tight right-hander, loop
                (10.0, 6.0), (9.2, 2.6), (6.0, 0.8), (2.0, 1.4),                   # flowing section, long straight
                (-1.0, 3.8), (-1.8, 7.0), (-0.2, 9.4), (1.6, 11.8), (0.0, 14.2),   # esses
                (-1.6, 17.0), (-1.4, 20.0)],                                       # last long right-hander
    ),
    "Spa-Francorchamps": dict(
        country="Belgium", km=7.0, width=13.0,
        note="Longest lap: a hairpin, a steep kink, a huge straight and fast sweepers.",
        points=[(0, 30), (-2.5, 31.2), (-5.0, 30.0), (-5.0, 27.4), (-2.6, 26.2),   # La Source hairpin
                (-0.5, 23.6), (1.8, 21.6), (1.2, 19.2), (3.4, 17.4),               # downhill kink
                (6.0, 12.0), (8.5, 6.0), (10.6, 1.4),                              # long straight
                (12.6, 0.6), (14.0, 2.4), (16.0, 3.0),                             # Les Combes chicane
                (17.5, 6.0), (16.0, 9.6), (13.0, 11.0), (12.4, 14.4),              # Malmedy, tight right
                (15.0, 17.6), (18.0, 20.6), (19.0, 24.4),                          # fast double-left
                (17.6, 28.4), (14.0, 30.6), (10.0, 31.4), (6.0, 32.0), (3.0, 31.4)],  # fast left to the bus stop
    ),
    "Interlagos": dict(
        country="Brazil", km=4.31, width=13.0,
        note="Runs anticlockwise. A tight, twisty lap with a steep run back to the start.",
        points=[(0, 18), (0, 12), (-1.4, 9.0), (0.6, 6.6), (3.6, 7.6), (5.0, 4.4),   # pit straight, S bends
                (8.0, 2.4), (12.0, 3.0), (14.6, 5.4),                                 # sweeping left, back straight
                (15.4, 9.0), (13.2, 11.6), (10.4, 11.2), (8.2, 13.4),                 # right-hander, tight infield
                (9.2, 16.4), (12.0, 17.8), (13.0, 20.6), (10.6, 23.0),               # hairpin-style bends
                (6.6, 22.6), (3.8, 20.6)],                                            # climb back to the line
    ),
}


def _spline(points, per_seg=24, alpha=0.5):
    """Closed centripetal Catmull-Rom spline through the points (no overshoot)."""
    P = np.asarray(points, dtype=float)
    n = len(P)
    out = []
    for i in range(n):
        p0, p1, p2, p3 = P[(i - 1) % n], P[i], P[(i + 1) % n], P[(i + 2) % n]
        t0 = 0.0
        t1 = t0 + np.linalg.norm(p1 - p0) ** alpha
        t2 = t1 + np.linalg.norm(p2 - p1) ** alpha
        t3 = t2 + np.linalg.norm(p3 - p2) ** alpha
        t = np.linspace(t1, t2, per_seg, endpoint=False)[:, None]
        a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
        a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
        a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
        b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
        b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
        out.append((t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2)
    return np.vstack(out)


def _closed_length(pts):
    return np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1).sum()


def _resample(pts, spacing):
    """Evenly spaced points along a closed polyline."""
    closed = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    n = max(int(round(cum[-1] / spacing)), 8)
    s = np.linspace(0, cum[-1], n, endpoint=False)
    return np.stack([np.interp(s, cum, closed[:, 0]), np.interp(s, cum, closed[:, 1])], axis=1)


def circuit_names():
    return list(CIRCUITS)


def load_circuit(name, game_scale=GAME_SCALE):
    spec = CIRCUITS[name]
    dense = _spline(spec["points"])
    factor = spec["km"] * 1000.0 * game_scale / _closed_length(dense)
    center = _resample(dense * factor, SPACING)
    track = Track(center, spec["width"], name=name)
    track.country, track.note, track.real_km = spec["country"], spec["note"], spec["km"]
    return track


# ---------------- validation ----------------
def min_corner_radius(track):
    """Tightest centerline corner radius in meters."""
    t = track.tangent
    ang = np.abs(np.arctan2(t[:, 0] * np.roll(t, -1, 0)[:, 1] - t[:, 1] * np.roll(t, -1, 0)[:, 0],
                            (t * np.roll(t, -1, 0)).sum(axis=1)))
    step = np.linalg.norm(np.roll(track.center, -1, 0) - track.center, axis=1)
    return float((step / np.maximum(ang, 1e-9)).min())


def _crossings(segs):
    """Number of crossing pairs among non-neighbouring segments of one polyline set."""
    a, b = segs[:, 0], segs[:, 1]
    m, count = len(segs), 0

    def orient(p, q, r):
        return (q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1]) - (q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0])

    for i in range(m):
        j = np.arange(i + 2, m)
        j = j[~((i == 0) & (j == m - 1))]
        if len(j) == 0:
            continue
        o1, o2 = orient(a[i], b[i], a[j]), orient(a[i], b[i], b[j])
        o3, o4 = orient(a[j], b[j], a[i]), orient(a[j], b[j], b[i])
        count += int(np.sum((o1 * o2 < 0) & (o3 * o4 < 0)))
    return count


def validate(track):
    """Problems found with a track (empty list = fine)."""
    problems = []
    r = min_corner_radius(track)
    if r < track.width / 2 * 1.05:
        problems.append(f"corner radius {r:.1f} m is too tight for width {track.width} m")
    n = len(track.left)
    bad = _crossings(np.stack([track.left, np.roll(track.left, -1, 0)], axis=1)) \
        + _crossings(np.stack([track.right, np.roll(track.right, -1, 0)], axis=1))
    if bad:
        problems.append(f"track edges cross themselves {bad} times")
    # track must not run too close to another part of itself
    k = int(8 * track.width / SPACING)
    d = np.linalg.norm(track.center[:, None] - track.center[None], axis=2)
    idx = np.arange(n)
    gap = np.minimum(np.abs(idx[:, None] - idx[None]), n - np.abs(idx[:, None] - idx[None]))
    d[gap < k] = np.inf
    if d.min() < track.width * 1.5:
        problems.append(f"track passes within {d.min():.1f} m of itself")
    return problems
