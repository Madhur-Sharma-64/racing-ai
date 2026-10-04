import numpy as np


def raycast(pos, angle, segments, max_range=100.0):
    """Distance from pos along `angle` to the nearest segment (or max_range)."""
    d = np.array([np.cos(angle), np.sin(angle)])
    a = segments[:, 0]                      # (M, 2)
    e = segments[:, 1] - a                  # (M, 2) segment direction
    ap = a - pos

    denom = d[0] * e[:, 1] - d[1] * e[:, 0]                 # cross(d, e)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (ap[:, 0] * e[:, 1] - ap[:, 1] * e[:, 0]) / denom   # distance along ray
        s = (ap[:, 0] * d[1] - ap[:, 1] * d[0]) / denom         # position on segment

    hit = (np.abs(denom) > 1e-12) & (t > 0) & (s >= 0) & (s <= 1)
    return float(min(t[hit].min(), max_range)) if hit.any() else float(max_range)


RAY_ANGLES = np.deg2rad([-60, -30, 0, 30, 60])


def cast_rays(pos, heading, segments, max_range=100.0):
    return np.array([raycast(pos, heading + a, segments, max_range) for a in RAY_ANGLES])
