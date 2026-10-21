import numpy as np


def raycast(pos, angle, segments, max_range=100.0):
    """Distance from pos along `angle` to the nearest segment (or max_range)."""
    if len(segments) == 0:
        return float(max_range)
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


RAY_ANGLES = np.deg2rad([-90, -60, -45, -30, 0, 30, 45, 60, 90])


def cast_rays(pos, heading, segments, max_range=100.0, midpoints=None, return_radar=False, radar_range=200.0):
    """Cast 9 sensor rays simultaneously with segment culling and vectorized intersections.

    If `return_radar=True`, also evaluates the forward ray (0 deg) out to `radar_range` (default 200m)
    in a single vectorized pass and returns `(rays, radar_200)`.
    """
    effective_max = max(max_range, radar_range) if return_radar else max_range
    if len(segments) == 0:
        empty_rays = np.full(len(RAY_ANGLES), float(max_range))
        return (empty_rays, float(radar_range)) if return_radar else empty_rays

    pos = np.asarray(pos, dtype=float)
    if midpoints is None:
        midpoints = (segments[:, 0] + segments[:, 1]) * 0.5

    # Spatial culling: only keep segments within (effective_max + buffer) of pos
    dx = midpoints[:, 0] - pos[0]
    dy = midpoints[:, 1] - pos[1]
    cutoff = (effective_max + 5.0) ** 2
    mask = (dx * dx + dy * dy) < cutoff
    nearby = segments[mask]

    if len(nearby) == 0:
        empty_rays = np.full(len(RAY_ANGLES), float(max_range))
        return (empty_rays, float(radar_range)) if return_radar else empty_rays

    a = nearby[:, 0]
    e = nearby[:, 1] - a
    ap = a - pos
    # t_num is independent of ray direction: (M,)
    t_num = ap[:, 0] * e[:, 1] - ap[:, 1] * e[:, 0]

    # Evaluate all 9 ray directions simultaneously: (K, 2)
    angles = heading + RAY_ANGLES
    d = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    denom = d[:, 0:1] * e[:, 1] - d[:, 1:2] * e[:, 0]       # (K, M)
    s_num = ap[:, 0] * d[:, 1:2] - ap[:, 1] * d[:, 0:1]     # (K, M)

    with np.errstate(divide="ignore", invalid="ignore"):
        t = t_num / denom                                   # (K, M)
        s = s_num / denom                                   # (K, M)

    hit = (np.abs(denom) > 1e-12) & (t > 0) & (s >= 0) & (s <= 1)
    t_hit = np.where(hit, t, np.inf)
    min_t = np.min(t_hit, axis=1)

    rays = np.clip(min_t, 0.0, max_range).astype(float)

    if return_radar:
        # Index 4 is the forward 0-degree ray
        radar_val = float(np.clip(min_t[4], 0.0, radar_range))
        return rays, radar_val

    return rays


def cast_radar_200m(pos, heading, segments, max_range=200.0, midpoints=None):
    """Cast a single dedicated long-range forward radar ray up to 200m."""
    return raycast(pos, heading, segments, max_range=max_range)
