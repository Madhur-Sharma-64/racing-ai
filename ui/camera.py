"""Coordinate transforms from the world (meters, y down) to screen pixels.
No pygame in here, so it can be unit-tested."""
import numpy as np


def follow_view(P, car_pos, heading, scale, cx, cy, heading_up=True):
    """Camera that follows the car and puts it at pixel (cx, cy).

    heading_up=True : the car always points up the screen (the world rotates).
    heading_up=False: north-up, the world keeps its orientation.
    P can be one point (2,) or many (N, 2). scale = pixels per meter.
    """
    rel = np.asarray(P, dtype=float) - np.asarray(car_pos, dtype=float)
    if heading_up:
        fwd = np.array([np.cos(heading), np.sin(heading)])
        side = np.array([-fwd[1], fwd[0]])          # the car's right-hand side
        return np.stack([cx + (rel @ side) * scale, cy - (rel @ fwd) * scale], axis=-1)
    return np.stack([cx + rel[..., 0] * scale, cy + rel[..., 1] * scale], axis=-1)


def fit_view(P, center, scale, cx, cy):
    """Fixed north-up view with world point `center` at pixel (cx, cy)."""
    P = np.asarray(P, dtype=float)
    return np.stack([cx + (P[..., 0] - center[0]) * scale,
                     cy + (P[..., 1] - center[1]) * scale], axis=-1)
