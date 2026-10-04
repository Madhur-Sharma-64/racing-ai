import numpy as np
from sim.raycast import raycast, cast_rays
from sim.track import Track


def test_ray_hits_known_wall():
    wall = np.array([[[10.0, -5.0], [10.0, 5.0]]])        # vertical wall at x = 10
    assert abs(raycast(np.array([0.0, 0.0]), 0.0, wall) - 10.0) < 1e-9


def test_ray_pointing_away_misses():
    wall = np.array([[[10.0, -5.0], [10.0, 5.0]]])
    assert raycast(np.array([0.0, 0.0]), np.pi, wall, max_range=100) == 100


def test_ray_capped_at_max_range():
    wall = np.array([[[500.0, -5.0], [500.0, 5.0]]])
    assert raycast(np.array([0.0, 0.0]), 0.0, wall, max_range=100) == 100


def test_track_is_closed_and_car_on_it():
    tr = Track.random(seed=1)
    assert tr.contains(tr.center[0])
    assert not tr.contains(tr.center[0] + 100)


def test_rays_on_centerline_are_bounded_by_track_width():
    tr = Track.random(seed=1)
    i = 50
    heading = np.arctan2(*tr.tangent[i][::-1])
    d = cast_rays(tr.center[i], heading, tr.segments)
    assert d[0] < 15 and d[4] < 15          # side rays hit the nearby walls
