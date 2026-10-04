import numpy as np
from sim.track import Track
from sim.lap import LapTimer


def drive_indices(timer, track, indices, dt=1.0):
    done = 0
    for i in indices:
        done += timer.update(track.center[i], dt)
    return done


def test_full_forward_lap_counts_once():
    tr = Track.random(seed=1)
    t = LapTimer(tr)
    n = len(tr.center)
    assert drive_indices(t, tr, list(range(1, n)) + [0]) == 1
    assert t.laps == 1 and t.last_time == n and t.best_time == n


def test_driving_backwards_never_counts():
    tr = Track.random(seed=1)
    t = LapTimer(tr)
    n = len(tr.center)
    drive_indices(t, tr, list(range(n - 1, -1, -1)) * 2)
    assert t.laps == 0


def test_half_lap_and_return_does_not_count():
    tr = Track.random(seed=1)
    t = LapTimer(tr)
    n = len(tr.center)
    drive_indices(t, tr, list(range(1, n // 2)) + list(range(n // 2, 0, -1)))
    assert t.laps == 0


def test_restart_lap_keeps_laps_and_best():
    tr = Track.random(seed=1)
    t = LapTimer(tr)
    n = len(tr.center)
    drive_indices(t, tr, list(range(1, n)) + [0])
    drive_indices(t, tr, range(1, 40))
    t.restart_lap(tr.center[0])
    assert t.laps == 1 and t.best_time == n
    assert t.lap_time == 0.0 and t.distance == 0.0


def test_best_time_is_the_minimum():
    tr = Track.random(seed=1)
    t = LapTimer(tr)
    n = len(tr.center)
    drive_indices(t, tr, list(range(1, n)) + [0], dt=2.0)   # slow lap
    drive_indices(t, tr, list(range(1, n)) + [0], dt=1.0)   # fast lap
    assert t.laps == 2 and t.best_time == n and t.last_time == n
