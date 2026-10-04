import numpy as np
from sim.car import Car


def test_full_gas_reaches_top_speed():
    c = Car([0, 0], 0.0)
    for _ in range(1200):                 # 60 s
        c.update(1, 0)
    assert 45 < c.speed < 60


def test_brake_stops_and_never_reverses():
    c = Car([0, 0], 0.0)
    c.speed = 30
    for _ in range(200):
        c.update(-1, 0)
    assert c.speed == 0.0


def test_straight_line_when_not_steering():
    c = Car([0, 0], 0.0)
    for _ in range(100):
        c.update(1, 0)
    assert abs(c.pos[1]) < 1e-9 and c.pos[0] > 0


def test_positive_steer_turns_right_on_screen():
    c = Car([0, 0], 0.0)
    c.speed = 10
    for _ in range(20):
        c.update(0, 1)
    assert c.heading > 0 and c.pos[1] > 0     # y down = right of heading 0


def test_grip_limits_turn_rate_at_high_speed():
    c = Car([0, 0], 0.0)
    c.speed = 50
    h0 = c.heading
    c.update(0, 1, dt=0.05)
    assert abs(c.heading - h0) <= (Car.MAX_LAT_ACCEL / c.speed) * 0.05 + 1e-9   # uses post-drag speed
