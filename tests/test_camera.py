import numpy as np
from ui.camera import follow_view, fit_view


def test_car_is_drawn_at_the_anchor_point():
    for heading in (0.0, 1.0, -2.5, 3.0):
        p = follow_view([10.0, -4.0], [10.0, -4.0], heading, 4.0, 500, 400)
        assert np.allclose(p, [500, 400])


def test_heading_up_puts_ahead_above_and_right_to_the_right():
    for heading in np.linspace(-3, 3, 9):
        car = np.array([5.0, 7.0])
        fwd = np.array([np.cos(heading), np.sin(heading)])
        right = np.array([-fwd[1], fwd[0]])
        ahead = follow_view(car + 20 * fwd, car, heading, 4.0, 500, 400)
        side = follow_view(car + 10 * right, car, heading, 4.0, 500, 400)
        assert np.allclose(ahead, [500, 400 - 80])       # straight up, 20 m * 4 px
        assert np.allclose(side, [500 + 40, 400])        # straight right, 10 m * 4 px


def test_north_up_ignores_heading():
    a = follow_view([3.0, 2.0], [0.0, 0.0], 0.0, 2.0, 100, 100, heading_up=False)
    b = follow_view([3.0, 2.0], [0.0, 0.0], 2.0, 2.0, 100, 100, heading_up=False)
    assert np.allclose(a, [106, 104]) and np.allclose(a, b)


def test_arrays_of_points_keep_their_shape():
    pts = np.random.rand(50, 2) * 100
    assert follow_view(pts, [0, 0], 0.7, 3.0, 0, 0).shape == (50, 2)
    assert fit_view(pts, [50, 50], 2.0, 10, 10).shape == (50, 2)


def test_fit_view_centers_the_given_point():
    assert np.allclose(fit_view([7.0, 9.0], [7.0, 9.0], 3.0, 200, 150), [200, 150])
