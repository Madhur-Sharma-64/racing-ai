from sim.tracks import (
    circuit_names, load_circuit, validate, CIRCUITS, GAME_SCALE,
    test_circuit_names as get_test_circuit_names, load_test_circuit, TEST_CIRCUITS,
)


def test_circuits_are_valid():
    for name in circuit_names():
        assert validate(load_circuit(name)) == []


def test_circuit_lengths_match_scaled_real_length():
    for name in circuit_names():
        t = load_circuit(name)
        expected = CIRCUITS[name]["km"] * 1000 * GAME_SCALE
        assert abs(t.length - expected) / expected < 0.01


def test_start_is_on_the_track_and_progress_is_zero():
    for name in circuit_names():
        t = load_circuit(name)
        assert t.contains(t.center[0]) and t.progress(t.center[0]) == 0


def test_test_circuits_are_valid():
    for name in get_test_circuit_names():
        assert validate(load_test_circuit(name)) == []


def test_test_circuit_lengths_match_scaled_real_length():
    for name in get_test_circuit_names():
        t = load_test_circuit(name)
        expected = TEST_CIRCUITS[name]["km"] * 1000 * GAME_SCALE
        assert abs(t.length - expected) / expected < 0.01


def test_track_telemetry_lookahead_and_lateral_offset():
    track = load_circuit("Monza")
    # Position car exactly on centerline
    lat, ang, c50, c100 = track.get_track_telemetry(track.center[10], track.tangent_angles[10])
    assert abs(lat) < 1e-4  # on centerline
    assert abs(ang) < 1e-4  # aligned with tangent
    assert -1.0 <= c50 <= 1.0
    assert -1.0 <= c100 <= 1.0

    # Position car on left boundary
    lat_l, _, _, _ = track.get_track_telemetry(track.left[10], track.tangent_angles[10])
    assert 0.8 < lat_l <= 1.2  # near +1.0 (left kerb)

    # Position car on right boundary
    lat_r, _, _, _ = track.get_track_telemetry(track.right[10], track.tangent_angles[10])
    assert -1.2 <= lat_r < -0.8  # near -1.0 (right kerb)


def test_random_extreme_generates_valid_tracks():
    from sim.track import Track
    for seed in (42, 101, 777):
        t = Track.random_extreme(seed=seed)
        assert validate(t) == []
        assert t.length > 500
        assert t.contains(t.center[0])
