import pytest
from sim.tracks import circuit_names, load_circuit, validate, CIRCUITS, GAME_SCALE


@pytest.mark.parametrize("name", circuit_names())
def test_circuit_is_valid(name):
    assert validate(load_circuit(name)) == []


@pytest.mark.parametrize("name", circuit_names())
def test_circuit_length_matches_scaled_real_length(name):
    t = load_circuit(name)
    expected = CIRCUITS[name]["km"] * 1000 * GAME_SCALE
    assert abs(t.length - expected) / expected < 0.01


@pytest.mark.parametrize("name", circuit_names())
def test_start_is_on_the_track_and_progress_is_zero(name):
    t = load_circuit(name)
    assert t.contains(t.center[0]) and t.progress(t.center[0]) == 0
import pytest
from sim.tracks import circuit_names, load_circuit, validate, CIRCUITS, GAME_SCALE


@pytest.mark.parametrize("name", circuit_names())
def test_circuit_is_valid(name):
    assert validate(load_circuit(name)) == []


@pytest.mark.parametrize("name", circuit_names())
def test_circuit_length_matches_scaled_real_length(name):
    t = load_circuit(name)
    expected = CIRCUITS[name]["km"] * 1000 * GAME_SCALE
    assert abs(t.length - expected) / expected < 0.01


@pytest.mark.parametrize("name", circuit_names())
def test_start_is_on_the_track_and_progress_is_zero(name):
    t = load_circuit(name)
    assert t.contains(t.center[0]) and t.progress(t.center[0]) == 0
