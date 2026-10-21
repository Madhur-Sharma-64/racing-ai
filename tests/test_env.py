import numpy as np

from sim.env import RaceEnv, MultiCarEnv
from sim.tracks import load_circuit


def test_race_env_reset():
    env = RaceEnv()
    obs, info = env.reset()
    assert obs.shape == (env.obs_dim,)
    assert 0.0 <= obs[env.obs_dim - 7] <= 1.0  # normalized speed
    assert np.all(obs[:env.obs_dim - 7] >= 0.0)
    assert -1.5 <= obs[env.obs_dim - 6] <= 1.5  # lateral position
    assert info["laps"] == 0
    assert info["speed"] == 0.0


def test_race_env_step_forward():
    env = RaceEnv()
    env.reset()
    # Step full gas straight
    obs, reward, terminated, truncated, info = env.step([1.0, 0.0])
    assert obs.shape == (env.obs_dim,)
    assert not terminated
    assert not truncated
    assert info["speed"] > 0.0
    assert info["on_track"]


def test_race_env_off_track_terminates():
    env = RaceEnv()
    env.reset()
    # Put car far off track
    env.car.pos = np.array([9999.0, 9999.0])
    obs, reward, terminated, truncated, info = env.step([0.0, 0.0])
    assert terminated
    assert not info["on_track"]
    assert reward < 0.0  # penalty for off-track


def test_multi_car_env_step():
    num_cars = 20
    menv = MultiCarEnv(num_cars=num_cars)
    obs = menv.reset()
    assert obs.shape == (num_cars, menv.obs_dim)

    actions = np.zeros((num_cars, 2))
    actions[:, 0] = 1.0  # full gas
    next_obs, fitness, done, info = menv.step(actions)
    assert next_obs.shape == (num_cars, menv.obs_dim)
    assert len(fitness) == num_cars
    assert not done
