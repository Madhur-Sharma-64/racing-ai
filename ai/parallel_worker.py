"""ZORO Turbo Training Worker Module for Multiprocessing."""
import numpy as np
from sim.env import MultiCarEnv
from ai.network import NeuralNetwork


def evaluate_single_track(task_args):
    """Worker function for multiprocessing pool.
    
    task_args: tuple of (track, weights_and_biases, num_cars, track_steps)
    """
    track, net_params, num_cars, track_steps = task_args

    env = MultiCarEnv(
        track=track,
        num_cars=num_cars,
        max_steps=track_steps,
        obs_dim=22,
        madx_mode=True,
    )
    obs = env.reset()

    # Reconstruct networks in child process
    layer_sizes = (22, 32, 16, 2)
    nets = [NeuralNetwork(layer_sizes) for _ in range(num_cars)]
    for i in range(num_cars):
        w_list, b_list = net_params[i]
        nets[i].weights = [w.copy() for w in w_list]
        nets[i].biases = [b.copy() for b in b_list]

    for _ in range(track_steps):
        actions = np.zeros((num_cars, 2), dtype=float)
        for i in range(num_cars):
            if env.alive[i]:
                actions[i] = nets[i].forward(obs[i])
        obs, fitness, done, info = env.step(actions)
        if done:
            break

    return env.fitness, env.best_lap_time
