import tempfile
import numpy as np

from ai.network import NeuralNetwork
from ai.evolution import Population
from sim.env import MultiCarEnv


def test_neural_network_forward():
    net = NeuralNetwork(layer_sizes=(6, 16, 2))
    inp = np.array([0.5, 0.4, 0.9, 0.2, 0.1, 0.8])
    out = net.forward(inp)
    assert out.shape == (2,)
    assert -1.0 <= out[0] <= 1.0
    assert -1.0 <= out[1] <= 1.0


def test_neural_network_mutate():
    net = NeuralNetwork(layer_sizes=(6, 8, 2))
    clone = net.clone()
    clone.mutate(mutation_rate=1.0, mutation_strength=0.5)
    assert not np.allclose(net.weights[0], clone.weights[0])


def test_neural_network_crossover():
    p1 = NeuralNetwork(layer_sizes=(6, 8, 2))
    p2 = NeuralNetwork(layer_sizes=(6, 8, 2))
    child = NeuralNetwork.crossover(p1, p2)
    assert child.layer_sizes == p1.layer_sizes
    assert child.weights[0].shape == p1.weights[0].shape


def test_neural_network_save_load():
    net = NeuralNetwork(layer_sizes=(6, 12, 2))
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        tmp_path = f.name
    net.save(tmp_path)
    loaded = NeuralNetwork.load(tmp_path)
    test_x = np.random.rand(6)
    assert np.allclose(net.forward(test_x), loaded.forward(test_x))


def test_population_evolution():
    pop = Population(size=10, elitism=2)
    fitnesses = [float(i * 10) for i in range(10)]
    stats = pop.evolve(fitnesses)
    assert stats["generation"] == 1
    assert stats["best_fitness"] == 90.0
    assert pop.generation == 2
    assert len(pop.networks) == 10


def test_load_seed_upgrades_legacy_6dim_to_16dim():
    legacy_net = NeuralNetwork(layer_sizes=(6, 32, 16, 2))
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        tmp_path = f.name
    legacy_net.save(tmp_path)

    pop = Population(size=10, layer_sizes=(16, 32, 16, 2))
    pop.load_seed(tmp_path, seed_count=5)
    assert pop.networks[0].layer_sizes == (16, 32, 16, 2)
    # Check forward pass with 16 inputs works
    dummy_obs = np.ones(16)
    act = pop.networks[0].forward(dummy_obs)
    assert act.shape == (2,)
    assert -1.0 <= act[0] <= 1.0
    assert -1.0 <= act[1] <= 1.0

