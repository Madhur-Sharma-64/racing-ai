"""Feedforward Neural Network for autonomous vehicle control.

Implemented in pure NumPy with support for forward inference, weight mutation,
crossover, and saving/loading model weights.

Architecture: (6, 32, 16, 2)
  - Hidden layers use LeakyReLU (avoids dead neurons, better gradient flow)
  - Output layer uses tanh (bounds throttle/steer to [-1, 1])
"""
from pathlib import Path
import numpy as np


class NeuralNetwork:
    """Multi-layer perceptron mapping sensor observations to vehicle actions.

    Inputs (6):
        - 5 normalized raycast distances [0, 1]
        - 1 normalized vehicle speed [0, 1]

    Outputs (2):
        - throttle: [-1, 1] (via tanh)
        - steer:    [-1, 1] (via tanh)

    Hidden layers use LeakyReLU activation (alpha=0.01) to prevent dead
    neurons and improve gradient flow over deep architectures.
    """

    LEAKY_ALPHA = 0.01  # slope for x < 0 in LeakyReLU

    def __init__(self, layer_sizes=(16, 32, 16, 2)):
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []

        # He / Xavier normal initialization
        for i in range(len(layer_sizes) - 1):
            n_in, n_out = layer_sizes[i], layer_sizes[i + 1]
            scale = np.sqrt(2.0 / (n_in + n_out))
            w = np.random.randn(n_in, n_out) * scale
            b = np.zeros(n_out)
            self.weights.append(w.astype(np.float64))
            self.biases.append(b.astype(np.float64))

    @staticmethod
    def _leaky_relu(z, alpha=0.01):
        """Leaky ReLU: f(z) = z if z > 0, else alpha*z."""
        return np.where(z > 0, z, alpha * z)

    def forward(self, x):
        """Forward pass for a single input vector (shape: (6,)) or batch (shape: (N, 6)).

        Hidden layers → LeakyReLU.
        Output layer  → tanh (bounds to [-1, 1]).
        """
        a = np.asarray(x, dtype=np.float64)
        num_hidden = len(self.weights) - 1

        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ w + b
            if i < num_hidden:
                a = self._leaky_relu(z, self.LEAKY_ALPHA)
            else:
                a = np.tanh(z)  # output layer: bound to [-1, 1]

        return a

    @classmethod
    def batch_forward(cls, networks, obs_matrix):
        """Vectorized forward pass for a list of N networks given an (N, 6) obs matrix.

        Stacks all weight matrices into 3-D tensors and processes the entire
        population in one shot using NumPy broadcasting — no Python loop over cars.

        Parameters:
            networks:   list of NeuralNetwork, length N
            obs_matrix: np.ndarray shape (N, input_dim)

        Returns:
            actions: np.ndarray shape (N, output_dim)
        """
        if not networks:
            return np.empty((0, 2))

        layer_sizes = networks[0].layer_sizes
        n = len(networks)
        num_hidden = len(layer_sizes) - 2  # number of hidden layers

        # Stack weights/biases: shapes (N, in, out) and (N, out)
        W = [np.stack([net.weights[i] for net in networks], axis=0)
             for i in range(len(layer_sizes) - 1)]
        B = [np.stack([net.biases[i] for net in networks], axis=0)
             for i in range(len(layer_sizes) - 1)]

        # a shape: (N, features)
        a = np.asarray(obs_matrix, dtype=np.float64)

        for i, (w, b) in enumerate(zip(W, B)):
            # w: (N, in, out), a: (N, in) → z: (N, out)
            z = np.einsum('ni,nio->no', a, w) + b
            if i < num_hidden:
                a = np.where(z > 0, z, cls.LEAKY_ALPHA * z)  # LeakyReLU
            else:
                a = np.tanh(z)  # output layer

        return a  # (N, output_dim)

    def mutate(self, mutation_rate=0.1, mutation_strength=0.2):
        """Mutate weights and biases in-place with Gaussian noise."""
        for i in range(len(self.weights)):
            # Mutate weights
            mask_w = np.random.rand(*self.weights[i].shape) < mutation_rate
            noise_w = np.random.randn(*self.weights[i].shape) * mutation_strength
            self.weights[i] += mask_w * noise_w

            # Mutate biases
            mask_b = np.random.rand(*self.biases[i].shape) < mutation_rate
            noise_b = np.random.randn(*self.biases[i].shape) * mutation_strength
            self.biases[i] += mask_b * noise_b

    def clone(self):
        """Create a deep copy of the neural network."""
        net = NeuralNetwork(self.layer_sizes)
        net.weights = [w.copy() for w in self.weights]
        net.biases = [b.copy() for b in self.biases]
        return net

    @classmethod
    def crossover(cls, parent_a, parent_b):
        """Create a child by uniform crossover between two parent networks."""
        child = parent_a.clone()
        for i in range(len(child.weights)):
            mask_w = np.random.rand(*child.weights[i].shape) > 0.5
            child.weights[i] = np.where(mask_w, parent_a.weights[i], parent_b.weights[i])

            mask_b = np.random.rand(*child.biases[i].shape) > 0.5
            child.biases[i] = np.where(mask_b, parent_a.biases[i], parent_b.biases[i])
        return child

    def save(self, filepath):
        """Save network architecture and weights to a compressed .npz archive."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {f"w_{i}": w for i, w in enumerate(self.weights)}
        data.update({f"b_{i}": b for i, b in enumerate(self.biases)})
        data["layer_sizes"] = np.array(self.layer_sizes)
        np.savez_compressed(path, **data)

    @classmethod
    def load(cls, filepath):
        """Load a network from a saved .npz archive."""
        archive = np.load(filepath)
        layer_sizes = tuple(archive["layer_sizes"])
        net = cls(layer_sizes)
        net.weights = [archive[f"w_{i}"] for i in range(len(layer_sizes) - 1)]
        net.biases = [archive[f"b_{i}"] for i in range(len(layer_sizes) - 1)]
        return net
