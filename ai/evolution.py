"""Genetic Algorithm / Neuroevolution engine for vehicle populations.

Manages populations of NeuralNetwork individuals, evaluates fitness across
generations, and applies elitism, tournament selection, crossover, and mutation.

Key features:
  - Adaptive mutation: spikes mutation when fitness plateaus to escape local optima
  - Random immigrants: injects fresh random individuals each generation for diversity
  - Elitism: preserves top N individuals untouched
  - Nuclear reset: if shocks fail for too long, wipe all but top 3 and start fresh
  - Dual-regime (MADX): splits population 50/50 into safe anchors and aggressive pace hunters
"""
from pathlib import Path
import numpy as np

from ai.network import NeuralNetwork


class Population:
    """Manages a population of neural networks evolving to master racetracks."""

    def __init__(
        self,
        size=80,
        layer_sizes=(16, 32, 16, 2),
        elitism=6,
        mutation_rate=0.12,
        mutation_strength=0.25,
        tournament_size=5,
        # --- Adaptive mutation ---
        plateau_window=10,       # generations to look back for plateau detection
        plateau_threshold=1.0,   # min improvement to not be considered a plateau
        shock_rate=0.40,         # mutation_rate during a shock
        shock_strength=0.70,     # mutation_strength during a shock
        shock_duration=8,        # how many generations the shock lasts
        # --- Random immigrants ---
        immigrant_fraction=0.10, # fraction of population replaced by random individuals
        # --- Nuclear reset ---
        nuclear_threshold=30,    # consecutive shock-gens with no improvement before nuclear reset
        nuclear_survivors=3,     # how many top elites survive a nuclear reset
        # --- Dual regime (MADX) ---
        dual_regime=False,       # if True, split population 50/50 into anchor + pace hunter
    ):
        self.size = size
        self.layer_sizes = layer_sizes
        self.elitism = min(elitism, size // 2)
        self.base_mutation_rate = mutation_rate
        self.base_mutation_strength = mutation_strength
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.tournament_size = tournament_size

        # Adaptive mutation params
        self.plateau_window = plateau_window
        self.plateau_threshold = plateau_threshold
        self.shock_rate = shock_rate
        self.shock_strength = shock_strength
        self.shock_duration = shock_duration
        self._shock_remaining = 0
        self._stale_shock_gens = 0   # counts consecutive gens in shock with no improvement
        self._last_best_at_shock = -np.inf

        # Nuclear reset params
        self.nuclear_threshold = nuclear_threshold
        self.nuclear_survivors = nuclear_survivors
        self._nuclear_count = 0      # total nuclear resets fired

        # Random immigrants
        self.immigrant_fraction = immigrant_fraction
        self.n_immigrants = max(1, int(size * immigrant_fraction))

        # Dual-regime (MADX mode)
        self.dual_regime = dual_regime

        self.networks = [NeuralNetwork(layer_sizes) for _ in range(size)]
        self.generation = 1
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.best_ever_network = None
        self.best_ever_fitness = -np.inf

    def get_actions(self, observations, alive_mask=None):
        """Compute actions for the whole population in one vectorized batch pass.

        Uses NeuralNetwork.batch_forward() to run all N networks simultaneously
        via stacked matrix ops — no Python loop over cars.
        """
        obs = np.asarray(observations, dtype=np.float64)

        if alive_mask is not None and not np.all(alive_mask):
            # Only run inference for alive cars; fill dead cars with zeros
            alive_idx = np.where(alive_mask)[0]
            alive_nets = [self.networks[i] for i in alive_idx]
            alive_obs = obs[alive_idx]
            alive_actions = NeuralNetwork.batch_forward(alive_nets, alive_obs)
            actions = np.zeros((self.size, 2), dtype=float)
            actions[alive_idx] = alive_actions
        else:
            actions = NeuralNetwork.batch_forward(self.networks, obs)

        return actions

    def _select_parent(self, ranked_indices, tournament_size):
        """Tournament selection: pick the fittest from random competitors."""
        candidates = np.random.choice(ranked_indices, size=tournament_size, replace=False)
        return min(candidates)  # lower rank index = higher fitness

    def _check_plateau(self):
        """Return True if best fitness hasn't improved meaningfully in plateau_window gens."""
        if len(self.best_fitness_history) < self.plateau_window:
            return False
        recent = self.best_fitness_history[-self.plateau_window:]
        return (max(recent) - min(recent)) < self.plateau_threshold

    def _update_adaptive_mutation(self, current_best, ranked):
        """Spike mutation if plateauing; nuclear-reset if shocks keep failing.

        Nuclear reset logic:
          - Counts how many consecutive generations we've been in shock with no
            improvement vs self._last_best_at_shock.
          - Once that counter hits nuclear_threshold, we wipe the population:
            keep only the top `nuclear_survivors` elites, replace everyone else
            with fresh random networks.
        """
        in_shock = self._shock_remaining > 0
        nuclear_fired = False

        if in_shock:
            self.mutation_rate = self.shock_rate
            self.mutation_strength = self.shock_strength
            self._shock_remaining -= 1

            # Track stagnation *while* in shock
            if current_best > self._last_best_at_shock + self.plateau_threshold:
                # Shock is working — reset stale counter
                self._last_best_at_shock = current_best
                self._stale_shock_gens = 0
            else:
                self._stale_shock_gens += 1

            # Nuclear reset if we've been stuck in shock-loops for too long
            if self._stale_shock_gens >= self.nuclear_threshold:
                self._nuclear_reset(ranked)
                self._stale_shock_gens = 0
                self._shock_remaining = 0
                self._last_best_at_shock = current_best
                self._nuclear_count += 1
                nuclear_fired = True

        elif self._check_plateau():
            # Plateau detected — fire a mutation shock
            self.mutation_rate = self.shock_rate
            self.mutation_strength = self.shock_strength
            self._shock_remaining = self.shock_duration
            self._last_best_at_shock = current_best
        else:
            # Normal operation: return to base rates
            self.mutation_rate = self.base_mutation_rate
            self.mutation_strength = self.base_mutation_strength
            self._stale_shock_gens = 0

        return nuclear_fired

    def _nuclear_reset(self, ranked):
        """Keep top `nuclear_survivors` elites; replace everyone else with fresh randoms."""
        survivors = [self.networks[ranked[i]].clone() for i in range(self.nuclear_survivors)]
        fresh = [NeuralNetwork(self.layer_sizes) for _ in range(self.size - self.nuclear_survivors)]
        self.networks = survivors + fresh

    def _breed_standard(self, ranked, new_networks):
        """Standard breeding: crossover + uniform mutation for all remaining slots."""
        while len(new_networks) < self.size:
            idx_a = self._select_parent(ranked, self.tournament_size)
            idx_b = self._select_parent(ranked, self.tournament_size)

            parent_a = self.networks[idx_a]
            parent_b = self.networks[idx_b]

            # 70% chance of crossover, 30% clone from fitter parent
            if np.random.rand() < 0.70:
                child = NeuralNetwork.crossover(parent_a, parent_b)
            else:
                child = parent_a.clone()

            # Apply adaptive mutation
            child.mutate(
                mutation_rate=self.mutation_rate,
                mutation_strength=self.mutation_strength,
            )
            new_networks.append(child)
        return new_networks

    def _breed_dual_regime(self, ranked, new_networks):
        """Dual-regime breeding: 50% Anchor (conservative) + 50% Pace Hunter (aggressive).

        Anchor regime:
          - Low mutation rate (base * 0.5), low strength (base * 0.6)
          - Bred from the top-50% of performers
          - Purpose: preserve reliable lap-completing behavior

        Pace Hunter regime:
          - High mutation rate (base * 2.5), high strength (base * 2.0)
          - Bred from top-30% (favor the absolute fastest)
          - Purpose: explore risky high-speed strategies aggressively
        """
        remaining = self.size - len(new_networks)
        n_anchor = remaining // 2
        n_pace = remaining - n_anchor

        # Anchor regime: conservative mutations from broad top-half gene pool
        anchor_mr = self.mutation_rate * 0.5
        anchor_ms = self.mutation_strength * 0.6
        top_half = ranked[:max(4, self.size // 2)]

        for _ in range(n_anchor):
            idx_a = self._select_parent(top_half, min(self.tournament_size, len(top_half)))
            idx_b = self._select_parent(top_half, min(self.tournament_size, len(top_half)))
            parent_a = self.networks[idx_a]
            parent_b = self.networks[idx_b]

            if np.random.rand() < 0.75:
                child = NeuralNetwork.crossover(parent_a, parent_b)
            else:
                child = parent_a.clone()
            child.mutate(anchor_mr, anchor_ms)
            new_networks.append(child)

        # Pace Hunter regime: aggressive mutations from elite top-30%
        pace_mr = self.mutation_rate * 2.5
        pace_ms = self.mutation_strength * 2.0
        top_elite = ranked[:max(4, int(self.size * 0.3))]

        for _ in range(n_pace):
            idx_a = self._select_parent(top_elite, min(self.tournament_size, len(top_elite)))
            idx_b = self._select_parent(top_elite, min(self.tournament_size, len(top_elite)))
            parent_a = self.networks[idx_a]
            parent_b = self.networks[idx_b]

            if np.random.rand() < 0.60:
                child = NeuralNetwork.crossover(parent_a, parent_b)
            else:
                child = parent_a.clone()
            child.mutate(pace_mr, pace_ms)
            new_networks.append(child)

        return new_networks

    def evolve(self, fitnesses):
        """Produce the next generation using elitism, selection, crossover, and mutation.

        Parameters:
            fitnesses: array-like of float, shape (self.size,)

        Returns:
            stats: dict of generation metrics
        """
        fitnesses = np.asarray(fitnesses, dtype=float)
        ranked = np.argsort(fitnesses)[::-1]  # best to worst indices

        best_fit = float(fitnesses[ranked[0]])
        avg_fit = float(np.mean(fitnesses))
        self.best_fitness_history.append(best_fit)
        self.avg_fitness_history.append(avg_fit)

        # Track all-time best individual
        if best_fit > self.best_ever_fitness:
            self.best_ever_fitness = best_fit
            self.best_ever_network = self.networks[ranked[0]].clone()

        # Update adaptive mutation rates BEFORE breeding (may trigger nuclear reset)
        nuclear_fired = self._update_adaptive_mutation(best_fit, ranked)
        in_shock = self._shock_remaining > 0 or self.mutation_rate == self.shock_rate

        # Build next generation
        new_networks = []

        # 1. Elitism: preserve top individuals untouched
        for i in range(self.elitism):
            new_networks.append(self.networks[ranked[i]].clone())

        # 2. Random immigrants for diversity
        for _ in range(self.n_immigrants):
            new_networks.append(NeuralNetwork(self.layer_sizes))

        # 3. Breed remaining slots — dual regime or standard
        if self.dual_regime:
            new_networks = self._breed_dual_regime(ranked, new_networks)
        else:
            new_networks = self._breed_standard(ranked, new_networks)

        self.networks = new_networks
        self.generation += 1

        stats = {
            "generation": self.generation - 1,
            "best_fitness": best_fit,
            "avg_fitness": avg_fit,
            "best_ever_fitness": self.best_ever_fitness,
            "mutation_rate": self.mutation_rate,
            "in_shock": in_shock,
            "nuclear_fired": nuclear_fired,
            "nuclear_count": self._nuclear_count,
            "dual_regime": self.dual_regime,
        }
        return stats

    def save_best(self, filepath):
        """Save the all-time best network to disk."""
        if self.best_ever_network is not None:
            self.best_ever_network.save(filepath)

    def load_seed(self, filepath, seed_count=10):
        """Seed the population with clones of a pre-trained network.

        Supports seamless architecture migration:
          - Same architecture: direct clone
          - Different architecture: transfer-learn compatible layers, initialize new neurons
        """
        loaded = NeuralNetwork.load(filepath)
        if loaded.layer_sizes != self.layer_sizes:
            # Architecture migration: transfer what we can, init the rest
            src_sizes = loaded.layer_sizes
            dst_sizes = self.layer_sizes
            upgraded = NeuralNetwork(dst_sizes)

            if src_sizes[0] == 16 and dst_sizes[0] == 22 and len(src_sizes) == len(dst_sizes):
                # 16-dim MADX -> 22-dim Universal
                # 16-dim: [0..8: 9 rays, 9: speed, 10: lat, 11: heading, 12: c50, 13: c100, 14: yaw, 15: latG]
                # 22-dim: [0..8: 9 rays, 9: radar, 10: speed, 11: lat, 12: heading, 13: c50, 14: c100, 15: c200, 16: yaw, 17: latG, 18..21: invariants]
                w0 = np.zeros((22, dst_sizes[1]), dtype=float)
                w0[0:9] = loaded.weights[0][0:9]       # 9 rays
                w0[9] = 0.0                            # 200m radar (zero initial bias)
                w0[10:15] = loaded.weights[0][9:14]    # speed, lat, heading, c50, c100
                w0[15] = 0.0                           # 200m curve lookahead (zero bias)
                w0[16:18] = loaded.weights[0][14:16]   # yaw, latG
                w0[18:22] = 0.0                        # 4 universal invariants (zero bias)
                upgraded.weights[0] = w0
                upgraded.biases[0] = loaded.biases[0].copy()
                for k in range(1, len(loaded.weights)):
                    copy_in = min(src_sizes[k], dst_sizes[k])
                    copy_out = min(src_sizes[k + 1], dst_sizes[k + 1])
                    upgraded.weights[k][:copy_in, :copy_out] = loaded.weights[k][:copy_in, :copy_out]
                    upgraded.biases[k][:copy_out] = loaded.biases[k][:copy_out]
            elif src_sizes[0] == 18 and dst_sizes[0] == 22:
                # 18-dim ZORO v1 -> 22-dim ZORO v2 Universal
                # Preserves all 18 trained weights directly, initializes 4 new universal invariants with 0.0
                w0 = np.zeros((22, dst_sizes[1]), dtype=float)
                w0[0:18] = loaded.weights[0][0:18]
                w0[18:22] = 0.0  # Zero bias for track_width, margin_left, margin_right, d_curvature
                upgraded.weights[0] = w0
                upgraded.biases[0] = loaded.biases[0].copy()
                for k in range(1, len(loaded.weights)):
                    copy_in = min(src_sizes[k], dst_sizes[k])
                    copy_out = min(src_sizes[k + 1], dst_sizes[k + 1])
                    upgraded.weights[k][:copy_in, :copy_out] = loaded.weights[k][:copy_in, :copy_out]
                    upgraded.biases[k][:copy_out] = loaded.biases[k][:copy_out]
            elif src_sizes[0] == 16 and dst_sizes[0] == 18 and len(src_sizes) == len(dst_sizes):
                # 16-dim: [9 rays (100m), speed, lat, heading, c50, c100, yaw, latG]
                # 18-dim: [9 rays (100m), 200m radar, speed, lat, heading, c50, c100, c200, yaw, latG]
                w0 = np.zeros((18, dst_sizes[1]), dtype=float)
                # 1. Map 9 standard 100m rays directly (inputs 0..8)
                w0[0:9] = loaded.weights[0][0:9]
                # 2. Initialize 200m forward radar with 0.0 (zero initial bias for perfect baseline preservation)
                w0[9] = 0.0
                # 3. Map speed, lat, heading, c50, c100 (inputs 10..14)
                w0[10] = loaded.weights[0][9]   # speed
                w0[11] = loaded.weights[0][10]  # lat_pos
                w0[12] = loaded.weights[0][11]  # angle_rel
                w0[13] = loaded.weights[0][12]  # curve_50
                w0[14] = loaded.weights[0][13]  # curve_100
                # 4. Initialize curve_200 lookahead with 0.0 (zero initial bias)
                w0[15] = 0.0
                # 5. Map yaw and lat_accel
                w0[16] = loaded.weights[0][14]  # norm_yaw
                w0[17] = loaded.weights[0][15]  # norm_lat
                upgraded.weights[0] = w0
                upgraded.biases[0] = loaded.biases[0].copy()
                for k in range(1, len(loaded.weights)):
                    copy_in = min(src_sizes[k], dst_sizes[k])
                    copy_out = min(src_sizes[k + 1], dst_sizes[k + 1])
                    upgraded.weights[k][:copy_in, :copy_out] = loaded.weights[k][:copy_in, :copy_out]
                    upgraded.biases[k][:copy_out] = loaded.biases[k][:copy_out]
            elif src_sizes[0] == 12 and dst_sizes[0] == 16 and len(src_sizes) == len(dst_sizes):
                # 12-dim: [5 rays (-60,-30,0,30,60), 7 telemetry]
                # 16-dim: [9 rays (-90,-60,-45,-30,0,30,45,60,90), 7 telemetry]
                ray_map = {1: 0, 3: 1, 4: 2, 5: 3, 7: 4}
                w0 = np.zeros((16, dst_sizes[1]), dtype=float)
                for dst_r, src_r in ray_map.items():
                    w0[dst_r] = loaded.weights[0][src_r]
                # Map 7 telemetry channels directly to inputs 9..15
                for i in range(7):
                    w0[9 + i] = loaded.weights[0][5 + i]
                upgraded.weights[0] = w0
                upgraded.biases[0] = loaded.biases[0].copy()
                for k in range(1, len(loaded.weights)):
                    copy_in = min(src_sizes[k], dst_sizes[k])
                    copy_out = min(src_sizes[k + 1], dst_sizes[k + 1])
                    upgraded.weights[k][:copy_in, :copy_out] = loaded.weights[k][:copy_in, :copy_out]
                    upgraded.biases[k][:copy_out] = loaded.biases[k][:copy_out]
            elif src_sizes[0] == 6 and dst_sizes[0] == 16 and len(src_sizes) == len(dst_sizes):
                # 6-dim: [5 rays, speed]
                ray_map = {1: 0, 3: 1, 4: 2, 5: 3, 7: 4}
                w0 = np.zeros((16, dst_sizes[1]), dtype=float)
                for dst_r, src_r in ray_map.items():
                    w0[dst_r] = loaded.weights[0][src_r]
                w0[9] = loaded.weights[0][5]  # speed
                upgraded.weights[0] = w0
                upgraded.biases[0] = loaded.biases[0].copy()
                for k in range(1, len(loaded.weights)):
                    copy_in = min(src_sizes[k], dst_sizes[k])
                    copy_out = min(src_sizes[k + 1], dst_sizes[k + 1])
                    upgraded.weights[k][:copy_in, :copy_out] = loaded.weights[k][:copy_in, :copy_out]
                    upgraded.biases[k][:copy_out] = loaded.biases[k][:copy_out]
            else:
                n_layers = min(len(src_sizes) - 1, len(dst_sizes) - 1)
                for k in range(n_layers):
                    src_in = src_sizes[k]
                    src_out = src_sizes[k + 1]
                    dst_in = dst_sizes[k]
                    dst_out = dst_sizes[k + 1]

                    copy_in = min(src_in, dst_in)
                    copy_out = min(src_out, dst_out)
                    upgraded.weights[k][:copy_in, :copy_out] = loaded.weights[k][:copy_in, :copy_out]
                    upgraded.biases[k][:copy_out] = loaded.biases[k][:copy_out]

                    if dst_in > src_in:
                        upgraded.weights[k][src_in:, :copy_out] = (
                            np.random.randn(dst_in - src_in, copy_out) * 0.05
                        )
                    if dst_out > src_out:
                        upgraded.weights[k][:copy_in, src_out:] = (
                            np.random.randn(copy_in, dst_out - src_out) * 0.05
                        )

            loaded = upgraded

        for i in range(min(seed_count, self.size)):
            clone = loaded.clone()
            if i > 0:
                clone.mutate(self.mutation_rate, self.mutation_strength)
            self.networks[i] = clone

