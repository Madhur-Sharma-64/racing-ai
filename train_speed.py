"""SPEED DEMON — Multi-Core High-Speed Circuit Training Pipeline.

Philosophy:
  - Aggressive raw velocity, full-throttle commitment, and blistering lap times.
  - Excludes procedural 'extreme' tracks (preserving realistic asphalt dynamics and authentic racing flow).
  - Expanded realistic dataset:
    * 5 Championship Circuits (Speedway Oval, Monza, Silverstone, Spa-Francorchamps, Interlagos)
    * 4 High-Speed Training Grounds (Hairpin Haven, Chicane Alley, Technical Karting, Snake Esses)
    * 5 Realistic High-Speed Sketches (The Great Turn, Big Mountain, The Teardrop, Giant Whale, Kidney Peanut)
    * Smooth Realistic Harmonic Loops (Track.random with real racing widths 13.0m - 16.0m)
  - Multi-Core CPU Parallelism across system cores (~3-5s per generation).

Usage:
    py train_speed.py --generations 200 --pop-size 60
"""
import argparse
import multiprocessing
import random
import shutil
import time
from pathlib import Path
import numpy as np

from sim.tracks import circuit_names, load_circuit, test_circuit_names
from sim.generator import Archetype, generate_track
from sim.track import Track
from sim.env import MultiCarEnv
from ai.evolution import Population
from ai.network import NeuralNetwork

SPEED_DEMON_ARCH = (22, 32, 16, 2)

# Realistic hand-drawn archetypes with authentic flowing racing curves
REALISTIC_SPEED_SKETCHES = [
    Archetype.GREAT_TURN.value,
    Archetype.MOUNTAIN.value,
    Archetype.TEARDROP.value,
    Archetype.WHALE.value,
    Archetype.PEANUT.value,
]


class SpeedDemonEnv(MultiCarEnv):
    """Multi-car environment optimized for pure top speed, momentum carry, and time attack."""

    def __init__(self, track=None, num_cars=60, dt=0.05, max_range=100.0,
                 max_steps=1800, obs_dim=22):
        super().__init__(
            track=track,
            num_cars=num_cars,
            dt=dt,
            max_range=max_range,
            max_steps=max_steps,
            obs_dim=obs_dim,
            madx_mode=False,
        )

    def step(self, actions):
        """Step all alive cars with high-velocity fitness shaping."""
        self.steps += 1
        actions = np.asarray(actions, dtype=float)
        length = self.track.length

        for i, car in enumerate(self.cars):
            if not self.alive[i]:
                continue

            throttle, steer = float(actions[i, 0]), float(actions[i, 1])
            car.update(throttle, steer, self.dt)

            on_track, progress, _ = self.track.query(car.pos)
            if not on_track:
                self.alive[i] = False
                # Penalize crashing so speed is controlled aggression, not reckless suicide
                self.fitness[i] -= 250.0
                continue

            dp = progress - self.prev_progress[i]
            if dp < -length / 2:
                dp += length
            elif dp > length / 2:
                dp -= length
            self.prev_progress[i] = progress

            lap_done = self.laps[i].update(car.pos, self.dt, progress=progress)

            # === SPEED DEMON FITNESS ===
            # 1. Forward progression
            self.fitness[i] += max(0.0, dp)

            # 2. Super-linear velocity multiplier (high speeds >180 km/h heavily rewarded)
            self.fitness[i] += (car.speed / 40.0) ** 1.8 * 0.08

            # 3. 100% Throttle Commitment
            if throttle > 0.85:
                self.fitness[i] += 0.03 * throttle

            # 4. Carrying rolling momentum through curved sections
            _, _, c50, _, _ = self.track.get_track_telemetry(car.pos, car.heading, lookahead_200=True)
            if abs(c50) > 0.12 and car.speed > 25.0:
                self.fitness[i] += (car.speed / 50.0) * 0.06

            # 5. Exponential Time-Attack Lap Completion
            if lap_done:
                self.total_laps += 1
                lap_t = self.laps[i].last_time
                if self.best_lap_time is None or lap_t < self.best_lap_time:
                    self.best_lap_time = lap_t

                # Steeper exponential reward: shaving seconds scores massive gains
                time_bonus = max(0.0, 5000.0 * np.exp(-lap_t / 32.0))
                self.fitness[i] += 2000.0 + time_bonus
                self.alive[i] = False  # Lap finished, retire car

        all_dead = not np.any(self.alive)
        truncated = self.steps >= self.max_steps
        done = all_dead or truncated
        return self.get_observations(), self.fitness.copy(), done, {
            "steps": self.steps,
            "best_lap_time": self.best_lap_time,
            "total_laps": self.total_laps,
        }


def evaluate_single_speed_track(task_args):
    """Worker function for multiprocessing pool."""
    track, net_params, num_cars, track_steps = task_args

    env = SpeedDemonEnv(
        track=track,
        num_cars=num_cars,
        max_steps=track_steps,
        obs_dim=22,
    )
    obs = env.reset()

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

    return env.fitness, env.best_lap_time, env.total_laps


def get_speed_training_catalog():
    """Build expanded real-world inspired training catalog without unphysical extremes."""
    test_circuits = set(test_circuit_names())

    champ_tracks = {}
    for name in circuit_names():
        if name not in test_circuits:
            try:
                champ_tracks[name] = load_circuit(name)
            except Exception:
                pass

    realistic_sketches = {}
    for arch_name in REALISTIC_SPEED_SKETCHES:
        try:
            realistic_sketches[arch_name] = generate_track(archetype=arch_name, seed=42)
        except Exception:
            pass

    return champ_tracks, realistic_sketches


def train_speed_demon(
    generations=200,
    pop_size=60,
    steps_per_track=1200,
    save_path="models/SPEED_DEMON.npz",
    seed_model="models/SPEED_DEMON.npz",
    num_workers=None,
    batch_size=4,
):
    save_file = Path(save_path)
    save_file.parent.mkdir(parents=True, exist_ok=True)

    if save_file.exists():
        backup_file = save_file.with_stem(save_file.stem + "_backup")
        shutil.copyfile(save_file, backup_file)
        print(f"[*] Backed up existing model to: {backup_file.name}")

    champ_tracks, sketch_tracks = get_speed_training_catalog()
    champ_list = list(champ_tracks.values())
    sketch_list = list(sketch_tracks.values())
    test_circuits = test_circuit_names()

    cpu_cores = multiprocessing.cpu_count()
    if num_workers is None:
        workers = min(batch_size, max(2, cpu_cores - 2))
    else:
        workers = max(1, min(num_workers, cpu_cores))

    print("=" * 115)
    print("  SPEED DEMON — EXPANDED REALISTIC HIGH-SPEED PIPELINE (PARALLEL MULTI-CORE)")
    print("=" * 115)
    print(f"  Role              : Pure top-speed aggression, apex momentum, and lowest lap times")
    print(f"  Architecture      : {SPEED_DEMON_ARCH} (22 Universal Inputs)")
    print(f"  Parallel Workers  : {workers} CPU Cores (System total: {cpu_cores})")
    print(f"  Training Set      : 9 Real-World Circuits + 5 Realistic Flowing Sketches + Smooth Harmonic Tracks")
    print(f"  Extreme Filtering : NO extreme procedural spikes (strictly realistic racing track geometry)")
    print(f"  Per-Gen Batch     : 2 Real-World Circuits + 1 Realistic Sketch + 1 Smooth Harmonic Loop")
    print(f"  Population        : {pop_size} cars")
    print(f"  Generations       : {generations}")
    print(f"  Held-Out Tests    : {', '.join(test_circuits)} (ZERO LEAKAGE)")
    print(f"  Target Save       : {save_file.resolve()}")
    print("=" * 115)

    pop = Population(
        size=pop_size,
        layer_sizes=SPEED_DEMON_ARCH,
        elitism=6,
        mutation_rate=0.15,
        mutation_strength=0.28,
        tournament_size=5,
        dual_regime=True,
    )

    seed_path = Path(seed_model) if seed_model else None
    if seed_path and seed_path.exists():
        print(f"\n  [*] Seeding from: {seed_path}")
        try:
            pop.load_seed(seed_path, seed_count=35)
            print(f"      [OK] Seeded 35 cars from {seed_path.name}\n")
        except Exception as e:
            print(f"      [!] Seed failed ({e}), starting fresh\n")

    header = (
        f"{'Gen':>5} | {'Active Realistic Tracks Mini-Batch':<42} | "
        f"{'Best Score':>10} | {'Avg Score':>9} | {'Best Lap':>9} | {'Laps':>5} | {'Time':>6}"
    )
    print(header)
    print("-" * 115)

    start_total_time = time.perf_counter()

    with multiprocessing.Pool(processes=workers) as pool:
        for gen in range(1, generations + 1):
            gen_start = time.perf_counter()

            # Dynamic realistic mini-batch of 4 tracks (no unphysical extremes):
            # 1. Two Real-world inspired circuits (Monza, Silverstone, Spa, Interlagos, Speedway, etc.)
            sampled_champ = random.sample(champ_list, 2)
            # 2. One Realistic Sketch with authentic racing arcs (Great Turn, Mountain, Teardrop, Whale, Peanut)
            sampled_sketch = [random.choice(sketch_list)]
            # 3. One smooth realistic harmonic track (Track.random with wide 13.0m - 15.0m racing width)
            r_seed = gen * 1000 + 89
            r_width = float(np.random.uniform(13.0, 15.5))
            sampled_random = [Track.random(seed=r_seed, width=r_width)]

            active = sampled_champ + sampled_sketch + sampled_random
            active_names = [t.name[:9] for t in active]

            net_params = [(net.weights, net.biases) for net in pop.networks]

            tasks = []
            for track in active:
                track_steps = max(steps_per_track, int(round(track.length / 1.5)))
                tasks.append((track, net_params, pop_size, track_steps))

            results = pool.map(evaluate_single_speed_track, tasks)

            total_fitness = np.zeros(pop_size, dtype=float)
            gen_best_lap = None
            gen_total_laps = 0

            for track_fitness, best_lap, laps in results:
                total_fitness += track_fitness
                gen_total_laps += laps
                if best_lap is not None:
                    if gen_best_lap is None or best_lap < gen_best_lap:
                        gen_best_lap = best_lap

            stats = pop.evolve(total_fitness)
            gen_time = time.perf_counter() - gen_start

            lap_str = f"{gen_best_lap:6.2f}s" if gen_best_lap else "--:--.--"
            tracks_str = ",".join(active_names)

            print(
                f"{gen:5d} | {tracks_str:<42} | "
                f"{stats['best_fitness']:10.1f} | {stats['avg_fitness']:9.1f} | "
                f"{lap_str:>9} | {gen_total_laps:>5} | {gen_time:5.2f}s",
                flush=True,
            )

            if gen % 10 == 0 or gen == generations:
                pop.save_best(save_file)
            if gen % 50 == 0:
                pop.save_best(save_file.with_stem(f"SPEED_DEMON_checkpoint_gen{gen}"))

    total_elapsed = time.perf_counter() - start_total_time
    print("\n" + "=" * 115)
    print("  SPEED DEMON REALISTIC TRAINING COMPLETE")
    print(f"  Total time       : {total_elapsed:.1f}s ({total_elapsed / generations:.2f}s / gen)")
    print(f"  Saved champion to: {save_file.resolve()}")
    print("=" * 115)


def main():
    parser = argparse.ArgumentParser(description="SPEED DEMON Expanded Realistic Multi-Core Trainer")
    parser.add_argument("--generations", type=int, default=200, help="Total generations (default: 200)")
    parser.add_argument("--pop-size", type=int, default=60, help="Population size (default: 60)")
    parser.add_argument("--steps", type=int, default=1200, help="Simulation steps per track rollout")
    parser.add_argument("--save-path", type=str, default="models/SPEED_DEMON.npz", help="Output model path")
    parser.add_argument("--seed-model", type=str, default="models/SPEED_DEMON.npz", help="Seed model path")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel worker processes")
    args = parser.parse_args()

    train_speed_demon(
        generations=args.generations,
        pop_size=args.pop_size,
        steps_per_track=args.steps,
        save_path=args.save_path,
        seed_model=args.seed_model,
        num_workers=args.workers,
    )


if __name__ == '__main__':
    main()
