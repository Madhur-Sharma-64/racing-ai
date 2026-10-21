"""KEEPER (Safe AI) — Universal Track Completion Multi-Core Curriculum Trainer.

Goal:
  100% completion on ANY circuit with zero crashes.
  Trained on a massive variety of real circuits, all hand-crafted sketch archetypes
  (The Cross, Amoeba, Great Turn, Mountain, Teardrop, Bacteria, Whale, Glyph, Peanut),
  and fresh procedural Random Extremes with variable road widths (7.5m - 16.0m).

Pipeline Highlights:
  1. Multi-Core CPU Parallelism: Runs track evaluations across CPU cores simultaneously.
  2. Diverse Training Batches (4 tracks / gen):
     - 1 Championship Track (Speedway, Monza, Silverstone, Spa, Interlagos, Hairpins)
     - 2 Hand-Crafted Sketch Archetypes (The Cross, Amoeba, Bacteria, Great Turn, etc.)
     - 1 Procedural Extreme Circuit (fresh random geometry + randomized road width every gen)
  3. Survival-First Fitness:
     - High rewards for progress and staying centered in lane.
     - Anti-stall protection (kills parked/stalling cars).
     - Steady cruising speed reward.
     - Flat lap completion bonus.

Usage:
    py train_safe.py --generations 200 --pop-size 60
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

KEEPER_ARCH = (22, 32, 16, 2)


class KeeperEnv(MultiCarEnv):
    """Multi-car environment tailored for survival, track-centering, and collision avoidance."""

    def __init__(self, track=None, num_cars=60, dt=0.05, max_range=100.0,
                 max_steps=2200, obs_dim=22):
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
        """Step all alive cars with survival-first fitness."""
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
                continue

            # Anti-stall: if car hasn't moved after 100 steps (~5s), kill it
            if self.steps > 100 and car.speed < 1.0:
                self.alive[i] = False
                continue

            dp = progress - self.prev_progress[i]
            if dp < -length / 2:
                dp += length
            elif dp > length / 2:
                dp -= length
            self.prev_progress[i] = progress

            lap_done = self.laps[i].update(car.pos, self.dt, progress=progress)

            # === KEEPER SURVIVAL & PROGRESSION FITNESS ===
            # 1. Forward progression (primary driver)
            self.fitness[i] += max(0.0, dp)

            # 2. Continuous Speed Bonus: steady reward for carrying speed (smoothly rewards moving)
            self.fitness[i] += (car.speed / 30.0) * 0.06

            # 3. Strong center-lane alignment reward (stay safely within track boundaries)
            lat_pos, _, _, _, _ = self.track.get_track_telemetry(car.pos, car.heading, lookahead_200=True)
            self.fitness[i] += max(0.0, 1.0 - abs(lat_pos)) * 0.035

            # 4. Smooth steering reward (discourage violent twitches/spins)
            self.fitness[i] += max(0.0, 1.0 - abs(steer)) * 0.015

            # 5. Lap Completion: Flat survival reward + moderate bonus for faster lap times
            if lap_done:
                self.total_laps += 1
                lap_t = self.laps[i].last_time
                if self.best_lap_time is None or lap_t < self.best_lap_time:
                    self.best_lap_time = lap_t
                
                time_bonus = max(0.0, 1500.0 * np.exp(-lap_t / 50.0))
                self.fitness[i] += 3500.0 + time_bonus
                self.alive[i] = False  # Completed clean lap, retire

        all_dead = not np.any(self.alive)
        truncated = self.steps >= self.max_steps
        done = all_dead or truncated
        return self.get_observations(), self.fitness.copy(), done, {
            "steps": self.steps,
            "best_lap_time": self.best_lap_time,
            "total_laps": self.total_laps,
        }


def evaluate_single_keeper_track(task_args):
    """Worker function for multiprocessing pool."""
    track, net_params, num_cars, track_steps = task_args

    env = KeeperEnv(
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


def get_training_catalog():
    """Build full training catalog: Championship circuits + all 9 hand sketches."""
    test_circuits = set(test_circuit_names())

    champ_tracks = {}
    for name in circuit_names():
        if name not in test_circuits:
            try:
                champ_tracks[name] = load_circuit(name)
            except Exception:
                pass

    sketch_tracks = {}
    for arch in Archetype:
        try:
            sketch_tracks[arch.value] = generate_track(archetype=arch.value, seed=42)
        except Exception:
            pass

    return champ_tracks, sketch_tracks


def train_keeper(
    generations=200,
    pop_size=60,
    steps_per_track=1400,
    save_path="models/KEEPER.npz",
    seed_model="models/KEEPER.npz",
    num_workers=None,
    batch_size=4,
):
    save_file = Path(save_path)
    save_file.parent.mkdir(parents=True, exist_ok=True)

    if save_file.exists():
        backup_file = save_file.with_stem(save_file.stem + "_backup")
        shutil.copyfile(save_file, backup_file)
        print(f"[*] Backed up existing model to: {backup_file.name}")

    champ_tracks, sketch_tracks = get_training_catalog()
    champ_list = list(champ_tracks.values())
    sketch_list = list(sketch_tracks.values())
    test_circuits = test_circuit_names()

    cpu_cores = multiprocessing.cpu_count()
    if num_workers is None:
        workers = min(batch_size, max(2, cpu_cores - 2))
    else:
        workers = max(1, min(num_workers, cpu_cores))

    print("=" * 115)
    print("  KEEPER — UNIVERSAL TRACK COMPLETION TRAINER (PARALLEL MULTI-CORE)")
    print("=" * 115)
    print(f"  Role              : Flawless track completion & collision avoidance on any circuit")
    print(f"  Architecture      : {KEEPER_ARCH} (22 Universal Inputs)")
    print(f"  Parallel Workers  : {workers} CPU Cores (System total: {cpu_cores})")
    print(f"  Training Slate    : 9 Championship + 9 Hand-Drawn Sketches + Procedural Extremes")
    print(f"  Per-Gen Batch     : 1 Championship + 2 Hand Sketches + 1 Procedural Extreme (Widths: 7.5m - 16.0m)")
    print(f"  Population        : {pop_size} cars")
    print(f"  Generations       : {generations}")
    print(f"  Held-Out Tests    : {', '.join(test_circuits)} (ZERO LEAKAGE)")
    print(f"  Target Save       : {save_file.resolve()}")
    print("=" * 115)

    pop = Population(
        size=pop_size,
        layer_sizes=KEEPER_ARCH,
        elitism=8,
        mutation_rate=0.12,
        mutation_strength=0.20,
        tournament_size=4,
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
        f"{'Gen':>5} | {'Active Diverse Tracks Mini-Batch':<42} | "
        f"{'Best Score':>10} | {'Avg Score':>9} | {'Best Lap':>9} | {'Laps':>5} | {'Time':>6}"
    )
    print(header)
    print("-" * 115)

    start_total_time = time.perf_counter()

    with multiprocessing.Pool(processes=workers) as pool:
        for gen in range(1, generations + 1):
            gen_start = time.perf_counter()

            # Dynamic diverse mini-batch of 4 varied circuits:
            # 1. One Championship circuit
            sampled_champ = [random.choice(champ_list)]
            # 2. Two distinct Hand-Drawn Sketches (The Cross, Amoeba, Bacteria, Whale, etc.)
            sampled_sketches = random.sample(sketch_list, 2)
            # 3. One fresh procedural extreme circuit with randomized road width (narrow to wide)
            r_width = float(np.random.uniform(7.5, 16.0))
            r_seed = gen * 1000 + 47
            sampled_random = [Track.random_extreme(seed=r_seed, width=r_width)]

            active = sampled_champ + sampled_sketches + sampled_random
            active_names = [t.name[:9] for t in active]

            net_params = [(net.weights, net.biases) for net in pop.networks]

            tasks = []
            for track in active:
                track_steps = max(steps_per_track, int(round(track.length / 0.9)))
                tasks.append((track, net_params, pop_size, track_steps))

            results = pool.map(evaluate_single_keeper_track, tasks)

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
                pop.save_best(save_file.with_stem(f"KEEPER_checkpoint_gen{gen}"))

    total_elapsed = time.perf_counter() - start_total_time
    print("\n" + "=" * 115)
    print("  KEEPER UNIVERSAL TRAINING COMPLETE")
    print(f"  Total time       : {total_elapsed:.1f}s ({total_elapsed / generations:.2f}s / gen)")
    print(f"  Saved champion to: {save_file.resolve()}")
    print("=" * 115)


def main():
    parser = argparse.ArgumentParser(description="KEEPER Universal Multi-Core Track Completion Trainer")
    parser.add_argument("--generations", type=int, default=200, help="Total generations (default: 200)")
    parser.add_argument("--pop-size", type=int, default=60, help="Population size (default: 60)")
    parser.add_argument("--steps", type=int, default=1400, help="Simulation steps per track rollout")
    parser.add_argument("--save-path", type=str, default="models/KEEPER.npz", help="Output model path")
    parser.add_argument("--seed-model", type=str, default="models/KEEPER.npz", help="Seed model path")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel worker processes")
    args = parser.parse_args()

    train_keeper(
        generations=args.generations,
        pop_size=args.pop_size,
        steps_per_track=args.steps,
        save_path=args.save_path,
        seed_model=args.seed_model,
        num_workers=args.workers,
    )


if __name__ == '__main__':
    main()
