"""MADX Universal Multi-Track Championship Trainer.

Trains the MADX AI across ALL training circuits and sketch archetypes
with STRICT exclusion of held-out test data (Suzuka, Monaco, Bahrain,
Albert Park, Red Bull Ring).

Training Pool (18 Core Tracks + Procedural Variations):
  - 9 Championship Tracks: Speedway Oval, Monza, Silverstone, Spa-Francorchamps,
                           Interlagos, Hairpin Haven, Chicane Alley,
                           Technical Karting Ring, Snake Esses.
  - 9 Sketch Archetypes  : The Cross, The Amoeba, Big Mountain, The Great Turn,
                           The Teardrop, Bacteria, Giant Whale, Complex Glyph,
                           Kidney Peanut.
  - Infinite Variations  : On-the-fly procedural tracks every generation.

Held-Out Test Data (NEVER TRAINED ON):
  - Suzuka, Monaco, Bahrain, Albert Park, Red Bull Ring.

Usage:
    py train_madx.py
    py train_madx.py --generations 200 --pop-size 100
    py train_madx.py --seed-model models/MADX.npz
"""
import argparse
import shutil
import time
from pathlib import Path
import numpy as np

from sim.tracks import circuit_names, load_circuit, test_circuit_names
from sim.generator import Archetype, generate_track
from sim.track import Track
from sim.env import MultiCarEnv
from ai.evolution import Population


# MADX architecture: 18 inputs (9 LiDAR rays + 1 Long-Range 200m Radar + 8 Telemetry), 3-layer MLP
MADX_ARCH = (18, 32, 16, 2)


def get_all_training_tracks():
    """Build and validate the full training catalog (Real + Sketch Archetypes).

    Strictly guarantees that NO test circuits are included in training.
    """
    test_circuits = set(test_circuit_names())

    # 1. Real championship circuits
    champ_tracks = {}
    for name in circuit_names():
        if name in test_circuits:
            raise RuntimeError(f"CRITICAL LEAK: Test circuit '{name}' found in training circuits!")
        try:
            champ_tracks[name] = load_circuit(name)
        except Exception as e:
            print(f"[!] Warning: Could not load circuit {name}: {e}")

    # 2. Sketch archetypes (using seed=42 for canonical layout)
    sketch_tracks = {}
    for arch in Archetype:
        try:
            sketch_tracks[arch.value] = generate_track(archetype=arch, seed=42)
        except Exception as e:
            print(f"[!] Warning: Could not generate sketch {arch.value}: {e}")

    return champ_tracks, sketch_tracks


def train_madx(
    generations=200,
    pop_size=100,
    steps_per_track=1100,
    save_path="models/MADX.npz",
    seed_model="models/MADX.npz",
    random_tracks=True,
    single_track=None,
    all_tracks=False,
):
    save_file = Path(save_path)
    save_file.parent.mkdir(parents=True, exist_ok=True)

    # 1. Safety Backup of existing model
    if save_file.exists():
        backup_file = save_file.with_stem(save_file.stem + "_backup")
        shutil.copyfile(save_file, backup_file)
        print(f"[*] Backed up current model to: {backup_file.name}")

    # 2. Load training catalog
    champ_tracks, sketch_tracks = get_all_training_tracks()
    champ_names = list(champ_tracks.keys())
    sketch_names = list(sketch_tracks.keys())

    test_circuits = test_circuit_names()
    print("=" * 120)
    print("  __  __    _    ____  __  __")
    print(" |  \\/  |  / \\  |  _ \\ \\ \\/ /")
    print(" | |\\/| | / _ \\ | | | | \\  / ")
    print(" | |  | |/ ___ \\| |_| | /  \\ ")
    print(" |_|  |_/_/   \\_\\____/ /_/\\_\\")
    print()
    print("  UNIVERSAL MULTI-TRACK CHAMPIONSHIP TRAINER")
    print("=" * 120)
    print(f"  Architecture      : {MADX_ARCH}  (18 inputs: 9 LiDAR + 1 Long Radar 200m + 8 Telemetry)")
    print(f"  Vision System     : 9 rays (100m) + 1 Long-Range Forward Radar (200m) + Curvature 200m")
    print(f"  Fitness Function  : MADX Precision Racing & Apex Momentum")
    print(f"  Evolution         : Dual-Regime (50% Anchor / 50% Pace Hunter)")
    print(f"  Population        : {pop_size} cars")
    print(f"  Generations       : {generations}")
    if single_track:
        print(f"  Training Mode     : DEDICATED SINGLE-TRACK -> '{single_track}'")
    elif all_tracks:
        print(f"  Training Mode     : SIMULTANEOUS ALL 18 TRACKS (9 Championship + 9 Hand-Drawn Sketches)")
        print(f"  Championship (9)  : {', '.join(champ_names)}")
        print(f"  Hand-Drawn (9)    : {', '.join(sketch_names)}")
    else:
        print(f"  Championship (9)  : {', '.join(champ_names)}")
        print(f"  Sketch Tracks (9) : {', '.join(sketch_names)}")
    print(f"  Test Data Excluded: {', '.join(test_circuits)} (ZERO LEAKAGE)")
    print(f"  Target Save       : {save_file.resolve()}")
    print("=" * 120)

    # 3. Initialize population with MADX architecture and dual-regime evolution
    pop = Population(
        size=pop_size,
        layer_sizes=MADX_ARCH,
        elitism=6,
        mutation_rate=0.14,
        mutation_strength=0.25,
        tournament_size=5,
        plateau_window=20,
        plateau_threshold=2.0,
        shock_rate=0.40,
        shock_strength=0.70,
        shock_duration=10,
        immigrant_fraction=0.06,
        nuclear_threshold=50,
        nuclear_survivors=4,
        dual_regime=True,
    )

    # 4. Seed from existing champion
    seed_path = Path(seed_model) if seed_model else None
    if seed_path and seed_path.exists():
        print(f"\n  [*] Seeding from: {seed_path}")
        try:
            pop.load_seed(seed_path, seed_count=35)
            print(f"      [OK] Successfully seeded 35 cars from {seed_path.name}\n")
        except Exception as e:
            print(f"      [!] Seed failed ({e}), starting fresh\n")
    else:
        print(f"\n  [*] No seed found at '{seed_model}', initializing fresh weights\n")

    header = (
        f"{'Gen':>5} | {'Active Tracks (Mini-Batch)':<40} | "
        f"{'Best Score':>10} | {'Avg Score':>9} | {'Best Lap':>9} | {'MutR':>5} | {'Regime':>6} | {'Time':>5}"
    )
    print(header)
    print("-" * 120)

    start_total_time = time.perf_counter()
    all_time_best_lap = None

    for gen in range(1, generations + 1):
        gen_start = time.perf_counter()

        if single_track:
            if single_track in champ_tracks:
                active = [champ_tracks[single_track]]
            elif single_track in sketch_tracks:
                active = [sketch_tracks[single_track]]
            else:
                active = [load_circuit(single_track)]
        elif all_tracks:
            # Full Championship + Hand-Drawn Sketches + Procedural Extreme Hairpins
            active = list(champ_tracks.values()) + list(sketch_tracks.values())
            if random_tracks:
                try:
                    active.append(Track.random_extreme(seed=gen * 71 + 3))
                except Exception:
                    pass
        else:
            # Mini-Batch Track Selection:
            # 1 Real track + 1 Sketch track + 1 Focused Priority Track
            c_idx = (gen - 1) % len(champ_names)
            s_idx = (gen - 1) % len(sketch_names)

            active = [
                champ_tracks[champ_names[c_idx]],
                sketch_tracks[sketch_names[s_idx]],
            ]

            # Prioritize Technical S-curves & Chicanes so the model never regresses on esses
            slot_mod = gen % 4
            if slot_mod == 0 and "Snake Esses" in champ_tracks:
                active.append(champ_tracks["Snake Esses"])
            elif slot_mod == 1 and "Chicane Alley" in champ_tracks:
                active.append(champ_tracks["Chicane Alley"])
            elif slot_mod == 2 and "Monza" in champ_tracks:
                active.append(champ_tracks["Monza"])
            elif random_tracks:
                try:
                    # Alternate between procedural sketches and extreme narrow hairpin tracks
                    if gen % 2 == 0:
                        active.append(Track.random_extreme(seed=gen * 101 + 13))
                    else:
                        active.append(generate_track(seed=gen * 101 + 13))
                except Exception:
                    pass

        total_fitness = np.zeros(pop_size, dtype=float)
        gen_best_lap = None
        active_track_names = []

        for track in active:
            active_track_names.append(track.name[:10])
            track_steps = max(steps_per_track, int(round(track.length / 1.5)))

            env = MultiCarEnv(
                track=track,
                num_cars=pop_size,
                max_steps=track_steps,
                madx_mode=True,
            )
            obs = env.reset()

            for _ in range(track_steps):
                actions = pop.get_actions(obs, env.alive)
                obs, fitness, done, info = env.step(actions)
                if done:
                    break

            total_fitness += env.fitness

            if env.best_lap_time is not None:
                if gen_best_lap is None or env.best_lap_time < gen_best_lap:
                    gen_best_lap = env.best_lap_time

        if gen_best_lap is not None:
            if all_time_best_lap is None or gen_best_lap < all_time_best_lap:
                all_time_best_lap = gen_best_lap

        # Evolve population with Dual-Regime
        stats = pop.evolve(total_fitness)
        gen_time = time.perf_counter() - gen_start

        lap_str = f"{gen_best_lap:6.2f}s" if gen_best_lap else "--:--.--"
        regime_tag = "DUAL" if stats.get("dual_regime") else "STD"
        mut_str = f"{stats.get('mutation_rate', 0):.2f}"
        tracks_str = ",".join(active_track_names)

        print(
            f"{gen:5d} | {tracks_str:<40} | "
            f"{stats['best_fitness']:10.1f} | {stats['avg_fitness']:9.1f} | "
            f"{lap_str:>9} | {mut_str:>5} | {regime_tag:>6} | {gen_time:4.2f}s",
            flush=True,
        )

        # Save checkpoint periodically
        if gen % 20 == 0 or gen == generations:
            pop.save_best(save_file)
        if gen % 100 == 0:
            pop.save_best(save_file.with_stem(f"{save_file.stem}_checkpoint_gen{gen}"))

    total_elapsed = time.perf_counter() - start_total_time
    print("\n" + "=" * 120)
    print("  TRAINING COMPLETE")
    print(f"  Total time       : {total_elapsed:.1f}s ({total_elapsed / generations:.2f}s / gen)")
    print(f"  Saved champion to: {save_file.resolve()}")
    print("=" * 120)


def main():
    parser = argparse.ArgumentParser(
        description="MADX Universal Multi-Track Training Pipeline"
    )
    parser.add_argument(
        "--generations", type=int, default=200,
        help="Total generations (default: 200)",
    )
    parser.add_argument(
        "--pop-size", type=int, default=100,
        help="Population size (default: 100)",
    )
    parser.add_argument(
        "--steps", type=int, default=1100,
        help="Simulation steps per track rollout (default: 1100)",
    )
    parser.add_argument(
        "--save-path", type=str, default="models/ZORO.npz",
        help="Target output path (default: models/ZORO.npz)",
    )
    parser.add_argument(
        "--seed-model", type=str, default="models/ZORO.npz",
        help="Model path to seed from (default: models/ZORO.npz)",
    )
    parser.add_argument(
        "--no-random-tracks", action="store_true",
        help="Disable procedural random tracks",
    )
    parser.add_argument(
        "--track", type=str, default=None,
        help="Train specifically on a single track (e.g. 'Speedway Oval')",
    )
    parser.add_argument(
        "--all-tracks", action="store_true",
        help="Simultaneously evaluate all 18 tracks (9 championship + 9 hand-drawn sketches) every generation",
    )
    args = parser.parse_args()

    train_madx(
        generations=args.generations,
        pop_size=args.pop_size,
        steps_per_track=args.steps,
        save_path=args.save_path,
        seed_model=args.seed_model,
        random_tracks=not args.no_random_tracks,
        single_track=args.track,
        all_tracks=args.all_tracks,
    )


if __name__ == "__main__":
    main()
