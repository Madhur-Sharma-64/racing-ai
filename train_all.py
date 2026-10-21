"""Multi-Track Championship Trainer with Curriculum Learning.

Trains a single generalized neural network across all 5 real-world inspired circuits
using progressive curriculum learning:

  Stage 1 (gens 1-15%)    : Speedway Oval only          -- learn to drive at all
  Stage 2 (gens 15-30%)   : + Monza                     -- add chicanes & braking
  Stage 3 (gens 30-50%)   : + Silverstone               -- fast sweepers
  Stage 4 (gens 50-70%)   : + Interlagos                -- anticlockwise twisty
  Stage 5 (gens 70-100%)  : + Spa-Francorchamps         -- full championship

Each track is only added once the AI has demonstrated competency on all current
tracks (or after the stage transition generation). This avoids overwhelming early
generations with circuits they cannot yet handle.

Usage:
    python train_all.py
    python train_all.py --generations 800 --pop-size 120
    python train_all.py --seed-model models/best_driver.npz
"""
import argparse
import time
from pathlib import Path
import numpy as np

from sim.tracks import load_circuit
from sim.track import Track
from sim.env import MultiCarEnv
from ai.evolution import Population


# ---------------------------------------------------------------------------
# Curriculum presets: list of (unlock_fraction, track_name) tuples.
# unlock_fraction=0.0 means the track is active from generation 1.
# ---------------------------------------------------------------------------
CURRICULUMS = {
    "extreme": [
        (0.00, "Technical Karting Ring"),
        (0.20, "Hairpin Haven"),
        (0.40, "Chicane Alley"),
        (0.60, "Snake Esses"),
    ],
    "grand_prix": [
        (0.00, "Speedway Oval"),
        (0.15, "Monza"),
        (0.30, "Silverstone"),
        (0.50, "Interlagos"),
        (0.70, "Spa-Francorchamps"),
    ],
    "master": [
        (0.00, "Technical Karting Ring"),
        (0.12, "Hairpin Haven"),
        (0.25, "Monza"),
        (0.37, "Chicane Alley"),
        (0.50, "Silverstone"),
        (0.62, "Snake Esses"),
        (0.75, "Interlagos"),
        (0.87, "Spa-Francorchamps"),
    ],
}


def active_tracks(gen, total_gens, all_tracks_by_name, curriculum_list):
    """Return the subset of tracks that are unlocked at this generation."""
    unlocked = []
    for frac, name in curriculum_list:
        if gen >= frac * total_gens:
            unlocked.append(all_tracks_by_name[name])
    return unlocked


def train_championship(
    curriculum_name="extreme",
    generations=500,
    pop_size=120,
    steps_per_track=1000,
    save_path="models/best_driver.npz",
    seed_model="models/best_driver.npz",
    random_tracks=False,
):
    save_file = Path(save_path)
    save_file.parent.mkdir(parents=True, exist_ok=True)

    curriculum = CURRICULUMS.get(curriculum_name, CURRICULUMS["extreme"])

    # Pre-load all track geometries once
    all_tracks = {name: load_circuit(name) for _, name in curriculum}
    all_track_list = [all_tracks[name] for _, name in curriculum]

    circuit_best_laps   = {name: None for _, name in curriculum}
    circuit_top_speeds  = {name: 0.0  for _, name in curriculum}
    circuit_best_meters = {name: 0.0  for _, name in curriculum}

    # Initialize population — 16 inputs = 9 LiDAR rays + 7 telemetry channels
    pop = Population(size=pop_size, layer_sizes=(16, 32, 16, 2))

    if seed_model and Path(seed_model).exists():
        print(f"[*] Seeding population from: {seed_model}")
        pop.load_seed(seed_model, seed_count=25)

    print("=" * 110)
    print(f"RACING AI - CHAMPIONSHIP TRAINING  (Curriculum: {curriculum_name.upper()})")
    print(f"Population: {pop_size} cars | Generations: {generations} | Steps/track: {steps_per_track}")
    print(f"Architecture: {pop.layer_sizes} Championship Vision | Adaptive mutation | Random immigrants")
    if random_tracks:
        print(f"Domain Randomization: Active (+1 unique procedural extreme track generated every generation)")
    print()
    print("Curriculum schedule:")
    for frac, name in curriculum:
        unlock_gen = max(1, int(frac * generations))
        print(f"  Gen {unlock_gen:4d}+  ->  {name}")
    print("=" * 110)

    header = (
        f"{'Gen':>5} | {'Stage':>5} | {'Best Score':>11} | {'Avg Score':>10} | "
        f"{'Tracks Active':<30} | {'Best Lap':>9} | {'MutR':>5} | {'Time':>6}"
    )
    print(f"\n{header}")
    print("-" * 120)

    start_total_time = time.perf_counter()
    prev_stage = 0

    for gen in range(1, generations + 1):
        gen_start = time.perf_counter()
        tracks = active_tracks(gen, generations, all_tracks, curriculum)
        if random_tracks:
            # Generate a fresh procedural extreme track for this generation
            try:
                rand_track = Track.random_extreme(seed=gen * 17 + 3)
                tracks = list(tracks) + [rand_track]
            except Exception:
                pass
        stage = len(tracks)

        # Print a banner when a new track is unlocked
        if stage > prev_stage and gen > 1:
            new_track = tracks[-1]
            print(f"\n{'':=<120}")
            print(f"  STAGE {stage} UNLOCKED: {new_track.name} ({new_track.real_km:.2f} km) added to championship!")
            print(f"{'':=<120}\n")
            pop.save_best(save_file.with_stem(f"{save_file.stem}_stage{stage - 1}_gen{gen}"))
        prev_stage = stage

        total_fitness = np.zeros(pop_size, dtype=float)
        track_names_active = []
        gen_best_lap = None

        # Evaluate on all currently active tracks
        for track in tracks:
            # Scale steps to track length: longer circuits like Spa (3.5km) need ~2200 steps
            track_steps = max(steps_per_track, int(round(track.length / 1.6)))
            env = MultiCarEnv(track=track, num_cars=pop_size, max_steps=track_steps)
            obs = env.reset()

            for _ in range(track_steps):
                actions = pop.get_actions(obs, env.alive)
                obs, fitness, done, info = env.step(actions)
                if done:
                    break

            total_fitness += env.fitness
            if track.name not in circuit_best_meters:
                circuit_best_meters[track.name] = 0.0
                circuit_best_laps[track.name] = None
                circuit_top_speeds[track.name] = 0.0

            circuit_best_meters[track.name] = max(circuit_best_meters[track.name], float(np.max(env.fitness)))
            track_names_active.append(track.name[:8])  # abbreviate for table

            if env.best_lap_time is not None:
                if gen_best_lap is None or env.best_lap_time < gen_best_lap:
                    gen_best_lap = env.best_lap_time
                if circuit_best_laps[track.name] is None or env.best_lap_time < circuit_best_laps[track.name]:
                    circuit_best_laps[track.name] = env.best_lap_time

            for c in env.cars:
                circuit_top_speeds[track.name] = max(circuit_top_speeds[track.name], c.speed * 3.6)

        # Evolve
        stats = pop.evolve(total_fitness)
        gen_time = time.perf_counter() - gen_start

        lap_str = f"{gen_best_lap:6.2f}s" if gen_best_lap else "--:--.--"
        if stats.get("nuclear_fired"):
            print(f"\n{'':!<120}")
            print(f"  !! NUCLEAR RESET #{stats['nuclear_count']} at gen {gen} !!  Kept top 3, randomized {pop_size - 3} cars.")
            print(f"{'':!<120}\n")

        shock_tag = " [SHOCK]" if stats.get("in_shock") else ""
        nuke_tag  = f" N#{stats['nuclear_count']}" if stats.get("nuclear_count", 0) > 0 else ""
        mut_str = f"{stats.get('mutation_rate', 0):.2f}"
        tracks_str = ",".join(track_names_active)

        print(
            f"{gen:5d} | {stage:5d} | {stats['best_fitness']:11.1f} | {stats['avg_fitness']:10.1f} | "
            f"{tracks_str:<30} | {lap_str:>9} | {mut_str:>5}{shock_tag}{nuke_tag} | {gen_time:5.2f}s"
        )

        # Save every 10 gens, or at the end
        if gen % 10 == 0 or gen == generations:
            pop.save_best(save_file)

    total_elapsed = time.perf_counter() - start_total_time
    print("-" * 120)
    print(f"[+] Training complete in {total_elapsed:.1f}s  ({total_elapsed / generations:.2f}s / gen)")
    print(f"[+] Champion model saved to: {save_file.resolve()}")

    # Final Report Card
    print("\n" + "=" * 90)
    print("FINAL TRAINING REPORT")
    print("-" * 90)
    print(f"{'Circuit':<24} | {'Length':>10} | {'Best Lap':>12} | {'Top Speed':>10} | Progress")
    print("-" * 90)
    for _, name in curriculum:
        track = all_tracks[name]
        lap_t = circuit_best_laps[name]
        lap_str = f"{lap_t:7.3f}s" if lap_t else "  --:--  "
        dist = circuit_best_meters[name]
        pct = min(100.0, (dist / track.length) * 100)
        completed = lap_t is not None
        status = "LAP COMPLETED!" if completed else f"{dist:.0f}m ({pct:.1f}%)"
        print(
            f"{name:<24} | {track.length / 1000:7.2f} km | {lap_str:>12} | "
            f"{circuit_top_speeds[name]:8.0f} km/h | {status}"
        )
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="Multi-track autonomous racing trainer")
    parser.add_argument(
        "--curriculum",
        type=str,
        choices=["extreme", "grand_prix", "master"],
        default="extreme",
        help="Curriculum preset to train on ('extreme', 'grand_prix', or 'master')",
    )
    parser.add_argument("--generations", type=int, default=500, help="Total generations (default 500)")
    parser.add_argument("--pop-size", type=int, default=120, help="Population size")
    parser.add_argument("--steps", type=int, default=1000, help="Sim steps per track")
    parser.add_argument("--save-path", type=str, default="models/best_driver.npz", help="Output model path")
    parser.add_argument(
        "--seed-model",
        type=str,
        default="models/best_driver.npz",
        help="Model path to seed from (default: models/best_driver.npz if present)",
    )
    parser.add_argument(
        "--random-tracks",
        action="store_true",
        help="Generate a fresh procedural extreme track every generation for domain randomization",
    )
    args = parser.parse_args()

    train_championship(
        curriculum_name=args.curriculum,
        generations=args.generations,
        pop_size=args.pop_size,
        steps_per_track=args.steps,
        save_path=args.save_path,
        seed_model=args.seed_model,
        random_tracks=args.random_tracks,
    )


if __name__ == "__main__":
    main()
