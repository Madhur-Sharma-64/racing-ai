"""Evaluation and Benchmarking Suite for Autonomous Racing AI.

Evaluates trained neural network models on held-out test circuits
(never seen during training) to test generalization, zero-shot driving,
lap completion, and racing telemetry.

Usage:
    python eval.py
    python eval.py --model models/best_driver.npz --dataset test
    python eval.py --model models/best_driver.npz --dataset both
    python eval.py --track Suzuka --render
    python eval.py --json reports/eval_results.json
"""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np

from ai.network import NeuralNetwork
from sim.tracks import (
    circuit_names,
    load_circuit,
    test_circuit_names,
    load_test_circuit,
)
from sim.env import RaceEnv
from sim.car import Car


def evaluate_track(net, track, max_steps=None, target_laps=1):
    """Run an evaluation rollout on a single track.

    Returns dict of performance metrics.
    """
    if max_steps is None:
        # Scale steps generously to track length (~60 km/h average expected)
        # track.length / 15m/s ~ seconds; seconds / 0.05s DT = steps * 2.5 safety margin
        max_steps = max(1500, int(round(track.length / 0.75)))

    n_in = int(net.layer_sizes[0])
    env = RaceEnv(track=track, dt=0.05, max_steps=max_steps, obs_dim=n_in)
    obs, info = env.reset()

    speeds = []
    throttles = []
    brakes = []
    steers = []
    max_progress = 0.0
    lap_times = []
    crash = False
    crash_step = None
    crash_progress = None

    steps = 0
    done = False
    start_time = time.perf_counter()

    while not done and steps < max_steps:
        # Forward pass
        action = net.forward(obs)
        throttle = float(action[0])
        steer = float(action[1])

        throttles.append(max(0.0, throttle))
        brakes.append(max(0.0, -throttle))
        steers.append(abs(steer))

        obs, reward, terminated, truncated, step_info = env.step(action)
        steps += 1

        speed_kmh = step_info["speed"] * 3.6
        speeds.append(speed_kmh)

        prog = float(step_info["progress"])
        if prog > max_progress:
            max_progress = prog

        if step_info.get("lap_completed") and step_info.get("best_time") is not None:
            lap_times.append(step_info["best_time"])
            if len(lap_times) >= target_laps:
                done = True
                break

        if terminated and not step_info.get("on_track", True):
            crash = True
            crash_step = steps
            crash_progress = prog
            done = True
            break

        if truncated:
            done = True
            break

    eval_time = time.perf_counter() - start_time
    lap_completed = len(lap_times) >= 1 or step_info.get("laps", 0) >= 1
    best_lap = min(lap_times) if lap_times else step_info.get("best_time")
    progress_pct = min(100.0, (max_progress / track.length) * 100.0) if not lap_completed else 100.0

    return {
        "track_name": track.name,
        "country": getattr(track, "country", "Unknown"),
        "real_km": getattr(track, "real_km", 0.0),
        "game_length_m": float(track.length),
        "lap_completed": bool(lap_completed),
        "best_lap_time": float(best_lap) if best_lap is not None else None,
        "max_progress_m": float(max_progress),
        "progress_pct": float(progress_pct),
        "steps_survived": int(steps),
        "max_steps": int(max_steps),
        "crash": bool(crash),
        "crash_step": crash_step,
        "crash_progress_m": crash_progress,
        "avg_speed_kmh": float(np.mean(speeds)) if speeds else 0.0,
        "top_speed_kmh": float(np.max(speeds)) if speeds else 0.0,
        "avg_throttle": float(np.mean(throttles)) if throttles else 0.0,
        "avg_brake": float(np.mean(brakes)) if brakes else 0.0,
        "avg_steer": float(np.mean(steers)) if steers else 0.0,
        "eval_wall_time_s": float(eval_time),
    }


def render_evaluation(net, track, max_steps=3000):
    """Visually watch the AI model evaluate on a track in pygame."""
    try:
        import pygame
        from sim.raycast import cast_rays
        from ui.camera import follow_view, fit_view
        from ui.render import (
            BG_TERRAIN,
            GREEN,
            RED,
            CYAN,
            draw_track,
            draw_car,
            draw_rays,
            draw_hud,
            draw_minimap,
            fmt_time,
        )
    except ImportError:
        print("[!] Pygame or UI rendering modules not available.")
        return

    pygame.init()
    W, H = 1100, 650
    screen = pygame.display.set_mode((W, H), pygame.RESIZABLE)
    pygame.display.set_caption(f"Racing AI Evaluation - {track.name}")
    font = pygame.font.SysFont("consolas", 18)
    clock = pygame.time.Clock()

    six_dim = bool(net.layer_sizes[0] == 6)
    env = RaceEnv(track=track, dt=0.05, max_steps=max_steps, six_dim=six_dim)
    obs, info = env.reset()

    # Track camera bounds
    pts = np.concatenate([track.left, track.right])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    track_center = (lo + hi) / 2
    track_size = hi - lo

    running = True
    car_cam = True
    zoom = 1.0

    print(f"[*] Watching AI on {track.name}. Controls: C=Camera mode, ESC=Close")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_c:
                    car_cam = not car_cam
            elif event.type == pygame.MOUSEWHEEL:
                zoom = max(0.4, min(3.0, zoom * (1.1 if event.y > 0 else 0.9)))

        action = net.forward(obs)
        obs, reward, terminated, truncated, step_info = env.step(action)

        car = env.car
        rays = cast_rays(car.pos, car.heading, track.segments, env.max_range, midpoints=track.seg_midpoints)

        sw, sh = screen.get_size()
        cam = follow_view(car.pos, car.heading, sw, sh, zoom) if car_cam else fit_view(track_center, track_size, sw, sh)

        screen.fill(BG_TERRAIN)
        draw_track(screen, cam, track)
        draw_rays(screen, cam, car.pos, car.heading, rays)
        draw_car(screen, cam, car, CYAN)

        # HUD
        hud_extra = [
            f"EVAL: {track.name} ({track.real_km:.2f}km)",
            f"Step: {step_info['steps']}/{max_steps}",
            f"Progress: {step_info['progress']:.0f}m / {track.length:.0f}m ({step_info['progress']/track.length*100:.1f}%)",
            f"Lap time: {fmt_time(step_info['lap_time'])}",
            f"Status: {'CRASH' if terminated and not step_info['on_track'] else 'LAP COMPLETE' if step_info.get('lap_completed') else 'RUNNING'}",
        ]
        draw_hud(screen, font, car, step_info["lap_time"], step_info.get("best_time"), step_info.get("laps", 0), extra=hud_extra)
        draw_minimap(screen, track, [car.pos], [CYAN])

        pygame.display.flip()
        clock.tick(60)

        if terminated or truncated:
            # Pause 1.5 seconds at the end before quitting
            pygame.time.delay(1500)
            running = False

    pygame.quit()


def print_report_table(title, results):
    """Print an ASCII telemetry and evaluation table."""
    print()
    print("=" * 115)
    print(f" {title.upper()}")
    print("=" * 115)
    header = (
        f"{'Circuit':<18} | {'Country':<12} | {'Length':>8} | {'Completed':>9} | "
        f"{'Lap Time':>10} | {'Prog %':>7} | {'Avg Km/h':>8} | {'Top Km/h':>8} | {'Status':<14}"
    )
    print(header)
    print("-" * 115)

    for r in results:
        comp_str = "YES" if r["lap_completed"] else "NO"
        lap_str = f"{r['best_lap_time']:.2f}s" if r["best_lap_time"] is not None else "--:--.--"
        prog_str = f"{r['progress_pct']:.1f}%"
        avg_spd = f"{r['avg_speed_kmh']:.1f}"
        top_spd = f"{r['top_speed_kmh']:.1f}"
        len_str = f"{r['game_length_m']:.0f}m"

        if r["lap_completed"]:
            status = "Clean Lap"
        elif r["crash"]:
            status = f"Off @ step {r['crash_step']}"
        else:
            status = "Timeout"

        row = (
            f"{r['track_name']:<18} | {r['country']:<12} | {len_str:>8} | {comp_str:>9} | "
            f"{lap_str:>10} | {prog_str:>7} | {avg_spd:>8} | {top_spd:>8} | {status:<14}"
        )
        print(row)

    print("-" * 115)
    num_tracks = len(results)
    completed_count = sum(1 for r in results if r["lap_completed"])
    comp_pct = (completed_count / num_tracks * 100.0) if num_tracks else 0.0
    avg_prog = np.mean([r["progress_pct"] for r in results]) if results else 0.0
    avg_speed = np.mean([r["avg_speed_kmh"] for r in results]) if results else 0.0

    print(
        f" SUMMARY: {completed_count}/{num_tracks} Completed ({comp_pct:.1f}%) | "
        f"Mean Progress: {avg_prog:.1f}% | Mean Speed: {avg_speed:.1f} km/h"
    )
    print("=" * 115)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Racing AI models across train & held-out test circuits.")
    parser.add_argument(
        "--model",
        type=str,
        default="models/MADX.npz",
        help="Path to .npz model file (default: models/MADX.npz)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["test", "train", "both"],
        default="both",
        help="Which circuit dataset to evaluate on ('test', 'train', or 'both', default: both)",
    )
    parser.add_argument("--track", type=str, default=None, help="Specific track to evaluate on")
    parser.add_argument("--steps", type=int, default=None, help="Override max simulation steps per track")
    parser.add_argument("--render", action="store_true", help="Watch rollout visually using Pygame")
    parser.add_argument("--json", type=str, default=None, help="Save evaluation metrics to JSON file")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[!] Error: Model file not found at: {model_path.resolve()}", file=sys.stderr)
        sys.exit(1)

    print(f"[*] Loading model from: {model_path.resolve()}")
    net = NeuralNetwork.load(model_path)
    print(f"[*] Architecture: {net.layer_sizes}")

    # Determine tracks
    test_names = test_circuit_names()
    train_names = circuit_names()

    if args.track:
        if args.track in test_names:
            tracks_to_eval = [("test", load_test_circuit(args.track))]
        elif args.track in train_names:
            tracks_to_eval = [("train", load_circuit(args.track))]
        else:
            print(f"[!] Unknown track name '{args.track}'. Available:", file=sys.stderr)
            print(f"    Train: {train_names}", file=sys.stderr)
            print(f"    Test:  {test_names}", file=sys.stderr)
            sys.exit(1)
    else:
        tracks_to_eval = []
        if args.dataset in ("train", "both"):
            for n in train_names:
                tracks_to_eval.append(("train", load_circuit(n)))
        if args.dataset in ("test", "both"):
            for n in test_names:
                tracks_to_eval.append(("test", load_test_circuit(n)))

    if args.render:
        # Visual render single track
        _, trk = tracks_to_eval[0]
        render_evaluation(net, trk, max_steps=args.steps or 3000)
        return

    # Headless benchmark
    all_results = {"train": [], "test": []}

    print(f"[*] Running evaluation on {len(tracks_to_eval)} track(s)...")
    for group, track in tracks_to_eval:
        res = evaluate_track(net, track, max_steps=args.steps)
        all_results[group].append(res)

    if all_results["test"]:
        print_report_table("Held-Out Test Circuits (Unseen Generalization)", all_results["test"])

    if all_results["train"]:
        print_report_table("Training Circuits (Championship Dataset)", all_results["train"])

    if args.json:
        out_file = Path(args.json)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n[+] Saved evaluation metrics to: {out_file.resolve()}")


if __name__ == "__main__":
    main()
