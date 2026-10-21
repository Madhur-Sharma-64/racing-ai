"""Interactive AI Training Visualizer using Neuroevolution / Genetic Algorithm.

Watch a swarm of cars learn to navigate race tracks using 5-ray sensor inputs
and pure NumPy neural networks.

Controls:
  SPACE     Toggle Fast / Visual training mode
  N         Next racetrack
  S         Save best model to models/best_driver.npz
  L         Load seed model from models/best_driver.npz
  R         Reset training
  C         Toggle camera (Car-Up / North-Up)
  + / -     Zoom in / out (or mouse wheel)
  ESC       Quit
"""
from pathlib import Path
import numpy as np
import pygame

from sim.env import MultiCarEnv
from sim.tracks import circuit_names, load_circuit
from sim.track import Track
from sim.raycast import cast_rays
from ai.evolution import Population
from ui.camera import follow_view, fit_view
from ui.render import (
    BG_TERRAIN,
    GREEN,
    GREEN_DARK,
    ASPHALT,
    RED,
    CYAN,
    YELLOW,
    WHITE,
    MUTED_GREY,
    ORANGE,
    draw_track,
    draw_car,
    draw_cars,
    draw_rays,
    draw_minimap,
    fmt_time,
    _draw_rounded_rect,
)

W, H = 1200, 700
DT = 0.05
CHOICES = circuit_names() + ["Random"]
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "best_driver.npz"


def build_track(choice, seed=1):
    return Track.random(seed) if choice == "Random" else load_circuit(choice)


def fit_track(track):
    pts = np.concatenate([track.left, track.right])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    return (lo + hi) / 2, (hi - lo)


def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H), pygame.RESIZABLE)
    pygame.display.set_caption("Racing AI - Autonomous Swarm Training")
    font = pygame.font.SysFont("consolas", 16)
    bold_font = pygame.font.SysFont("consolas", 18, bold=True)
    big_font = pygame.font.SysFont("consolas", 36, bold=True)
    clock = pygame.time.Clock()

    pop_size = 60
    track_idx = CHOICES.index("Speedway Oval")  # start on easier oval or Monza
    seed = 1
    track = build_track(CHOICES[track_idx], seed)
    center, size = fit_track(track)

    env = MultiCarEnv(track=track, num_cars=pop_size, dt=DT, max_steps=max(2500, int(track.length / 1.6)))
    pop = Population(size=pop_size, elitism=4, mutation_rate=0.12, mutation_strength=0.25)

    obs = env.reset()
    fast_mode = False
    multi_track_mode = False
    heading_up = True
    zoom = 4.0
    status_msg = "Press SPACE for Turbo Mode | M for Multi-Track"
    status_timer = 3.0
    all_time_best_lap = None
    actions = np.zeros((pop_size, 2), dtype=float)

    def restart_track(new_idx):
        nonlocal track_idx, seed, track, center, size, env, obs
        track_idx = new_idx % len(CHOICES)
        if CHOICES[track_idx] == "Random":
            seed += 1
        track = build_track(CHOICES[track_idx], seed)
        center, size = fit_track(track)
        env = MultiCarEnv(track=track, num_cars=pop_size, dt=DT, max_steps=max(2500, int(track.length / 1.6)))
        obs = env.reset()

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.MOUSEWHEEL:
                zoom = float(np.clip(zoom * 1.1 ** e.y, 1.2, 10.0))
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_SPACE:
                    fast_mode = not fast_mode
                    status_msg = "FAST TURBO MODE (20,000+ steps/s)" if fast_mode else "VISUAL MODE (60 FPS)"
                    status_timer = 2.0
                elif e.key == pygame.K_m:
                    multi_track_mode = not multi_track_mode
                    status_msg = f"Multi-Track Auto-Rotation: {'ON' if multi_track_mode else 'OFF'}"
                    status_timer = 2.0
                elif e.key == pygame.K_c:
                    heading_up = not heading_up
                elif e.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    zoom = min(10.0, zoom * 1.15)
                elif e.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    zoom = max(1.2, zoom / 1.15)
                elif e.key == pygame.K_n:
                    restart_track(track_idx + 1)
                    status_msg = f"Track: {track.name}"
                    status_timer = 2.0
                elif e.key == pygame.K_s:
                    pop.save_best(MODEL_PATH)
                    status_msg = f"Saved best model to {MODEL_PATH}"
                    status_timer = 3.0
                elif e.key == pygame.K_l:
                    if MODEL_PATH.exists():
                        pop.load_seed(MODEL_PATH)
                        obs = env.reset()
                        status_msg = f"Loaded seed model from {MODEL_PATH}"
                    else:
                        status_msg = f"No model file found at {MODEL_PATH}"
                    status_timer = 3.0
                elif e.key == pygame.K_r:
                    pop = Population(size=pop_size)
                    obs = env.reset()
                    status_msg = "Reset population to Generation 1"
                    status_timer = 2.0
                elif e.key == pygame.K_ESCAPE:
                    running = False

        # If fast mode, run multiple physics steps per frame
        steps_per_frame = 40 if fast_mode else 1
        for _ in range(steps_per_frame):
            actions = pop.get_actions(obs, env.alive)
            obs, fitness, done, info = env.step(actions)

            if done:
                # Evolve generation
                stats = pop.evolve(fitness)
                if multi_track_mode:
                    restart_track(track_idx + 1)
                    status_msg = f"GEN {stats['generation']} Complete | Next: {track.name}"
                else:
                    obs = env.reset()
                    status_msg = f"GEN {stats['generation']} Complete | Best: {stats['best_fitness']:.1f} m"
                status_timer = 2.0
                break

        # Rendering — vibrant Grand Prix turf and asphalt
        screen.fill(BG_TERRAIN)
        w, h = screen.get_size()

        # Find leader among alive cars (or best fitness if all dead)
        alive_indices = np.nonzero(env.alive)[0]
        if len(alive_indices) > 0:
            leader_idx = alive_indices[np.argmax(env.fitness[alive_indices])]
        else:
            leader_idx = int(np.argmax(env.fitness))

        # Leader car and lap stats
        leader = env.cars[leader_idx]
        leader_lap = env.laps[leader_idx]
        speed_kmh = leader.speed * 3.6

        # Check for new best lap
        if env.best_lap_time is not None:
            if all_time_best_lap is None or env.best_lap_time < all_time_best_lap:
                all_time_best_lap = env.best_lap_time
                status_msg = f"NEW RECORD LAP: {fmt_time(all_time_best_lap)}!"
                status_timer = 4.0

        # Camera view function
        cx, cy = w / 2, (h * 0.68 if heading_up else h / 2)
        def view(P):
            return follow_view(P, leader.pos, leader.heading, zoom, cx, cy, heading_up)

        # Draw track
        draw_track(screen, track, view, (w, h), scale=zoom)

        # Draw all alive cars
        for i, car in enumerate(env.cars):
            if not env.alive[i]:
                continue
            if i == leader_idx:
                draw_car(screen, car.pos, car.heading, view, color=RED)
            else:
                draw_car(screen, car.pos, car.heading, view, color=(120, 180, 240))

        # Draw leader sensor rays
        if env.alive[leader_idx]:
            leader_dists = cast_rays(
                leader.pos,
                leader.heading,
                track.segments,
                midpoints=track.seg_midpoints,
            )
            draw_rays(screen, leader.pos, leader.heading, leader_dists, view, line_width=2)

        # Mini-map with all active cars
        other_positions = [car.pos for i, car in enumerate(env.cars) if env.alive[i] and i != leader_idx]
        draw_minimap(screen, track, leader.pos, center, size, other_positions=other_positions)

        # HUD Overlay — Glassmorphism swarm panel (top-left)
        px, py, pw, ph = 8, 8, 248, 250
        _draw_rounded_rect(screen, (10, 14, 22), (px, py, pw, ph), radius=12, alpha=200)

        def hud_text(s, f, x, y, col=WHITE):
            screen.blit(f.render(s, True, col), (x, y))

        mode_str = "  TURBO" if fast_mode else "  VISUAL"
        mode_col = ORANGE if fast_mode else CYAN
        hud_text(mode_str, bold_font, px + 10, py + 10, mode_col)

        # Divider
        pygame.draw.line(screen, (50, 55, 70), (px + 10, py + 32), (px + pw - 10, py + 32), 1)

        hud_text(f"GEN     {pop.generation}",          bold_font, px + 10, py + 40, YELLOW)
        alive_count = int(np.sum(env.alive))
        alive_col = (80, 220, 80) if alive_count > pop_size // 2 else (220, 100, 60)
        hud_text(f"ALIVE   {alive_count} / {pop_size}",   font, px + 10, py + 62, alive_col)
        hud_text(f"TRACK   {track.name}",             font, px + 10, py + 82, (160, 165, 180))
        rot_col = YELLOW if multi_track_mode else (100, 100, 110)
        hud_text(f"MULTI   {'ON' if multi_track_mode else 'OFF'}", font, px + 10, py + 102, rot_col)

        # Divider
        pygame.draw.line(screen, (50, 55, 70), (px + 10, py + 120), (px + pw - 10, py + 120), 1)

        # Leader telemetry
        hud_text("LEADER", bold_font, px + 10, py + 128, CYAN)
        speed_col = ORANGE if speed_kmh > 250 else CYAN
        hud_text(f"  {speed_kmh:5.1f} km/h",                font, px + 10, py + 150, speed_col)
        hud_text(f"  LAP    {fmt_time(leader_lap.lap_time)}", font, px + 10, py + 170)
        rec_col = (80, 255, 100) if all_time_best_lap else WHITE
        hud_text(f"  RECORD {fmt_time(all_time_best_lap)}",   font, px + 10, py + 190, rec_col)
        hud_text(f"  LAPS   {env.total_laps}",                font, px + 10, py + 210)

        # Lap progress bar
        track_prog = track.progress(leader.pos)
        pct = track_prog / track.length
        bar_x, bar_y, bar_w, bar_h = px + 10, py + 232, pw - 20, 8
        pygame.draw.rect(screen, (40, 44, 55), (bar_x, bar_y, bar_w, bar_h), border_radius=4)
        pygame.draw.rect(screen, CYAN, (bar_x, bar_y, int(pct * bar_w), bar_h), border_radius=4)
        hud_text(f"  {track_prog:.0f}m / {track.length:.0f}m ({pct*100:.1f}%)",
                 font, px + 10, py + 244, (140, 145, 165))

        # AI output
        if len(actions) > leader_idx:
            act_f, act_g = actions[leader_idx][0], actions[leader_idx][1]
            gas_col  = (80, 220, 80)  if act_f >= 0 else (220, 80, 60)
            steer_col = YELLOW
            hud_text(f"  GAS {act_f:+.2f}  STEER {act_g:+.2f}",
                     font, px + 10, py + 264, (150, 200, 255))

        # Bottom control legend
        controls = "SPACE: Fast/Visual | M: Multi-Track | N: Track | S: Save | L: Load | R: Reset | C: Camera"
        hud_text(controls, font, 12, h - 26, (180, 180, 180))

        # Status notification banner
        if status_timer > 0:
            status_timer -= DT
            banner = font.render(status_msg, True, YELLOW)
            screen.blit(banner, (w / 2 - banner.get_width() / 2, 20))

        pygame.display.flip()
        if not fast_mode:
            clock.tick(int(1 / DT))
        else:
            clock.tick(120)  # limit frame draw rate in turbo mode so GPU doesn't spin

    pygame.quit()


if __name__ == "__main__":
    main()
