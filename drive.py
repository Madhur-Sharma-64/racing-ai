"""Drive the car by hand, with a start screen, lap timer, and optional AI rival.

SPACE / ENTER  start        UP / W  gas        DOWN / S  brake
LEFT / RIGHT or A / D  steer
P  pause       R  restart lap       N  next track (resets laps)
G  toggle AI rival car (if models/best_driver.npz exists)     M  toggle audio
E  cycle season / weather (Spring Sakura, Autumn, Cold, Summer)
On the start screen: LEFT / RIGHT pick the track.
C  camera (car points up / north up)     mouse wheel or + / -  zoom
ESC  back to the start screen (press again to quit)
"""
from pathlib import Path
import numpy as np
import pygame

from sim.track import Track
from sim.tracks import circuit_names, load_circuit, test_circuit_names, load_test_circuit
from sim.car import Car
from sim.lap import LapTimer
from sim.raycast import cast_rays
from sim.env import build_observation
from ai.network import NeuralNetwork
from ui.camera import follow_view, fit_view
from ui.sound import SoundEngine
from ui.environment import ENV
from ui.render import (
    BG_TERRAIN,
    RED,
    CYAN,
    draw_track,
    draw_car,
    draw_rays,
    draw_hud,
    draw_minimap,
    draw_menu_overlay,
    draw_pause_overlay,
    draw_car_tag,
    fmt_time,
    VFX,
)

from sim.generator import generate_track, get_archetype_names

W, H = 1100, 650
FPS = 60
DEFAULT_DT = 1.0 / FPS
CHOICES = (
    circuit_names()
    + [f"[Test] {n}" for n in test_circuit_names()]
    + [f"[Sketch] {n}" for n in get_archetype_names()]
    + ["Random", "Random Extreme"]
)

KEEPER_MODEL_PATH = Path("models/KEEPER.npz")
SPEED_DEMON_MODEL_PATH = Path("models/SPEED_DEMON.npz")
MADX_MODEL_PATH = Path("models/MADX.npz")
ZORO_MODEL_PATH = Path("models/ZORO.npz")

KEEPER_COLOR = (0, 230, 255)       # Cyan (Safety / Focus)
SPEED_DEMON_COLOR = (255, 120, 0)   # Vivid Demon Orange
MADX_COLOR = (255, 215, 0)         # Gold
ZORO_COLOR = (46, 204, 113)        # Zoro Emerald Green
PLAYER_COLOR = (255, 60, 60)       # Red


def spawn(track, slot=0):
    """Spawn car on starting grid slot (0: Player, 1: Safe AI, 2: Current AI).

    Uses track centerline indices backwards along the circuit and scales lateral
    offset safely according to track width so the car is ALWAYS 100% on track,
    even on ultra-narrow or curved start-finish straights like Monaco.
    """
    dist_back = {0: 0.0, 1: 5.0, 2: 10.0, 3: 15.0, 4: 20.0}.get(slot, 0.0)
    lat_frac = {0: 0.0, 1: +0.28, 2: -0.28, 3: +0.20, 4: -0.20}.get(slot, 0.0)

    pts_back = int(round(dist_back / 4.0))
    idx = (0 - pts_back) % len(track.center)

    # Safe lateral offset clamped strictly inside track boundaries
    lat_offset = np.clip(track.width * 0.5 * lat_frac, -track.width * 0.35, track.width * 0.35)
    pos = track.center[idx] + track.normal[idx] * lat_offset
    heading = np.arctan2(track.tangent[idx][1], track.tangent[idx][0])
    return Car(pos, heading)


def fit(track):
    pts = np.concatenate([track.left, track.right])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    return (lo + hi) / 2, (hi - lo)            # track center and size in meters


def build_track(choice, seed):
    if choice == "Random":
        return Track.random(seed)
    if choice == "Random Extreme":
        return Track.random_extreme(seed)
    if choice.startswith("[Test] "):
        return load_test_circuit(choice[7:])
    if choice.startswith("[Sketch] "):
        return generate_track(archetype=choice[9:], seed=seed)
    return load_circuit(choice)


class AIRival:
    """Manages an autonomous AI competitor on track."""

    def __init__(self, name, model_path, color, slot):
        self.name = name
        self.model_path = Path(model_path)
        self.color = color
        self.slot = slot
        self.net = None
        self.car = None
        self.lap = None
        self.reload()

    def reload(self):
        if self.model_path.exists():
            try:
                self.net = NeuralNetwork.load(self.model_path)
            except Exception:
                self.net = None
        else:
            self.net = None

    def reset(self, track):
        if self.net is not None:
            self.car = spawn(track, slot=self.slot)
            self.lap = LapTimer(track)
            self.lap.start(self.car.pos)
        else:
            self.car = None
            self.lap = None

    def step(self, track, dt=DEFAULT_DT):
        if self.net is None or self.car is None or self.lap is None:
            return
        rays = cast_rays(
            self.car.pos,
            self.car.heading,
            track.segments,
            midpoints=track.seg_midpoints,
        )
        n_in = self.net.layer_sizes[0]
        if n_in == 6:
            # Legacy 6-input model expects 5 rays + speed (total 6):
            legacy_rays = rays[[1, 3, 4, 5, 7]]  # -60,-30,0,30,60 from 9-ray array
            obs = build_observation(self.car, track, legacy_rays, six_dim=True, obs_dim=6)
        elif n_in == 12:
            # Legacy 12-input model expects 5 rays + 7 telemetry:
            legacy_rays = rays[[1, 3, 4, 5, 7]]  # -60,-30,0,30,60 from 9-ray array
            obs = build_observation(self.car, track, legacy_rays, obs_dim=12)
        elif n_in == 16:
            # 16-input MADX (classic architecture before 200m line)
            obs = build_observation(self.car, track, rays, obs_dim=16)
        elif n_in == 18:
            # 18-input ZORO v1 (9 rays + 200m radar + 8 telemetry)
            obs = build_observation(self.car, track, obs_dim=18)
        else:
            # 22-input ZORO v2 Universal (18 inputs + 4 universal geometric invariants)
            obs = build_observation(self.car, track, obs_dim=22)
        act = self.net.forward(obs)
        self.car.update(float(act[0]), float(act[1]), dt)
        on_track, prog, _ = track.query(self.car.pos)
        if not on_track:
            self.car = spawn(track, slot=self.slot)
            self.lap.restart_lap(self.car.pos)
        else:
            self.lap.update(self.car.pos, dt, progress=prog)


def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H), pygame.RESIZABLE)
    pygame.display.set_caption("Racing AI - Human vs AI Championship")
    font = pygame.font.SysFont("consolas", 18)
    big = pygame.font.SysFont("consolas", 52, bold=True)
    mid = pygame.font.SysFont("consolas", 30, bold=True)
    clock = pygame.time.Clock()

    idx = CHOICES.index("Monza")
    seed = 1
    track = build_track(CHOICES[idx], seed)
    center, size = fit(track)
    car = spawn(track, slot=0)
    lap = LapTimer(track)
    steer = 0.0                                # smoothed keyboard steering
    state = "menu"                             # menu | playing | paused
    message, message_t = "", 0.0
    heading_up = True                          # camera: car points up the screen
    zoom = 4.5                                 # pixels per meter while following the car
    sound_engine = SoundEngine()

    # Build initial roadside environment (trees, ambient petals/leaves/snow)
    ENV.build_trees_for_track(track)

    # 4 Autonomous AI Rivals: KEEPER, SPEED DEMON, MADX, and ZORO
    rivals = [
        AIRival("KEEPER", KEEPER_MODEL_PATH, KEEPER_COLOR, slot=1),
        AIRival("SPEED DEMON", SPEED_DEMON_MODEL_PATH, SPEED_DEMON_COLOR, slot=2),
        AIRival("MADX", MADX_MODEL_PATH, MADX_COLOR, slot=3),
        AIRival("ZORO", ZORO_MODEL_PATH, ZORO_COLOR, slot=4),
    ]
    ai_enabled = False

    def set_track(new_idx):
        nonlocal idx, seed, track, center, size, car, lap, steer
        idx = new_idx % len(CHOICES)
        if CHOICES[idx] == "Random":
            seed += 1
        track = build_track(CHOICES[idx], seed)
        center, size = fit(track)
        lap = LapTimer(track)
        car = spawn(track, slot=0)
        steer = 0.0
        for r in rivals:
            r.reset(track)
        # Regenerate roadside trees & ambient particles for the new track
        ENV.build_trees_for_track(track)

    def new_run():
        nonlocal car, steer
        car = spawn(track, slot=0)
        steer = 0.0
        lap.start(car.pos)
        for r in rivals:
            r.reset(track)

    running = True
    while running:
        raw_dt = clock.tick(FPS) / 1000.0
        dt = float(np.clip(raw_dt, 0.001, 0.05))

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.MOUSEWHEEL:
                zoom = float(np.clip(zoom * 1.1 ** e.y, 1.5, 12.0))
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_c:
                    heading_up = not heading_up
                elif e.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    zoom = min(12.0, zoom * 1.15)
                elif e.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    zoom = max(1.5, zoom / 1.15)
                elif e.key == pygame.K_m:
                    unmuted = sound_engine.toggle_mute()
                    message = f"AUDIO: {'ON' if unmuted else 'MUTED'}"
                    message_t = 2.0
                elif e.key == pygame.K_e:
                    # Cycle season / weather environment
                    new_season = ENV.cycle_season()
                    ENV.build_trees_for_track(track)
                    message = f"SEASON: {new_season.icon} {new_season.name.upper()}"
                    message_t = 2.5
                elif e.key == pygame.K_g:
                    # Toggle AI rivals
                    ai_enabled = not ai_enabled
                    for r in rivals:
                        r.reload()
                        r.reset(track)
                    active_count = sum(1 for r in rivals if r.net is not None)
                    if ai_enabled:
                        message = f"AI RIVALS ON ({active_count} CARS)"
                    else:
                        message = "AI RIVALS OFF"
                    message_t = 2.0
                elif state == "menu":
                    if e.key in (pygame.K_SPACE, pygame.K_RETURN):
                        new_run()
                        state = "playing"
                    elif e.key == pygame.K_LEFT:
                        set_track(idx - 1)
                    elif e.key == pygame.K_RIGHT:
                        set_track(idx + 1)
                    elif e.key == pygame.K_ESCAPE:
                        running = False
                elif e.key == pygame.K_ESCAPE:
                    state = "menu"
                elif e.key == pygame.K_p:
                    state = "paused" if state == "playing" else "playing"
                elif e.key == pygame.K_r:
                    car = spawn(track, slot=0)
                    steer = 0.0
                    lap.restart_lap(car.pos)
                    for r in rivals:
                        r.reset(track)
                elif e.key == pygame.K_n:
                    set_track(idx + 1)
                    new_run()

        # ---- simulation (only while playing) ----
        if state == "playing":
            # Human player
            k = pygame.key.get_pressed()
            f = float(k[pygame.K_UP] or k[pygame.K_w]) - float(k[pygame.K_DOWN] or k[pygame.K_s])
            target = (float(k[pygame.K_RIGHT] or k[pygame.K_d])
                      - float(k[pygame.K_LEFT] or k[pygame.K_a]))
            steer += np.clip(target - steer, -4.0 * dt, 4.0 * dt)
            prev_pos = car.pos.copy()
            car.update(f, steer, dt)

            # VFX: Skid marks and tire smoke during heavy sliding or hard braking
            is_braking = (f < -0.1)
            is_sliding = abs(car.lat_accel) > 20.0 or (is_braking and car.speed > 8.0)
            if is_sliding and car.speed > 5.0:
                VFX.add_skid(prev_pos, car.pos, width=0.45)
                fwd = np.array([np.cos(car.heading), np.sin(car.heading)])
                side = np.array([-np.sin(car.heading), np.cos(car.heading)])
                smoke_pos = car.pos - fwd * 1.5 + side * (1.4 if car.lat_accel > 0 else -1.4)
                VFX.add_smoke(smoke_pos, -fwd * (car.speed * 0.3), size=1.8, count=1)
                if abs(car.lat_accel) > 28.0:
                    VFX.add_sparks(car.pos - fwd * 1.8, car.heading, count=2)

            VFX.update(dt)

            on_track, progress, _ = track.query(car.pos)
            if not on_track:
                car = spawn(track, slot=0)
                steer = 0.0
                lap.restart_lap(car.pos)
                message, message_t = "OFF TRACK - lap restarted", 1.5
            elif lap.update(car.pos, dt, progress=progress):
                message, message_t = f"LAP {lap.laps}:  {fmt_time(lap.last_time)}", 3.0

            # Step all AI rivals
            if ai_enabled:
                for r in rivals:
                    r.step(track, dt)

            is_throttle = (f > 0.05)
            is_turning = (abs(steer) > 0.15)
            # Dynamic Audio: reacts only to active throttle, braking, or steering/turning
            sound_engine.update(car, is_braking=is_braking, is_throttle=is_throttle, is_turning=is_turning)

            # Update environment: shedding leaves/petals, wind, and car aerodynamic wake
            all_cars = [car] + [r.car for r in rivals if r.car is not None] if ai_enabled else [car]
            ENV.update(dt, cars=all_cars)

            message_t = max(0.0, message_t - dt)
        else:
            sound_engine.silence()
            # Keep environment alive even in menu/pause (ambient wind drift)
            ENV.update(dt, cars=[car])

        # ---- drawing ----
        screen.fill(ENV.season.terrain)
        w, h = screen.get_size()
        if state == "menu":
            scale = min((w - 40) / size[0], (h - 70) / size[1])

            def view(P):
                return fit_view(P, center, scale, w / 2, h / 2 + 15)
        else:
            scale = zoom
            cx, cy = w / 2, (h * 0.68 if heading_up else h / 2)

            def view(P):
                return follow_view(P, car.pos, car.heading, scale, cx, cy, heading_up)

        # Draw track geometry with ambient shadows and 3D kerbs
        draw_track(screen, track, view, (w, h), scale=scale)

        # Draw roadside trees (overhanging canopies on top of track edges)
        ENV.draw_trees(screen, view, (w, h), scale=scale)

        # Draw persistent skid marks on tarmac
        VFX.draw_skids(screen, view, scale)

        # Draw AI rival cars & badges if enabled
        other_positions = []
        if ai_enabled:
            for r in rivals:
                if r.car is not None:
                    draw_car(screen, r.car.pos, r.car.heading, view, color=r.color, speed=r.car.speed)
                    draw_car_tag(screen, view, r.car.pos, r.name, r.color, font)
                    other_positions.append((r.car.pos, r.color))

        # Raycast distance sensors for player car (9 rays + 200m forward radar)
        dists, radar_dist = cast_rays(
            car.pos,
            car.heading,
            track.segments,
            midpoints=track.seg_midpoints,
            return_radar=True,
            radar_range=200.0,
        )
        draw_rays(screen, car.pos, car.heading, dists, view, line_width=1 if state == "menu" else 2, radar_dist=radar_dist)

        # Player car with brake light indicator
        is_player_braking = (state == "playing" and f < -0.1)
        draw_car(screen, car.pos, car.heading, view, color=PLAYER_COLOR, speed=car.speed, braking=is_player_braking)
        draw_car_tag(screen, view, car.pos, "YOU", PLAYER_COLOR, font)

        # Particle effects (Tire smoke and titanium sparks)
        VFX.draw_particles(screen, view, scale)

        # Environment particles: shedding cherry blossoms / autumn leaves / snowflakes
        ENV.draw_particles(screen, view, (w, h), scale=scale)

        # HUD (Broadcast F1 digital dash)
        rival_hud_list = [
            {"name": r.name, "car": r.car, "lap": r.lap, "color": r.color}
            for r in rivals
            if r.car is not None
        ] if ai_enabled else None

        draw_hud(
            screen,
            car,
            lap,
            track,
            message=message if (message_t > 0 and state == "playing") else "",
            rivals=rival_hud_list,
            font=font,
            message_font=big,
            ai_enabled=ai_enabled,
        )

        # Mini-map
        if state != "menu":
            draw_minimap(screen, track, car.pos, center, size, other_positions=other_positions)

        # Overlay states
        if state == "menu":
            draw_menu_overlay(screen, track, font, mid, big)
        elif state == "paused":
            draw_pause_overlay(screen, font, big)

        pygame.display.flip()
    sound_engine.close()
    pygame.quit()


if __name__ == "__main__":
    main()
