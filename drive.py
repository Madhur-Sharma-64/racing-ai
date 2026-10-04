"""Drive the car by hand, with a start screen, lap timer and lap counter.

SPACE / ENTER  start        UP / W  gas        DOWN / S  brake
LEFT / RIGHT or A / D  steer
P  pause       R  restart lap       N  next track (resets laps)
On the start screen: LEFT / RIGHT pick the track.
C  camera (car points up / north up)     mouse wheel or + / -  zoom
ESC  back to the start screen (press again to quit)
"""
import numpy as np
import pygame

from sim.track import Track
from sim.tracks import circuit_names, load_circuit
from sim.car import Car
from sim.lap import LapTimer
from sim.raycast import cast_rays, RAY_ANGLES
from ui.camera import follow_view, fit_view

W, H = 1100, 650
YELLOW, GREY, WHITE = (240, 210, 90), (60, 60, 66), (235, 235, 235)
GREEN, RED = (40, 120, 60), (230, 80, 40)
DT = 0.05
CHOICES = circuit_names() + ["Random"]


def spawn(track):
    return Car(track.center[0], np.arctan2(track.tangent[0][1], track.tangent[0][0]))


def fit(track):
    pts = np.concatenate([track.left, track.right])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    return (lo + hi) / 2, (hi - lo)            # track center and size in meters


def build_track(choice, seed):
    return Track.random(seed) if choice == "Random" else load_circuit(choice)


def fmt(t):
    """Seconds -> mm:ss.mmm  (None -> placeholder)."""
    if t is None:
        return "--:--.---"
    m, s = divmod(t, 60)
    return f"{int(m):02d}:{s:06.3f}"


def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H), pygame.RESIZABLE)
    pygame.display.set_caption("Racing AI")
    font = pygame.font.SysFont("consolas", 18)
    big = pygame.font.SysFont("consolas", 52, bold=True)
    mid = pygame.font.SysFont("consolas", 30, bold=True)
    clock = pygame.time.Clock()

    idx = CHOICES.index("Monza")
    seed = 1
    track = build_track(CHOICES[idx], seed)
    center, size = fit(track)
    car = spawn(track)
    lap = LapTimer(track)
    steer = 0.0                                # smoothed keyboard steering
    state = "menu"                             # menu | playing | paused
    message, message_t = "", 0.0
    heading_up = True                          # camera: car points up the screen
    zoom = 4.5                                 # pixels per meter while following the car

    def text(s, f, x, y, color=WHITE, centered=False):
        img = f.render(s, True, color)
        if centered:
            x -= img.get_width() / 2
        screen.blit(img, (x, y))

    def set_track(new_idx):
        nonlocal idx, seed, track, center, size, car, lap, steer
        idx = new_idx % len(CHOICES)
        if CHOICES[idx] == "Random":
            seed += 1
        track = build_track(CHOICES[idx], seed)
        center, size = fit(track)
        lap = LapTimer(track)
        car = spawn(track)
        steer = 0.0

    def new_run():
        nonlocal car, steer
        car = spawn(track)
        steer = 0.0
        lap.start(car.pos)

    running = True
    while running:
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
                    car = spawn(track)
                    steer = 0.0
                    lap.restart_lap(car.pos)
                elif e.key == pygame.K_n:
                    set_track(idx + 1)
                    new_run()

        # ---- simulation (only while playing) ----
        f = 0.0
        if state == "playing":
            k = pygame.key.get_pressed()
            f = float(k[pygame.K_UP] or k[pygame.K_w]) - float(k[pygame.K_DOWN] or k[pygame.K_s])
            target = (float(k[pygame.K_RIGHT] or k[pygame.K_d])
                      - float(k[pygame.K_LEFT] or k[pygame.K_a]))
            steer += np.clip(target - steer, -4 * DT, 4 * DT)   # ramp like a real wheel
            car.update(f, steer, DT)

            if not track.contains(car.pos):
                car = spawn(track)
                steer = 0.0
                lap.restart_lap(car.pos)
                message, message_t = "OFF TRACK - lap restarted", 1.5
            elif lap.update(car.pos, DT):
                message, message_t = f"LAP {lap.laps}:  {fmt(lap.last_time)}", 3.0
            message_t = max(0.0, message_t - DT)

        # ---- drawing ----
        screen.fill(GREEN)
        w, h = screen.get_size()
        if state == "menu":
            # whole track in view, so you can preview it while choosing
            scale = min((w - 40) / size[0], (h - 70) / size[1])

            def view(P):
                return fit_view(P, center, scale, w / 2, h / 2 + 15)
        else:
            # camera follows the car; in heading-up mode the car sits low so you see ahead
            scale = zoom
            cx, cy = w / 2, (h * 0.68 if heading_up else h / 2)

            def view(P):
                return follow_view(P, car.pos, car.heading, scale, cx, cy, heading_up)

        L, R = view(track.left), view(track.right)

        def on_screen(P):
            return (P[:, 0] > -80) & (P[:, 0] < w + 80) & (P[:, 1] > -80) & (P[:, 1] < h + 80)

        inL, inR = on_screen(L), on_screen(R)
        visible = inL | inR | np.roll(inL, -1) | np.roll(inR, -1)     # skip off-screen pieces
        for i in np.nonzero(visible)[0]:
            j = (i + 1) % len(L)
            pygame.draw.polygon(screen, GREY, [L[i].tolist(), L[j].tolist(),
                                               R[j].tolist(), R[i].tolist()])
        pygame.draw.lines(screen, WHITE, True, L.tolist(), 2)
        pygame.draw.lines(screen, WHITE, True, R.tolist(), 2)
        pygame.draw.line(screen, WHITE, L[0].tolist(), R[0].tolist(), max(2, int(scale * 0.8)))

        dists = cast_rays(car.pos, car.heading, track.segments)
        origin = view(car.pos).tolist()
        for a, d in zip(RAY_ANGLES, dists):
            end = car.pos + d * np.array([np.cos(car.heading + a), np.sin(car.heading + a)])
            pygame.draw.line(screen, YELLOW, origin, view(end).tolist(), 1 if state == "menu" else 2)

        fwd = np.array([np.cos(car.heading), np.sin(car.heading)])
        side = np.array([-fwd[1], fwd[0]])
        body = np.array([car.pos + fwd * 4, car.pos - fwd * 2 + side * 1.5,
                         car.pos - fwd * 2 - side * 1.5])
        pygame.draw.polygon(screen, RED, view(body).tolist())

        # HUD
        text(f"{car.speed * 3.6:4.0f} km/h", font, 12, 10)
        text(f"LAP TIME  {fmt(lap.lap_time)}", font, 12, 34)
        text(f"LAPS DONE {lap.laps}", font, 12, 58)
        text(f"LAST      {fmt(lap.last_time)}", font, 12, 82)
        text(f"BEST      {fmt(lap.best_time)}", font, 12, 106)
        text(track.name, font, 12, 130, (170, 170, 170))
        if message_t > 0 and state == "playing":
            text(message, big, screen.get_width() / 2, 20, YELLOW, centered=True)

        if state != "menu":
            # mini-map, top right: the whole track with the car on it
            bw, bh = 240, 180
            x0, y0 = w - bw - 12, 12
            panel = pygame.Surface((bw, bh), pygame.SRCALPHA)
            panel.fill((0, 0, 0, 150))
            screen.blit(panel, (x0, y0))
            ms = min((bw - 24) / size[0], (bh - 24) / size[1])

            def mini(P):
                return fit_view(P, center, ms, x0 + bw / 2, y0 + bh / 2)

            pygame.draw.lines(screen, (200, 200, 200), True, mini(track.center).tolist(), 3)
            pygame.draw.line(screen, WHITE, mini(track.left[0]).tolist(),
                             mini(track.right[0]).tolist(), 3)
            pygame.draw.circle(screen, RED, mini(car.pos).tolist(), 5)
            text("camera: " + ("car up" if heading_up else "north up") + "   (C)   zoom: wheel / + -",
                 font, 12, h - 28, (150, 150, 150))

        # menu / pause overlay
        if state != "playing":
            w, h = screen.get_size()
            shade = pygame.Surface((w, h), pygame.SRCALPHA)
            shade.fill((0, 0, 0, 170))
            screen.blit(shade, (0, 0))
            if state == "menu":
                text("RACING AI", big, w / 2, h / 2 - 170, YELLOW, centered=True)
                text(f"<  {track.name}  >", mid, w / 2, h / 2 - 95, WHITE, centered=True)
                if hasattr(track, "country"):
                    info = (f"{track.country}  |  real lap {track.real_km:.2f} km  |  "
                            f"game lap {track.length / 1000:.2f} km")
                    note = track.note
                else:
                    info = f"game lap {track.length / 1000:.2f} km"
                    note = "A new random track every time you pick it."
                text(info, font, w / 2, h / 2 - 55, (190, 190, 190), centered=True)
                text(note, font, w / 2, h / 2 - 30, (150, 150, 150), centered=True)
                text("LEFT / RIGHT  choose track      SPACE  start", font, w / 2, h / 2 + 30,
                     YELLOW, centered=True)
                text("Arrows / WASD  drive     P  pause     R  restart lap",
                     font, w / 2, h / 2 + 70, (190, 190, 190), centered=True)
                text("N  next track     C  camera     wheel  zoom     ESC  quit", font, w / 2, h / 2 + 96,
                     (190, 190, 190), centered=True)
            else:
                text("PAUSED", big, w / 2, h / 2 - 50, YELLOW, centered=True)
                text("Press P to resume", font, w / 2, h / 2 + 20, centered=True)

        pygame.display.flip()
        clock.tick(int(1 / DT))
    pygame.quit()


if __name__ == "__main__":
    main()
