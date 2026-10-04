"""Step 3: drive the car by hand.  UP gas, DOWN brake, LEFT/RIGHT steer.
R = reset, N = new track."""
import numpy as np
import pygame

from sim.track import Track
from sim.car import Car
from sim.raycast import cast_rays, RAY_ANGLES

W, H = 900, 900
YELLOW, GREY, WHITE = (240, 210, 90), (60, 60, 66), (235, 235, 235)
DT = 0.05
SCALE = 2.0


def spawn(track):
    return Car(track.center[0], np.arctan2(track.tangent[0][1], track.tangent[0][0]))


def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    font = pygame.font.SysFont("consolas", 16)
    clock = pygame.time.Clock()

    seed = 1
    track = Track.random(seed)
    car = spawn(track)
    steer = 0.0                       # smoothed keyboard steering (humans only)

    def px(p):
        return (W / 2 + p[0] * SCALE, H / 2 + p[1] * SCALE)

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_r:
                    car = spawn(track)
                if e.key == pygame.K_n:
                    seed += 1
                    track = Track.random(seed)
                    car = spawn(track)

        k = pygame.key.get_pressed()
        f = float(k[pygame.K_UP]) - float(k[pygame.K_DOWN])
        target = float(k[pygame.K_RIGHT]) - float(k[pygame.K_LEFT])
        steer += np.clip(target - steer, -4 * DT, 4 * DT)   # ramp like a real wheel
        car.update(f, steer, DT)

        if not track.contains(car.pos):
            car = spawn(track)        # crash -> back to start

        dists = cast_rays(car.pos, car.heading, track.segments)

        screen.fill((40, 120, 60))
        pygame.draw.polygon(screen, GREY, [px(p) for p in track.left] +
                            [px(p) for p in track.right[::-1]])
        pygame.draw.lines(screen, WHITE, True, [px(p) for p in track.left], 2)
        pygame.draw.lines(screen, WHITE, True, [px(p) for p in track.right], 2)

        for a, d in zip(RAY_ANGLES, dists):
            end = car.pos + d * np.array([np.cos(car.heading + a), np.sin(car.heading + a)])
            pygame.draw.line(screen, YELLOW, px(car.pos), px(end), 1)

        # car body as a triangle pointing along the heading
        fwd = np.array([np.cos(car.heading), np.sin(car.heading)])
        side = np.array([-fwd[1], fwd[0]])
        body = [car.pos + fwd * 4, car.pos - fwd * 2 + side * 1.5, car.pos - fwd * 2 - side * 1.5]
        pygame.draw.polygon(screen, (230, 80, 40), [px(p) for p in body])

        info = (f"{car.speed * 3.6:5.0f} km/h   throttle={f:+.0f}  steer={steer:+.2f}   "
                f"progress={track.progress(car.pos):.0f}/{track.length:.0f}m   seed={seed}")
        screen.blit(font.render(info, True, WHITE), (10, 10))
        pygame.display.flip()
        clock.tick(int(1 / DT))
    pygame.quit()


if __name__ == "__main__":
    main()
