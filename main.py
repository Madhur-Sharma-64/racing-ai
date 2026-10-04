"""Steps 1-2: view the track and the 5 rays.
Mouse = car position, LEFT/RIGHT arrows = rotate heading, N = new random track."""
import numpy as np
import pygame

from sim.track import Track
from sim.raycast import cast_rays, RAY_ANGLES

W, H = 900, 900
YELLOW, GREY, WHITE = (240, 210, 90), (60, 60, 66), (235, 235, 235)


def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    font = pygame.font.SysFont("consolas", 16)
    clock = pygame.time.Clock()

    seed = 1
    track = Track.random(seed)
    heading = 0.0
    scale = 2.0                                   # pixels per meter

    def to_px(p):
        return (W / 2 + p[0] * scale, H / 2 + p[1] * scale)

    def to_world(px):
        return np.array([(px[0] - W / 2) / scale, (px[1] - H / 2) / scale])

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            if e.type == pygame.KEYDOWN and e.key == pygame.K_n:
                seed += 1
                track = Track.random(seed)

        keys = pygame.key.get_pressed()
        heading += (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]) * 0.04

        pos = to_world(pygame.mouse.get_pos())
        dists = cast_rays(pos, heading, track.segments)

        screen.fill((40, 120, 60))
        pygame.draw.polygon(screen, GREY, [to_px(p) for p in track.left] +
                            [to_px(p) for p in track.right[::-1]])
        pygame.draw.lines(screen, WHITE, True, [to_px(p) for p in track.left], 2)
        pygame.draw.lines(screen, WHITE, True, [to_px(p) for p in track.right], 2)

        for a, d in zip(RAY_ANGLES, dists):
            end = pos + d * np.array([np.cos(heading + a), np.sin(heading + a)])
            pygame.draw.line(screen, YELLOW, to_px(pos), to_px(end), 2)
            screen.blit(font.render(f"{d:.0f}m", True, WHITE), to_px(end))
        pygame.draw.circle(screen, (230, 80, 40), to_px(pos), 6)

        status = "ON TRACK" if track.contains(pos) else "OFF TRACK"
        info = f"{status}  progress={track.progress(pos):.0f}/{track.length:.0f}m  seed={seed}"
        screen.blit(font.render(info, True, WHITE), (10, 10))
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


if __name__ == "__main__":
    main()
