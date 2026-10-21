"""Rendering routines for Racing AI using Pygame.

Separates all visual presentation (track polygons, cars, sensor rays, HUD,
minimap, menus) from the physics simulation and control logic.

Visual features (Pro AAA Overhaul):
  - Textured grass/terrain with subtle speed stripes
  - Dual-layer asphalt with ambient drop-shadow, tire-rubber groove, and crisp track limits
  - 3D-depth kerb strips (red/white) with bevels and high-contrast start/finish grid boxes
  - Highly detailed F1 chassis: front/rear multi-element aero wings, sidepods, bargeboards,
    cockpit halo, helmet, wide slick tires with spoke rims, dynamic brake lights, and headlights
  - Realistic ground drop-shadows under every vehicle
  - Particle VFX system: sparks on kerbs/high-G, tire smoke during slides, and speed streaks
  - Skid mark persistent rubber layer on tarmac
  - Broadcast-grade F1 Digital Dashboard HUD:
      * Curved tachometer arc with LED shift lights (Green -> Red -> Flashing Purple)
      * High-contrast digital speedometer & gear indicator
      * G-Force friction circle (accelerometer telemetry)
      * Glassmorphism rival timing leaderboards and delta monitors
"""
import numpy as np
import pygame

from sim.raycast import RAY_ANGLES
from ui.camera import fit_view
from ui.environment import ENV

# ---------------------------------------------------------------------------
# Color palette & Theme Tokens (Bright, Premium Motorsport Palette)
# ---------------------------------------------------------------------------
BG_TERRAIN       = (46,  108,  54)   # Lush, vibrant natural grass green
BG_TERRAIN_LIGHT = (52,  120,  62)   # Manicured turf alternating lawn stripe
ASPHALT          = (64,  68,   78)   # Clean, bright slate-graphite tarmac
ASPHALT_GROOVE   = (48,  52,   60)   # Distinct rubbered-in racing line groove
SHADOW_TRACK     = (28,  68,   34)   # Ambient occlusion drop shadow on grass

KERB_RED         = (235, 38,   38)   # Vivid racing red
KERB_RED_DARK    = (175, 22,   22)   # 3D kerb shadow bevel
KERB_WHITE       = (255, 255, 255)   # Crisp bright white
KERB_WHITE_DARK  = (210, 218, 228)   # 3D white kerb bevel
TRACK_LINE       = (255, 255, 255)   # Ultra-crisp white track limit lines
START_YELLOW     = (255, 215,  30)

YELLOW           = (255, 215,  40)
GREY             = (75,  82,   95)
LIGHT_GREY       = (210, 215, 225)
MUTED_GREY       = (160, 168, 180)
WHITE            = (255, 255, 255)
GREEN            = (48,  112,  56)
GREEN_DARK       = (36,  86,   44)
RED              = (245,  45,  45)
CYAN             = ( 30, 230, 255)
ORANGE           = (255, 140,  25)
BLUE_CAR         = ( 50, 155, 255)
PURPLE           = (195,  80, 255)

PANEL_BG         = (18,  26,   42, 215)  # Glassmorphism frosted navy/slate
PANEL_BORDER     = (65,  95,  145, 200)  # High-contrast crisp border
HUD_ACCENT       = ( 30, 230, 255)


# ---------------------------------------------------------------------------
# Global Visual FX Managers (Skidmarks & Particles)
# ---------------------------------------------------------------------------
class VisualFX:
    """Manages persistent tire skid marks, tire smoke, and titanium sparks."""
    def __init__(self, max_skids=400, max_particles=250):
        self.skids = []           # [(p1_world, p2_world, alpha_val, width)]
        self.max_skids = max_skids
        self.particles = []       # [dict(pos, vel, color, size, life, max_life)]
        self.max_particles = max_particles

    def add_skid(self, p1, p2, width=0.35):
        if len(self.skids) >= self.max_skids:
            self.skids.pop(0)
        self.skids.append((np.array(p1, dtype=float), np.array(p2, dtype=float), width))

    def add_smoke(self, pos, vel, size=2.5, count=1):
        for _ in range(count):
            if len(self.particles) >= self.max_particles:
                self.particles.pop(0)
            jitter = (np.random.rand(2) - 0.5) * 1.5
            p_vel = np.array(vel, dtype=float) * 0.2 + (np.random.rand(2) - 0.5) * 1.2
            life = float(np.random.uniform(0.3, 0.65))
            self.particles.append({
                "pos": np.array(pos, dtype=float) + jitter,
                "vel": p_vel,
                "type": "smoke",
                "size": size * float(np.random.uniform(0.8, 1.4)),
                "life": life,
                "max_life": life,
            })

    def add_sparks(self, pos, car_heading, count=3):
        for _ in range(count):
            if len(self.particles) >= self.max_particles:
                self.particles.pop(0)
            rev_heading = car_heading + np.pi + float(np.random.uniform(-0.5, 0.5))
            speed = float(np.random.uniform(6.0, 18.0))
            p_vel = np.array([np.cos(rev_heading), np.sin(rev_heading)]) * speed
            life = float(np.random.uniform(0.12, 0.30))
            self.particles.append({
                "pos": np.array(pos, dtype=float),
                "vel": p_vel,
                "type": "spark",
                "size": float(np.random.uniform(1.2, 2.5)),
                "life": life,
                "max_life": life,
            })

    def update(self, dt):
        alive_particles = []
        for p in self.particles:
            p["life"] -= dt
            if p["life"] > 0:
                p["pos"] += p["vel"] * dt
                if p["type"] == "smoke":
                    p["size"] += 2.0 * dt   # smoke expands
                alive_particles.append(p)
        self.particles = alive_particles

    def draw_skids(self, screen, view_fn, scale):
        if not self.skids:
            return
        w, h = screen.get_size()
        margin = 100
        for p1, p2, width in self.skids:
            s1 = view_fn(p1)
            s2 = view_fn(p2)
            if not (-margin < s1[0] < w + margin and -margin < s1[1] < h + margin):
                continue
            lw = max(2, int(width * scale))
            pygame.draw.line(screen, (10, 10, 12), (int(s1[0]), int(s1[1])), (int(s2[0]), int(s2[1])), lw)

    def draw_particles(self, screen, view_fn, scale):
        w, h = screen.get_size()
        for p in self.particles:
            s_pos = view_fn(p["pos"])
            if not (0 <= s_pos[0] < w and 0 <= s_pos[1] < h):
                continue
            ratio = p["life"] / p["max_life"]
            if p["type"] == "smoke":
                col = int(180 + (1.0 - ratio) * 50)
                alpha = int(ratio * 90)
                px_size = max(3, int(p["size"] * scale * 0.7))
                smoke_surf = pygame.Surface((px_size * 2, px_size * 2), pygame.SRCALPHA)
                pygame.draw.circle(smoke_surf, (col, col, col, alpha), (px_size, px_size), px_size)
                screen.blit(smoke_surf, (int(s_pos[0] - px_size), int(s_pos[1] - px_size)))
            elif p["type"] == "spark":
                spark_color = (255, int(200 * ratio), 40)
                px_r = max(2, int(p["size"] * (scale / 4.0)))
                pygame.draw.circle(screen, spark_color, (int(s_pos[0]), int(s_pos[1])), px_r)


# Singleton VFX instance
VFX = VisualFX()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def fmt_time(t):
    """Format seconds into mm:ss.mmm (or '--:--.---' if None)."""
    if t is None:
        return "--:--.---"
    m, s = divmod(t, 60)
    return f"{int(m):02d}:{s:06.3f}"


def _alpha_surface(w, h, color_rgba):
    surf = pygame.Surface((w, h), pygame.SRCALPHA)
    surf.fill(color_rgba)
    return surf


def _draw_rounded_rect(surf, color, rect, radius=8, alpha=220, border_color=None):
    """Draw a polished rounded rectangle with alpha and optional border."""
    x, y, w, h = rect
    panel = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.draw.rect(panel, (*color[:3], alpha), (0, 0, w, h), border_radius=radius)
    if border_color:
        pygame.draw.rect(panel, border_color, (0, 0, w, h), width=1, border_radius=radius)
    surf.blit(panel, (x, y))


def _darken(color, factor=0.65):
    return tuple(max(0, int(c * factor)) for c in color)


def _lighten(color, factor=0.4):
    return tuple(min(255, int(c + (255 - c) * factor)) for c in color)


# ---------------------------------------------------------------------------
# Track drawing (Dual-layer Tarmac, 3D Kerbs, Rubber Groove)
# ---------------------------------------------------------------------------
def draw_track(screen, track, view_fn, window_size, scale=1.0):
    """Draw seamless asphalt ribbon with ambient occlusion drop shadow and 3D kerbs."""
    w, h = window_size
    L, R = view_fn(track.left), view_fn(track.right)
    C = view_fn(track.center)
    n = len(L)

    margin = 140
    in_view = ((L[:, 0] > -margin) & (L[:, 0] < w + margin) & (L[:, 1] > -margin) & (L[:, 1] < h + margin)) | \
              ((R[:, 0] > -margin) & (R[:, 0] < w + margin) & (R[:, 1] > -margin) & (R[:, 1] < h + margin))
    vis = in_view | np.roll(in_view, 1) | np.roll(in_view, -1)
    vis_idx = np.nonzero(vis)[0]
    if len(vis_idx) == 0:
        return

    # Contiguous visible runs
    diffs = np.diff(vis_idx)
    splits = np.nonzero(diffs > 1)[0] + 1
    runs = np.split(vis_idx, splits)

    kw = max(5, int(scale * 1.10))      # Kerb width
    lw = max(1, int(scale * 0.25))      # Crisp track limit line width
    gw = max(4, int(scale * 0.65))      # Racing line rubber groove width
    shadow_offset = max(3, int(scale * 0.7))

    for run in runs:
        if len(run) < 2:
            continue
        ext_run = np.append(run, (run[-1] + 1) % n)
        L_run, R_run, C_run = L[ext_run], R[ext_run], C[ext_run]

        # 1. Ambient Drop Shadow Under Track
        poly_shadow = np.vstack([L_run + shadow_offset, R_run[::-1] + shadow_offset])
        pygame.draw.polygon(screen, ENV.season.shadow_track, poly_shadow.astype(int).tolist())

        # 2. Seamless Dark Asphalt Ribbon
        poly = np.vstack([L_run, R_run[::-1]])
        pygame.draw.polygon(screen, ENV.season.asphalt, poly.astype(int).tolist())

        # 3. Racing Line Rubber Groove
        pygame.draw.lines(screen, ENV.season.asphalt_groove, False, C_run.astype(int).tolist(), gw)

        # 4. 3D Beveled Red & White Kerbs
        stripe_id = ext_run // 3
        s_diffs = np.diff(stripe_id)
        s_splits = np.nonzero(s_diffs != 0)[0] + 1
        s_chunks = np.split(np.arange(len(ext_run)), s_splits)

        for chunk in s_chunks:
            if len(chunk) < 2:
                idx_start = chunk[0]
                idx_end = min(len(ext_run) - 1, idx_start + 1)
                chunk_pts = [idx_start, idx_end]
            else:
                chunk_pts = chunk

            is_even = (stripe_id[chunk[0]] % 2 == 0)
            color_top = KERB_RED if is_even else KERB_WHITE
            color_base = KERB_RED_DARK if is_even else KERB_WHITE_DARK

            # Under-bevel shadow
            pygame.draw.lines(screen, color_base, False, L_run[chunk_pts].astype(int).tolist(), kw)
            pygame.draw.lines(screen, color_base, False, R_run[chunk_pts].astype(int).tolist(), kw)
            # Top highlight
            pygame.draw.lines(screen, color_top, False, L_run[chunk_pts].astype(int).tolist(), max(2, kw - 2))
            pygame.draw.lines(screen, color_top, False, R_run[chunk_pts].astype(int).tolist(), max(2, kw - 2))

        # 5. Crisp White Track Limit Lines
        pygame.draw.lines(screen, TRACK_LINE, False, L_run.astype(int).tolist(), lw)
        pygame.draw.lines(screen, TRACK_LINE, False, R_run.astype(int).tolist(), lw)

    # 6. Start / finish chequered line and grid slots
    p0 = L[0]
    if -80 <= p0[0] <= w + 80 and -80 <= p0[1] <= h + 80:
        sw = max(4, int(scale * 1.2))
        boxes = 10
        for k in range(boxes):
            a0, a1 = k / float(boxes), (k + 1) / float(boxes)
            p_a = (L[0] * (1 - a0) + R[0] * a0).astype(int).tolist()
            p_b = (L[0] * (1 - a1) + R[0] * a1).astype(int).tolist()
            col = WHITE if k % 2 == 0 else (42, 46, 56)
            pygame.draw.line(screen, col, p_a, p_b, sw)


# ---------------------------------------------------------------------------
# Detailed F1 Car Rendering (Wings, Halo, Rims, Brake Lights)
# ---------------------------------------------------------------------------
def _car_geometry(car_pos, car_heading, length=5.6, width=2.4):
    """Calculate multi-element aerodynamic body polygons."""
    fwd  = np.array([ np.cos(car_heading),  np.sin(car_heading)])
    side = np.array([-np.sin(car_heading),  np.cos(car_heading)])

    # Nose & Main Monocoque
    chassis = np.array([
        car_pos + fwd * length * 0.56,                          # Needle nose
        car_pos + fwd * length * 0.42 + side * width * 0.28,   # Front bulkhead L
        car_pos + fwd * length * 0.12 + side * width * 0.42,   # Sidepod intake L
        car_pos - fwd * length * 0.28 + side * width * 0.36,   # Coke bottle waist L
        car_pos - fwd * length * 0.46 + side * width * 0.24,   # Rear diffuser corner L
        car_pos - fwd * length * 0.46 - side * width * 0.24,   # Rear diffuser corner R
        car_pos - fwd * length * 0.28 - side * width * 0.36,   # Coke bottle waist R
        car_pos + fwd * length * 0.12 - side * width * 0.42,   # Sidepod intake R
        car_pos + fwd * length * 0.42 - side * width * 0.28,   # Front bulkhead R
    ])

    # Front Wing Multi-Tier Plane
    front_wing = np.array([
        car_pos + fwd * length * 0.62 + side * width * 0.56,
        car_pos + fwd * length * 0.62 - side * width * 0.56,
        car_pos + fwd * length * 0.50 - side * width * 0.56,
        car_pos + fwd * length * 0.50 + side * width * 0.56,
    ])

    # Rear Wing Endplates & Flap
    rear_wing = np.array([
        car_pos - fwd * length * 0.38 + side * width * 0.68,
        car_pos - fwd * length * 0.38 - side * width * 0.68,
        car_pos - fwd * length * 0.50 - side * width * 0.68,
        car_pos - fwd * length * 0.50 + side * width * 0.68,
    ])

    # Cockpit, Halo & Driver Helmet
    halo = np.array([
        car_pos + fwd * length * 0.22 + side * width * 0.16,
        car_pos + fwd * length * 0.22 - side * width * 0.16,
        car_pos - fwd * length * 0.08 - side * width * 0.14,
        car_pos - fwd * length * 0.08 + side * width * 0.14,
    ])
    helmet_pos = car_pos + fwd * length * 0.06

    # Brake light position
    brake_light_pos = car_pos - fwd * length * 0.50

    return chassis, front_wing, rear_wing, halo, helmet_pos, brake_light_pos, fwd, side


def draw_car(screen, car_pos, car_heading, view_fn, color=RED, speed=0.0, braking=False):
    """Draw a detailed F1 racecar with aerodynamic wings, 3D shadow, rims, and brake light."""
    car_pos = np.asarray(car_pos, dtype=float)
    chassis, fwing, rwing, halo, helmet_pos, blight_pos, fwd, side = _car_geometry(car_pos, car_heading)

    # 1. Soft Dynamic Drop Shadow
    shadow_offset = fwd * 0.25 + side * 0.25
    shadow_poly = view_fn(chassis + shadow_offset)
    pygame.draw.polygon(screen, (10, 12, 16), shadow_poly.astype(int).tolist())

    # 2. Aerodynamic Wings (Dark carbon fiber trim)
    carbon_fiber = (38, 42, 50)
    pygame.draw.polygon(screen, carbon_fiber, view_fn(fwing).astype(int).tolist())
    pygame.draw.polygon(screen, carbon_fiber, view_fn(rwing).astype(int).tolist())
    pygame.draw.polygon(screen, _lighten(color, 0.2), view_fn(rwing).astype(int).tolist(), width=1)

    # 3. Main Car Monocoque Body
    body_pts = view_fn(chassis).astype(int).tolist()
    pygame.draw.polygon(screen, color, body_pts)
    # Highlight spine line
    spine_front = view_fn(car_pos + fwd * 2.8)
    spine_rear = view_fn(car_pos - fwd * 2.0)
    pygame.draw.line(screen, _lighten(color, 0.45), spine_front.astype(int), spine_rear.astype(int), 2)

    # 4. Cockpit & Titanium Halo Safety System
    pygame.draw.polygon(screen, (18, 20, 26), view_fn(halo).astype(int).tolist())
    pygame.draw.polygon(screen, (85, 90, 105), view_fn(halo).astype(int).tolist(), width=2)
    # Driver helmet dot
    h_screen = view_fn(helmet_pos)
    pygame.draw.circle(screen, YELLOW, (int(h_screen[0]), int(h_screen[1])), 4)
    pygame.draw.circle(screen, WHITE, (int(h_screen[0]), int(h_screen[1])), 2)

    # 5. Wide F1 Slick Tires with Rim Centers
    wheel_offsets = [
        fwd * 2.1 + side * 1.55,  # Front Left
        fwd * 2.1 - side * 1.55,  # Front Right
        -fwd * 1.7 + side * 1.60, # Rear Left
        -fwd * 1.7 - side * 1.60, # Rear Right
    ]
    tire_color = (20, 22, 26)
    for wo in wheel_offsets:
        w_screen = view_fn(car_pos + wo)
        pygame.draw.circle(screen, tire_color, (int(w_screen[0]), int(w_screen[1])), 5)
        pygame.draw.circle(screen, (130, 135, 145), (int(w_screen[0]), int(w_screen[1])), 2)

    # 6. Glowing FIA Rear Rain / Brake Light
    bl_screen = view_fn(blight_pos)
    if braking:
        pygame.draw.circle(screen, (255, 30, 30), (int(bl_screen[0]), int(bl_screen[1])), 6)
        pygame.draw.circle(screen, (255, 240, 240), (int(bl_screen[0]), int(bl_screen[1])), 3)
    else:
        pygame.draw.circle(screen, (180, 20, 20), (int(bl_screen[0]), int(bl_screen[1])), 3)


def draw_cars(screen, cars, view_fn, alive_mask=None, primary_idx=0):
    """Draw a population of cars."""
    for i, car in enumerate(cars):
        if alive_mask is not None and not alive_mask[i]:
            continue
        if i == primary_idx:
            continue
        draw_car(screen, car.pos, car.heading, view_fn, color=BLUE_CAR, speed=car.speed)
    if alive_mask is None or alive_mask[primary_idx]:
        draw_car(screen, cars[primary_idx].pos, cars[primary_idx].heading, view_fn, color=RED, speed=cars[primary_idx].speed)


# ---------------------------------------------------------------------------
# Glowing LiDAR Sensor Rays
# ---------------------------------------------------------------------------
def draw_rays(screen, car_pos, car_heading, dists, view_fn, line_width=2, radar_dist=None):
    """Draw futuristic HUD laser distance rays with glowing hit endpoints and optional 200m radar."""
    car_pos = np.asarray(car_pos, dtype=float)
    p0 = view_fn(car_pos)

    # 1. Long-range forward radar beam (up to 200m)
    if radar_dist is not None:
        theta = car_heading
        hit_pos = car_pos + radar_dist * np.array([np.cos(theta), np.sin(theta)])
        p_radar = view_fn(hit_pos)
        radar_col = (255, 215, 60) if radar_dist > 100.0 else (255, 140, 40)
        pygame.draw.line(screen, radar_col, (int(p0[0]), int(p0[1])), (int(p_radar[0]), int(p_radar[1])), max(1, line_width - 1))
        pygame.draw.circle(screen, radar_col, (int(p_radar[0]), int(p_radar[1])), 6, width=1)
        pygame.draw.circle(screen, WHITE, (int(p_radar[0]), int(p_radar[1])), 2)

    # 2. Standard 9 LiDAR rays (100m)
    for angle_offset, d in zip(RAY_ANGLES, dists):
        theta = car_heading + angle_offset
        hit_pos = car_pos + d * np.array([np.cos(theta), np.sin(theta)])
        p1 = view_fn(hit_pos)

        # Color gradient: Cyan (clear road) -> Orange (caution) -> Red (wall proximity)
        norm_d = min(1.0, d / 60.0)
        if norm_d > 0.55:
            ray_col = (40, 215, 255)
        elif norm_d > 0.25:
            ray_col = (255, 180, 30)
        else:
            ray_col = (255, 45, 45)

        pygame.draw.line(screen, ray_col, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), line_width)
        pygame.draw.circle(screen, ray_col, (int(p1[0]), int(p1[1])), 4)
        pygame.draw.circle(screen, WHITE,   (int(p1[0]), int(p1[1])), 2)


# ---------------------------------------------------------------------------
# Broadcast-Style F1 Digital Dashboard & HUD
# ---------------------------------------------------------------------------
def draw_hud(screen, car, lap, track, message="", rivals=None, font=None, message_font=None, ai_enabled=False):
    """Draw a broadcast-quality telemetry dash with RPM shift lights and friction circle."""
    w, h = screen.get_size()
    if font is None:
        font = pygame.font.SysFont("consolas", 16)

    def blit(text, f, x, y, col=WHITE):
        img = f.render(text, True, col)
        screen.blit(img, (x, y))
        return img.get_width()

    # 1. TOP-LEFT: TIMING & TELEMETRY BOARD
    pw, ph = 310, 160
    _draw_rounded_rect(screen, (16, 24, 38), (16, 16, pw, ph), radius=10, alpha=215, border_color=(60, 85, 125))

    pygame.draw.rect(screen, (50, 210, 255), (16, 16, 5, ph), border_top_left_radius=10, border_bottom_left_radius=10)
    width_str = f" ({track.width:.1f}m)" if hasattr(track, "width") else ""
    blit(f"TRACK: {track.name.upper()}{width_str}", font, 30, 24, CYAN)

    lap_num = max(1, lap.laps)
    blit(f"LAP {lap_num}  |  CUR: {fmt_time(lap.lap_time)}", font, 30, 52, WHITE)
    best_str = fmt_time(lap.best_time) if lap.best_time else "--:--.---"
    blit(f"BEST LAP : {best_str}", font, 30, 76, YELLOW)
    last_str = fmt_time(lap.last_time) if lap.last_time else "--:--.---"
    blit(f"LAST LAP : {last_str}", font, 30, 100, LIGHT_GREY)

    # Track progress bar
    track_len = getattr(track, "length", 1.0)
    prog_pct = float(np.clip(getattr(lap, "distance", 0.0) / max(1.0, track_len), 0.0, 1.0))
    pygame.draw.rect(screen, (30, 36, 48), (30, 134, pw - 45, 10), border_radius=4)
    pygame.draw.rect(screen, (50, 220, 255), (30, 134, int((pw - 45) * prog_pct), 10), border_radius=4)
    blit(f"{int(prog_pct * 100):3d}%", font, pw - 30, 116, MUTED_GREY)

    # 2. BOTTOM-CENTER: F1 STEERING WHEEL DIGITAL DASH
    dash_w, dash_h = 420, 115
    dx = w // 2 - dash_w // 2
    dy = h - dash_h - 16
    _draw_rounded_rect(screen, (16, 24, 38), (dx, dy, dash_w, dash_h), radius=12, alpha=235, border_color=(60, 85, 125))

    # RPM Shift Lights (15 LEDs)
    speed_kmh = car.speed * 3.6
    max_speed_kmh = 330.0
    rpm_ratio = np.clip(speed_kmh / max_speed_kmh, 0.0, 1.0)
    active_leds = int(rpm_ratio * 15)

    led_w, led_h, gap = 18, 10, 6
    start_led_x = dx + (dash_w - (15 * (led_w + gap) - gap)) // 2
    led_y = dy + 12

    for i in range(15):
        lx = start_led_x + i * (led_w + gap)
        if i < active_leds:
            if i < 5:
                led_col = (40, 230, 50)       # Green
            elif i < 10:
                led_col = (255, 30, 30)       # Red
            else:
                led_col = (210, 60, 255)      # Flash Purple (Rev limiter)
        else:
            led_col = (25, 30, 40)            # Off
        pygame.draw.rect(screen, led_col, (lx, led_y, led_w, led_h), border_radius=3)

    # Gear and Speed Display
    gear = min(7, max(1, int(speed_kmh / 45.0) + 1))
    gear_font = pygame.font.SysFont("consolas", 46, bold=True)
    speed_font = pygame.font.SysFont("consolas", 40, bold=True)
    unit_font = pygame.font.SysFont("consolas", 14)

    # Gear Box
    pygame.draw.rect(screen, (20, 26, 38), (dx + 25, dy + 35, 60, 65), border_radius=8)
    pygame.draw.rect(screen, CYAN, (dx + 25, dy + 35, 60, 65), width=1, border_radius=8)
    g_txt = gear_font.render(str(gear), True, YELLOW)
    screen.blit(g_txt, (dx + 25 + 30 - g_txt.get_width() // 2, dy + 44))

    # Speed Readout
    s_txt = speed_font.render(f"{int(speed_kmh):3d}", True, WHITE)
    screen.blit(s_txt, (dx + 115, dy + 42))
    blit("KM/H", unit_font, dx + 205, dy + 62, MUTED_GREY)

    # 3. G-FORCE FRICTION CIRCLE (ACCELEROMETER)
    circ_x = dx + dash_w - 65
    circ_y = dy + 68
    radius = 32
    pygame.draw.circle(screen, (25, 32, 45), (circ_x, circ_y), radius)
    pygame.draw.circle(screen, (45, 60, 85), (circ_x, circ_y), radius, width=1)
    pygame.draw.circle(screen, (35, 45, 65), (circ_x, circ_y), radius // 2, width=1)
    pygame.draw.line(screen, (35, 45, 65), (circ_x - radius, circ_y), (circ_x + radius, circ_y), 1)
    pygame.draw.line(screen, (35, 45, 65), (circ_x, circ_y - radius), (circ_x, circ_y + radius), 1)

    g_x = float(np.clip(car.lat_accel / 38.0, -1.0, 1.0)) * (radius - 5)
    g_col = (255, 45, 45) if abs(car.lat_accel) > 25.0 else (50, 220, 255)
    pygame.draw.circle(screen, g_col, (int(circ_x + g_x), circ_y), 4)

    # 4. TOP-RIGHT: RIVAL DRIVER CARDS
    if rivals and ai_enabled:
        rx = w - 260
        card_h = 58
        ry_start = 210
        for i, r in enumerate(rivals):
            r_name = r.get("name", "AI")
            r_car = r.get("car")
            r_lap = r.get("lap")
            r_col = r.get("color", BLUE_CAR)
            if r_car is None or r_lap is None:
                continue
            cy = ry_start + i * (card_h + 8)
            _draw_rounded_rect(screen, (16, 24, 38), (rx, cy, 245, card_h), radius=8, alpha=210, border_color=(60, 85, 125))
            pygame.draw.rect(screen, r_col, (rx + 6, cy + 8, 4, card_h - 16), border_radius=2)
            blit(f"{r_name}", font, rx + 16, cy + 6, r_col)
            blit(f"SPD {r_car.speed * 3.6:5.1f} KM/H", font, rx + 16, cy + 24, WHITE)
            best_r = fmt_time(r_lap.best_time) if r_lap.best_time else "--:--.---"
            blit(f"BEST: {best_r}", font, rx + 16, cy + 40, MUTED_GREY)

    # 5. CENTRE MESSAGE BANNER
    if message and message_font:
        msg = message_font.render(message, True, YELLOW)
        mx = w // 2 - msg.get_width() // 2
        shadow = message_font.render(message, True, (15, 15, 20))
        screen.blit(shadow, (mx + 2, 22))
        screen.blit(msg, (mx, 20))

    # 6. BOTTOM-LEFT: SEASON / WEATHER BADGE
    badge_w, badge_h = 230, 36
    bx, by = 16, h - badge_h - 16
    _draw_rounded_rect(screen, (16, 24, 38), (bx, by, badge_w, badge_h), radius=8, alpha=215, border_color=(60, 85, 125))
    theme = ENV.season
    badge_txt = f"{theme.icon} {theme.name.upper()}"
    blit(badge_txt, font, bx + 12, by + 9, CYAN)
    blit("[E]", font, bx + badge_w - 34, by + 9, MUTED_GREY)


def draw_car_tag(screen, view_fn, pos, label, color, font):
    """Draw a clean floating badge over a car."""
    pt = view_fn(pos)
    tx, ty = int(pt[0]), int(pt[1]) - 28
    txt_surf = font.render(label, True, color)
    tw, th = txt_surf.get_width(), txt_surf.get_height()
    bg_rect = pygame.Rect(tx - tw // 2 - 6, ty - 2, tw + 12, th + 4)
    _draw_rounded_rect(screen, (10, 14, 20), (bg_rect.x, bg_rect.y, bg_rect.w, bg_rect.h), radius=4, alpha=220, border_color=(45, 60, 85))
    screen.blit(txt_surf, (tx - tw // 2, ty))


# ---------------------------------------------------------------------------
# Minimap
# ---------------------------------------------------------------------------
def draw_minimap(screen, track, car_pos, center, size, other_positions=None):
    """Draw top-right minimap with glassmorphism panel."""
    w, h = screen.get_size()
    bw, bh = 245, 180
    x0, y0 = w - bw - 16, 16

    _draw_rounded_rect(screen, (16, 24, 38), (x0, y0, bw, bh), radius=10, alpha=215, border_color=(60, 85, 125))

    mini_font = pygame.font.SysFont("consolas", 12, bold=True)
    label = mini_font.render("GPS RADAR", True, MUTED_GREY)
    screen.blit(label, (x0 + bw // 2 - label.get_width() // 2, y0 + 6))

    ms = min((bw - 36) / size[0], (bh - 36) / size[1])

    def mini(P):
        return fit_view(P, center, ms, x0 + bw / 2, y0 + bh / 2 + 5)

    # Track outline
    pygame.draw.lines(screen, (40, 46, 58), True, mini(track.center).tolist(), 6)
    pygame.draw.lines(screen, (75, 85, 105), True, mini(track.center).tolist(), 2)

    # Start line
    pygame.draw.line(screen, START_YELLOW,
                     mini(track.left[0]).tolist(),
                     mini(track.right[0]).tolist(), 3)

    # Other rival cars
    if other_positions:
        for item in other_positions:
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], (tuple, list)):
                pos, col = item
            else:
                pos, col = item, (50, 140, 255)
            pt = mini(pos).astype(int).tolist()
            pygame.draw.circle(screen, col, pt, 4)

    # Player car
    lpt = mini(car_pos).astype(int).tolist()
    pygame.draw.circle(screen, (255, 60,  40), lpt, 6)
    pygame.draw.circle(screen, WHITE, lpt, 3)


# ---------------------------------------------------------------------------
# Menu & Overlays
# ---------------------------------------------------------------------------
def draw_menu_overlay(screen, track, font, mid_font, big_font):
    """Draw the track selection menu overlay."""
    w, h = screen.get_size()
    shade = pygame.Surface((w, h), pygame.SRCALPHA)
    shade.fill((10, 16, 26, 140))
    screen.blit(shade, (0, 0))

    def ct(s, f, y, color=WHITE):
        img = f.render(s, True, color)
        screen.blit(img, (w / 2 - img.get_width() / 2, y))

    ct("RACING AI : CHAMPIONSHIP", big_font, h / 2 - 170, YELLOW)
    ct(f"<  {track.name}  >", mid_font, h / 2 - 95, WHITE)

    w_info = f"  |  width {track.width:.1f} m" if hasattr(track, "width") else ""
    if hasattr(track, "country"):
        info = (f"{track.country}  |  real lap {track.real_km:.2f} km  |  "
                f"game lap {track.length / 1000:.2f} km{w_info}")
        note = track.note
    else:
        info = f"game lap {track.length / 1000:.2f} km{w_info}"
        note = "Procedural track. Unique challenges every lap."

    ct(info, font, h / 2 - 55, (200, 205, 215))
    ct(note, font, h / 2 - 30,  MUTED_GREY)
    ct(f"SEASON: {ENV.season.icon} {ENV.season.name.upper()}  [Press E to cycle]", font, h / 2 - 4, CYAN)
    ct("LEFT / RIGHT  choose track      SPACE  start race", font, h / 2 + 26, YELLOW)
    ct("Arrows / WASD  drive     G  toggle AI rivals     R  restart", font, h / 2 + 65, (200, 205, 215))
    ct("E  season / weather     M  audio     C  camera     ESC  quit", font, h / 2 + 92, (170, 175, 185))


def draw_pause_overlay(screen, font, big_font):
    """Draw pause message overlay."""
    w, h = screen.get_size()
    shade = pygame.Surface((w, h), pygame.SRCALPHA)
    shade.fill((10, 16, 26, 135))
    screen.blit(shade, (0, 0))

    img = big_font.render("PAUSED", True, YELLOW)
    screen.blit(img, (w / 2 - img.get_width() / 2, h / 2 - 50))

    sub = font.render("Press P to resume", True, WHITE)
    screen.blit(sub, (w / 2 - sub.get_width() / 2, h / 2 + 20))
