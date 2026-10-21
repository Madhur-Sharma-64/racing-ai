"""Dynamic Environment & Seasonal Weather System for Racing AI.

Supports 4 rich seasons:
  - Spring: Cherry Blossom (Sakura) trees with vibrant pink floral canopies shedding
    fluttering pink flower petals onto the asphalt, roadside, and racecars.
  - Autumn: Warm golden-amber terrain with fiery maple & oak trees shedding golden and
    crimson leaves in the autumn breeze.
  - Cold: Frosted alpine ground, icy tarmac, and snow-dusted evergreen conifer pines with
    gently falling snowflakes and frost motes.
  - Summer: Classic European motorsport summer with lush emerald green turf and sunlit canopies.

Provides:
  - Procedural roadside tree generation adhering strictly outside track limits.
  - Dynamic shedding particle simulation with wind flutter, aerodynamic car wake swirls,
    and petals settling/falling over the cars.
  - Season-specific ground and track color palettes.
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any
import numpy as np
import pygame


# ---------------------------------------------------------------------------
# Season Themes & Color Tokens
# ---------------------------------------------------------------------------
@dataclass
class SeasonTheme:
    id: str
    name: str
    icon: str
    desc: str
    # Palette
    terrain: Tuple[int, int, int]
    terrain_stripe: Tuple[int, int, int]
    shadow_track: Tuple[int, int, int]
    asphalt: Tuple[int, int, int]
    asphalt_groove: Tuple[int, int, int]
    # Trees
    tree_type: str
    trunk_color: Tuple[int, int, int]
    foliage_shadow: Tuple[int, int, int]
    foliage_mid: Tuple[int, int, int]
    foliage_highlight: Tuple[int, int, int]
    foliage_accent: Tuple[int, int, int]
    # Particles (Shedding leaves / petals / snow)
    particle_name: str
    particle_colors: List[Tuple[int, int, int]]
    particle_size_range: Tuple[float, float]
    particle_count_ambient: int
    wind_speed: float


SEASONS: Dict[str, SeasonTheme] = {
    "spring": SeasonTheme(
        id="spring",
        name="Spring Sakura",
        icon="🌸",
        desc="Blooming cherry blossom trees shedding soft pink petals across the track and cars",
        terrain=(62, 138, 70),
        terrain_stripe=(72, 152, 82),
        shadow_track=(34, 78, 42),
        asphalt=(65, 68, 76),
        asphalt_groove=(48, 50, 58),
        tree_type="sakura",
        trunk_color=(75, 48, 42),
        foliage_shadow=(215, 95, 145),      # Deep rose shadow
        foliage_mid=(255, 175, 205),        # Main cherry blossom pink
        foliage_highlight=(255, 225, 240),  # Soft white-pink highlight
        foliage_accent=(240, 130, 170),     # Floral accent cluster
        particle_name="sakura_petal",
        particle_colors=[
            (255, 182, 210),
            (255, 198, 222),
            (255, 220, 238),
            (245, 145, 185),
            (255, 240, 248),
        ],
        particle_size_range=(0.35, 0.75),   # in meters
        particle_count_ambient=180,
        wind_speed=4.5,
    ),
    "autumn": SeasonTheme(
        id="autumn",
        name="Golden Autumn",
        icon="🍂",
        desc="Warm golden turf and fiery maple trees shedding swirling amber & crimson leaves",
        terrain=(112, 92, 50),
        terrain_stripe=(126, 104, 58),
        shadow_track=(60, 48, 26),
        asphalt=(62, 60, 64),
        asphalt_groove=(46, 44, 48),
        tree_type="autumn",
        trunk_color=(68, 48, 35),
        foliage_shadow=(185, 48, 22),       # Dark russet shadow
        foliage_mid=(235, 105, 30),         # Fiery maple orange
        foliage_highlight=(248, 192, 45),   # Golden amber highlight
        foliage_accent=(210, 65, 25),       # Crimson accent
        particle_name="autumn_leaf",
        particle_colors=[
            (235, 100, 30),
            (245, 185, 40),
            (205, 55, 25),
            (225, 140, 35),
            (180, 42, 20),
        ],
        particle_size_range=(0.40, 0.85),
        particle_count_ambient=180,
        wind_speed=6.5,
    ),
    "cold": SeasonTheme(
        id="cold",
        name="Alpine Cold",
        icon="❄️",
        desc="Frosted mountain circuit, icy asphalt, and snow-dusted pines with drifting snowflakes",
        terrain=(108, 130, 135),
        terrain_stripe=(122, 146, 152),
        shadow_track=(52, 68, 74),
        asphalt=(56, 62, 72),
        asphalt_groove=(42, 46, 56),
        tree_type="pine",
        trunk_color=(50, 44, 40),
        foliage_shadow=(26, 52, 45),        # Deep alpine conifer
        foliage_mid=(42, 78, 68),           # Frosted pine green
        foliage_highlight=(230, 242, 250),  # Pure snow-capped tips
        foliage_accent=(185, 215, 228),     # Ice sheen accent
        particle_name="snowflake",
        particle_colors=[
            (255, 255, 255),
            (235, 245, 255),
            (215, 235, 250),
            (245, 250, 255),
        ],
        particle_size_range=(0.20, 0.50),
        particle_count_ambient=220,
        wind_speed=3.8,
    ),
    "summer": SeasonTheme(
        id="summer",
        name="Summer GP",
        icon="☀️",
        desc="Lush European motorsport greens, warm asphalt, and sunlit oak canopies",
        terrain=(46, 108, 54),
        terrain_stripe=(52, 120, 62),
        shadow_track=(28, 68, 34),
        asphalt=(64, 68, 78),
        asphalt_groove=(48, 52, 60),
        tree_type="summer",
        trunk_color=(60, 46, 36),
        foliage_shadow=(34, 82, 40),        # Deep forest green
        foliage_mid=(54, 128, 62),          # Lush summer leaf
        foliage_highlight=(84, 175, 90),    # Bright sunlit crown
        foliage_accent=(65, 145, 75),       # Mid-canopy accent
        particle_name="summer_mote",
        particle_colors=[
            (240, 255, 220),
            (255, 250, 210),
            (225, 245, 205),
        ],
        particle_size_range=(0.25, 0.55),
        particle_count_ambient=90,
        wind_speed=3.0,
    ),
}

ORDERED_SEASONS = ["spring", "autumn", "cold", "summer"]


# ---------------------------------------------------------------------------
# Roadside Tree Data Structure
# ---------------------------------------------------------------------------
@dataclass
class RoadsideTree:
    pos: np.ndarray             # (x, y) world meters
    radius: float               # canopy radius (m)
    trunk_radius: float         # trunk base radius (m)
    variation: int              # integer seed for canopy cluster offsets
    shed_cooldown: float = 0.0  # timer for shedding leaves/petals


# ---------------------------------------------------------------------------
# Falling Leaf / Petal Particle
# ---------------------------------------------------------------------------
@dataclass
class SheddingParticle:
    pos: np.ndarray             # (x, y) in world meters
    vel: np.ndarray             # (vx, vy) in world m/s
    size: float                 # size in meters
    color: Tuple[int, int, int]
    life: float                 # remaining lifespan in seconds
    max_life: float
    angle: float                # rotation angle in radians
    spin_rate: float            # rad/s
    flutter_phase: float        # sine wave phase
    flutter_freq: float         # oscillation speed
    flutter_amp: float          # sway amplitude
    stuck_to_car: bool = False  # temporarily resting on a car chassis
    stuck_offset: np.ndarray = field(default_factory=lambda: np.zeros(2))
    stuck_time: float = 0.0


# ---------------------------------------------------------------------------
# Environment Controller
# ---------------------------------------------------------------------------
class EnvironmentManager:
    """Manages seasonal themes, roadside trees, and falling leaf/petal particles."""

    def __init__(self, season_key: str = "spring"):
        self.season_key = season_key if season_key in SEASONS else "spring"
        self.season = SEASONS[self.season_key]
        self.trees: List[RoadsideTree] = []
        self.particles: List[SheddingParticle] = []
        self.sim_time = 0.0
        self.track_bounds = None   # (min_x, max_x, min_y, max_y)

    def set_season(self, season_key: str) -> SeasonTheme:
        """Switch active season and regenerate particle effects."""
        if season_key not in SEASONS:
            season_key = "spring"
        self.season_key = season_key
        self.season = SEASONS[self.season_key]
        self.particles.clear()
        return self.season

    def cycle_season(self) -> SeasonTheme:
        """Cycle to the next season in order."""
        curr_idx = ORDERED_SEASONS.index(self.season_key)
        next_key = ORDERED_SEASONS[(curr_idx + 1) % len(ORDERED_SEASONS)]
        return self.set_season(next_key)

    def build_trees_for_track(self, track, density: float = 1.0):
        """Procedurally generate beautiful groves of trees strictly along the roadside."""
        self.trees.clear()
        self.particles.clear()

        n_pts = len(track.center)
        if n_pts < 10:
            return

        tangents = track.tangent
        # 2D Normal vectors pointing outward left and right
        # Left normal = (-ty, tx), Right normal = (ty, -tx)
        normals_left = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)
        normals_right = -normals_left

        track_length = track.length
        # Step every ~18 to 28 meters along the track
        step_pts = max(3, int(22.0 / 4.0))  # approx every 22m

        rng = np.random.RandomState(42)  # Deterministic placement per track
        min_x, max_x = np.min(track.center[:, 0]), np.max(track.center[:, 0])
        min_y, max_y = np.min(track.center[:, 1]), np.max(track.center[:, 1])
        self.track_bounds = (min_x - 50.0, max_x + 50.0, min_y - 50.0, max_y + 50.0)

        # Place trees along left and right verges
        for i in range(0, n_pts, step_pts):
            p_left = track.left[i]
            p_right = track.right[i]
            n_l = normals_left[i]
            n_r = normals_right[i]

            # Left side tree or grove
            if rng.rand() < 0.75 * density:
                dist = rng.uniform(5.5, 16.0)
                pos = p_left + n_l * dist
                radius = rng.uniform(4.5, 8.5)
                self.trees.append(RoadsideTree(
                    pos=pos,
                    radius=radius,
                    trunk_radius=rng.uniform(0.9, 1.4),
                    variation=int(rng.randint(0, 10000)),
                ))
                # Occasional companion tree for natural clustering
                if rng.rand() < 0.40:
                    c_offset = rng.uniform(4.0, 9.0) * n_l + rng.uniform(-5.0, 5.0) * tangents[i]
                    self.trees.append(RoadsideTree(
                        pos=pos + c_offset,
                        radius=rng.uniform(3.5, 6.5),
                        trunk_radius=rng.uniform(0.7, 1.1),
                        variation=int(rng.randint(0, 10000)),
                    ))

            # Right side tree or grove
            if rng.rand() < 0.75 * density:
                dist = rng.uniform(5.5, 16.0)
                pos = p_right + n_r * dist
                radius = rng.uniform(4.5, 8.5)
                self.trees.append(RoadsideTree(
                    pos=pos,
                    radius=radius,
                    trunk_radius=rng.uniform(0.9, 1.4),
                    variation=int(rng.randint(0, 10000)),
                ))
                if rng.rand() < 0.40:
                    c_offset = rng.uniform(4.0, 9.0) * n_r + rng.uniform(-5.0, 5.0) * tangents[i]
                    self.trees.append(RoadsideTree(
                        pos=pos + c_offset,
                        radius=rng.uniform(3.5, 6.5),
                        trunk_radius=rng.uniform(0.7, 1.1),
                        variation=int(rng.randint(0, 10000)),
                    ))

        # Seed initial ambient falling petals / leaves
        self._seed_ambient_particles()

    def _seed_ambient_particles(self):
        """Pre-populate ambient drifting leaves/petals around trees."""
        if not self.trees:
            return
        rng = np.random
        theme = self.season
        for _ in range(theme.particle_count_ambient):
            tree = self.trees[rng.randint(len(self.trees))]
            angle = rng.uniform(0, 2 * np.pi)
            dist = rng.uniform(0.5, tree.radius * 2.2)
            pos = tree.pos + np.array([np.cos(angle), np.sin(angle)]) * dist
            col = theme.particle_colors[rng.randint(len(theme.particle_colors))]
            sz = rng.uniform(theme.particle_size_range[0], theme.particle_size_range[1])
            life = rng.uniform(2.0, 8.0)
            self.particles.append(SheddingParticle(
                pos=pos,
                vel=np.array([rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5)]),
                size=sz,
                color=col,
                life=life,
                max_life=life,
                angle=rng.uniform(0, 2 * np.pi),
                spin_rate=rng.uniform(-2.5, 2.5),
                flutter_phase=rng.uniform(0, 2 * np.pi),
                flutter_freq=rng.uniform(2.0, 4.5),
                flutter_amp=rng.uniform(0.8, 1.8),
            ))

    def update(self, dt: float, cars: List[Any] = None):
        """Update wind, tree shedding, flutter physics, and car aerodynamic wakes."""
        self.sim_time += dt
        theme = self.season
        rng = np.random

        # Dynamic ambient wind vector with slow oscillation
        wind_base_angle = 0.4 + 0.25 * np.sin(self.sim_time * 0.12)
        wind_mag = theme.wind_speed * (0.85 + 0.35 * np.sin(self.sim_time * 0.45))
        wind_vec = np.array([np.cos(wind_base_angle), np.sin(wind_base_angle)]) * wind_mag

        # Reference position for proximity culling (first car or origin)
        ref_pos = cars[0].pos if (cars and cars[0] is not None) else np.zeros(2)
        SHED_RADIUS_SQ = 120.0 * 120.0  # Only shed from trees within 120m of player

        # 1. Tree shedding: ONLY nearby trees spawn new falling petals/leaves
        max_particles = theme.particle_count_ambient
        for tree in self.trees:
            tree.shed_cooldown -= dt
            if tree.shed_cooldown <= 0.0:
                tree.shed_cooldown = float(rng.uniform(0.8, 2.5))
                # Skip trees far from the camera/player
                dx = tree.pos[0] - ref_pos[0]
                dy = tree.pos[1] - ref_pos[1]
                if dx * dx + dy * dy > SHED_RADIUS_SQ:
                    continue
                if len(self.particles) < max_particles:
                    off_ang = rng.uniform(0, 2 * np.pi)
                    off_r = rng.uniform(0.2, tree.radius * 0.9)
                    spawn_pos = tree.pos + np.array([np.cos(off_ang), np.sin(off_ang)]) * off_r
                    col = theme.particle_colors[rng.randint(len(theme.particle_colors))]
                    sz = rng.uniform(theme.particle_size_range[0], theme.particle_size_range[1])
                    life = rng.uniform(4.5, 9.0)
                    self.particles.append(SheddingParticle(
                        pos=spawn_pos,
                        vel=wind_vec * 0.4 + rng.uniform(-0.4, 0.4, size=2),
                        size=sz,
                        color=col,
                        life=life,
                        max_life=life,
                        angle=rng.uniform(0, 2 * np.pi),
                        spin_rate=rng.uniform(-3.0, 3.0),
                        flutter_phase=rng.uniform(0, 2 * np.pi),
                        flutter_freq=rng.uniform(2.5, 5.0),
                        flutter_amp=rng.uniform(1.0, 2.2),
                    ))

        # 2. Update and step existing shedding particles
        alive_particles: List[SheddingParticle] = []
        for p in self.particles:
            p.life -= dt
            if p.life <= 0.0:
                continue

            p.angle += p.spin_rate * dt
            p.flutter_phase += p.flutter_freq * dt
            # Perpendicular flutter to the wind vector
            flutter_offset = np.array([-wind_vec[1], wind_vec[0]]) / (wind_mag + 1e-4) * (
                np.sin(p.flutter_phase) * p.flutter_amp
            )

            # Check car interactions (only for particles close to a car)
            if cars:
                for c in cars:
                    if c is None:
                        continue
                    diff = p.pos - c.pos
                    dist_sq = diff[0] * diff[0] + diff[1] * diff[1]

                    # Skip particles too far from any car (>10m)
                    if dist_sq > 100.0:
                        continue

                    # 1. Aerodynamic wake slipstream swirl within 6.5 meters
                    if dist_sq < 42.25 and c.speed > 8.0:
                        dist = np.sqrt(dist_sq) + 1e-4
                        norm_dir = diff / dist
                        car_fwd = np.array([np.cos(c.heading), np.sin(c.heading)])
                        wake_push = norm_dir * (c.speed * 0.35) * (1.0 - dist / 6.5)
                        wake_drag = car_fwd * (c.speed * 0.25) * (1.0 - dist / 6.5)
                        p.vel += (wake_push + wake_drag) * dt * 8.0
                        p.spin_rate += float(rng.uniform(-8.0, 8.0)) * dt

                    # 2. Shedding directly ON the car body (within 2.2m)
                    if dist_sq < 4.84 and not p.stuck_to_car and rng.rand() < 0.15:
                        p.stuck_to_car = True
                        p.stuck_offset = np.array([rng.uniform(-1.6, 1.6), rng.uniform(-0.8, 0.8)])
                        p.stuck_time = float(rng.uniform(0.4, 1.4))

            if p.stuck_to_car:
                p.stuck_time -= dt
                if cars and cars[0] is not None:
                    c = cars[0]
                    fwd = np.array([np.cos(c.heading), np.sin(c.heading)])
                    side = np.array([-np.sin(c.heading), np.cos(c.heading)])
                    p.pos = c.pos + fwd * p.stuck_offset[0] + side * p.stuck_offset[1]
                if p.stuck_time <= 0.0 or (cars and cars[0] and cars[0].speed > 55.0):
                    p.stuck_to_car = False
                    p.vel = wind_vec + rng.uniform(-1.0, 1.0, size=2)
            else:
                p.vel = p.vel * 0.94 + (wind_vec + flutter_offset) * 0.06
                p.pos += p.vel * dt

            alive_particles.append(p)

        self.particles = alive_particles

    # -----------------------------------------------------------------------
    # Rendering: Trees & Falling Petals
    # -----------------------------------------------------------------------
    def draw_trees(self, screen: pygame.Surface, view_fn, window_size, scale: float):
        """Draw trees with layered canopies, shadows, and season-specific foliage."""
        w, h = window_size
        theme = self.season
        margin = 160

        for tree in self.trees:
            screen_pos = view_fn(tree.pos)
            sx, sy = screen_pos[0], screen_pos[1]
            if not (-margin < sx < w + margin and -margin < sy < h + margin):
                continue

            px_r = max(5, int(tree.radius * scale))
            trunk_r = max(2, int(tree.trunk_radius * scale))

            # 1. Soft Ground Drop Shadow under tree
            shadow_offset = (int(px_r * 0.35), int(px_r * 0.40))
            shadow_surf = pygame.Surface((px_r * 2 + 10, px_r * 2 + 10), pygame.SRCALPHA)
            pygame.draw.ellipse(
                shadow_surf,
                (*theme.shadow_track, 140),
                (5, 5, px_r * 2, int(px_r * 1.5))
            )
            screen.blit(shadow_surf, (int(sx - px_r + shadow_offset[0]), int(sy - px_r * 0.75 + shadow_offset[1])))

            # 2. Trunk Base
            pygame.draw.circle(screen, theme.trunk_color, (int(sx), int(sy)), trunk_r)

            # 3. Multi-Layered Seasonal Foliage / Blossoms
            if theme.tree_type == "sakura":
                self._draw_sakura_canopy(screen, sx, sy, px_r, tree.variation, theme)
            elif theme.tree_type == "pine":
                self._draw_pine_canopy(screen, sx, sy, px_r, tree.variation, theme)
            else:
                self._draw_broadleaf_canopy(screen, sx, sy, px_r, tree.variation, theme)

    def _draw_sakura_canopy(self, screen, sx, sy, r, var, theme):
        """Draw luxurious, blooming cherry blossom (Sakura) canopy with pink petals."""
        # Multi-cluster blossom puff offsets
        sub_clusters = [
            ( 0.00,  0.00, 0.75),   # Core center
            (-0.45, -0.25, 0.55),   # Upper-left
            ( 0.42, -0.30, 0.52),   # Upper-right
            (-0.35,  0.35, 0.50),   # Lower-left
            ( 0.38,  0.32, 0.52),   # Lower-right
            (-0.10, -0.52, 0.45),   # Top crown
            ( 0.55,  0.05, 0.46),   # Right flank
        ]

        # Pass 1: Deep rose under-shadow
        for ox, oy, sr in sub_clusters:
            cx, cy = int(sx + ox * r), int(sy + oy * r)
            cr = max(3, int(r * sr))
            pygame.draw.circle(screen, theme.foliage_shadow, (cx + 2, cy + 2), cr)

        # Pass 2: Main vibrant cherry blossom pink
        for ox, oy, sr in sub_clusters:
            cx, cy = int(sx + ox * r), int(sy + oy * r)
            cr = max(3, int(r * sr))
            pygame.draw.circle(screen, theme.foliage_mid, (cx, cy), cr)

        # Pass 3: Delicate blossom accent rosettes
        for ox, oy, sr in sub_clusters[:4]:
            cx, cy = int(sx + ox * r * 0.7), int(sy + oy * r * 0.7)
            cr = max(2, int(r * sr * 0.7))
            pygame.draw.circle(screen, theme.foliage_accent, (cx, cy), cr)

        # Pass 4: Soft white-pink top highlights
        for ox, oy, sr in sub_clusters:
            cx, cy = int(sx + ox * r - r * 0.12), int(sy + oy * r - r * 0.15)
            cr = max(2, int(r * sr * 0.45))
            pygame.draw.circle(screen, theme.foliage_highlight, (cx, cy), cr)

    def _draw_pine_canopy(self, screen, sx, sy, r, var, theme):
        """Draw snow-dusted evergreen conifer pine."""
        # Tiered conical circles with snow caps
        tiers = [(0.0, 0.3, 0.95), (0.0, 0.0, 0.75), (0.0, -0.35, 0.55), (0.0, -0.65, 0.35)]
        for ox, oy, sr in tiers:
            cx, cy = int(sx + ox * r), int(sy + oy * r)
            cr = max(3, int(r * sr))
            pygame.draw.circle(screen, theme.foliage_shadow, (cx, cy + 2), cr)
            pygame.draw.circle(screen, theme.foliage_mid, (cx, cy), cr)
            # Snow cap crest on top
            snow_r = max(2, int(cr * 0.55))
            pygame.draw.circle(screen, theme.foliage_highlight, (cx, int(cy - cr * 0.35)), snow_r)

    def _draw_broadleaf_canopy(self, screen, sx, sy, r, var, theme):
        """Draw autumn maple or summer oak broadleaf canopy."""
        sub_clusters = [
            ( 0.00,  0.00, 0.80),
            (-0.40, -0.20, 0.58),
            ( 0.40, -0.25, 0.55),
            (-0.30,  0.35, 0.52),
            ( 0.35,  0.30, 0.50),
            ( 0.00, -0.48, 0.45),
        ]
        # Shadow base
        for ox, oy, sr in sub_clusters:
            cx, cy = int(sx + ox * r), int(sy + oy * r)
            cr = max(3, int(r * sr))
            pygame.draw.circle(screen, theme.foliage_shadow, (cx + 2, cy + 3), cr)
        # Mid tone
        for ox, oy, sr in sub_clusters:
            cx, cy = int(sx + ox * r), int(sy + oy * r)
            cr = max(3, int(r * sr))
            pygame.draw.circle(screen, theme.foliage_mid, (cx, cy), cr)
        # Accent & highlight
        for ox, oy, sr in sub_clusters:
            cx, cy = int(sx + ox * r - r * 0.12), int(sy + oy * r - r * 0.14)
            cr = max(2, int(r * sr * 0.42))
            pygame.draw.circle(screen, theme.foliage_highlight, (cx, cy), cr)

    def draw_particles(self, screen: pygame.Surface, view_fn, window_size, scale: float):
        """Draw falling leaves, pink sakura petals, or snowflakes over the track and cars."""
        w, h = window_size
        theme = self.season

        for p in self.particles:
            s_pos = view_fn(p.pos)
            sx, sy = s_pos[0], s_pos[1]
            if not (-10 < sx < w + 10 and -10 < sy < h + 10):
                continue

            px_sz = max(2, int(p.size * scale))
            alpha_ratio = min(1.0, p.life / 0.8) if p.life < 0.8 else 1.0

            if theme.tree_type == "sakura":
                # Draw delicate rotated pink cherry blossom petal
                cos_a, sin_a = np.cos(p.angle), np.sin(p.angle)
                half_w = max(1.5, px_sz * 0.65)
                half_h = max(2.5, px_sz * 1.35)
                # 4-point curved diamond/oval petal polygon
                pts = [
                    (sx + (-half_w * sin_a), sy + (half_w * cos_a)),
                    (sx + (half_h * cos_a),  sy + (half_h * sin_a)),
                    (sx + (half_w * sin_a),  sy + (-half_w * cos_a)),
                    (sx + (-half_h * cos_a * 0.6), sy + (-half_h * sin_a * 0.6)),
                ]
                pygame.draw.polygon(screen, p.color, [(int(x), int(y)) for x, y in pts])
                # Tiny white petal shine
                pygame.draw.circle(screen, (255, 245, 250), (int(sx), int(sy)), max(1, int(px_sz * 0.35)))

            elif theme.tree_type == "cold":
                # Snowflake / frost crystal dot with delicate glow
                r = max(1, int(px_sz * 0.75))
                pygame.draw.circle(screen, p.color, (int(sx), int(sy)), r)
                if r >= 2:
                    pygame.draw.circle(screen, (255, 255, 255), (int(sx), int(sy)), max(1, r - 1))

            else:
                # Autumn leaf: rotated pointed leaf polygon
                cos_a, sin_a = np.cos(p.angle), np.sin(p.angle)
                half_w = max(1.5, px_sz * 0.70)
                half_h = max(2.5, px_sz * 1.40)
                pts = [
                    (sx + (-half_w * sin_a), sy + (half_w * cos_a)),
                    (sx + (half_h * cos_a),  sy + (half_h * sin_a)),
                    (sx + (half_w * sin_a),  sy + (-half_w * cos_a)),
                    (sx + (-half_h * cos_a * 0.7), sy + (-half_h * sin_a * 0.7)),
                ]
                pygame.draw.polygon(screen, p.color, [(int(x), int(y)) for x, y in pts])


# Global Singleton Environment
ENV = EnvironmentManager("spring")
