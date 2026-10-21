"""Simulation environment for single-car and multi-car racing.

Compatible with standard reinforcement learning interfaces (reset, step)
and high-speed headless batch simulation for neuroevolution / genetic algorithms.
"""
import numpy as np

from sim.car import Car
from sim.lap import LapTimer
from sim.raycast import cast_rays, RAY_ANGLES
from sim.tracks import load_circuit

# Observation sizes:
#   5 rays → 12-dim (legacy)  |  9 rays → 16-dim (MADX)  |  9 rays + 200m radar + curve_200 → 18-dim (MADX v2)
NUM_RAYS = len(RAY_ANGLES)
MADX_OBS_DIM = 18


def spawn_car(track, lat_frac=0.0):
    """Create a car at the start line aligned with the track tangent with optional lateral offset."""
    offset = float(np.clip(lat_frac, -0.35, 0.35)) * (track.width * 0.5)
    pos = track.center[0] + track.normal[0] * offset
    return Car(pos, np.arctan2(track.tangent[0][1], track.tangent[0][0]))


def build_observation(
    car,
    track,
    rays=None,
    radar_200=None,
    max_range=100.0,
    radar_range=200.0,
    six_dim=False,
    obs_dim=18,
):
    """Build standardized racing observation vector.

    Dimensions supported:
      18-dim (MADX 200m Lookahead): 9 LiDAR rays, 1 forward 200m radar,
                                   speed, lat_pos, heading_err, curve_50, curve_100, curve_200,
                                   yaw_rate, lat_accel
      16-dim (MADX 100m Baseline) : 9 LiDAR rays, speed, lat_pos, heading_err, curve_50, curve_100,
                                   yaw_rate, lat_accel
      12-dim (Legacy 5-ray)       : 5 LiDAR rays, speed, lat_pos, heading_err, curve_50, curve_100,
                                   yaw_rate, lat_accel
      6-dim  (Basic Raycast)      : 5 LiDAR rays, speed
    """
    if six_dim or obs_dim == 6:
        if rays is None:
            all_rays = cast_rays(car.pos, car.heading, track.segments, max_range, midpoints=track.seg_midpoints)
            rays = all_rays[[1, 3, 4, 5, 7]] if len(all_rays) > 5 else all_rays
        return np.append(rays / max_range, car.speed / 50.0).astype(np.float32)

    if obs_dim >= 22:
        if rays is None or radar_200 is None:
            rays, radar_200 = cast_rays(
                car.pos,
                car.heading,
                track.segments,
                max_range,
                midpoints=track.seg_midpoints,
                return_radar=True,
                radar_range=radar_range,
            )
        norm_rays = rays / max_range
        norm_radar = float(radar_200 / radar_range)
        norm_speed = car.speed / 100.0
        lat_pos, angle_rel, curve_50, curve_100, curve_200 = track.get_track_telemetry(
            car.pos, car.heading, lookahead_200=True
        )
        norm_yaw = float(np.clip(car.yaw_rate / 2.0, -1.0, 1.0))
        norm_lat = float(np.clip(car.lat_accel / 40.0, -1.0, 1.0))

        # 4 Universal Geometric Physics Invariants (Track-Agnostic)
        track_w = float(getattr(track, 'width', 12.0))
        half_w = track_w * 0.5
        lat_offset_m = lat_pos * half_w
        margin_left = float(np.clip((half_w - lat_offset_m) / 10.0, 0.0, 2.0))
        margin_right = float(np.clip((half_w + lat_offset_m) / 10.0, 0.0, 2.0))
        d_curvature = float(np.clip((curve_100 - curve_50) * 2.0, -1.0, 1.0))
        norm_track_width = float(np.clip(track_w / 20.0, 0.2, 1.0))

        return np.array([
            *norm_rays,          # 0..8: 9 LiDAR rays (100m)
            norm_radar,          # 9: 200m Long-Range Forward Radar
            norm_speed,          # 10: Normalized Speed
            lat_pos,             # 11: Lateral Offset [-1, 1]
            angle_rel,           # 12: Heading Alignment [-1, 1]
            curve_50,            # 13: Curvature 50m ahead
            curve_100,           # 14: Curvature 100m ahead
            curve_200,           # 15: Curvature 200m ahead
            norm_yaw,            # 16: Normalized Yaw Rate
            norm_lat,            # 17: Normalized Lateral Accel
            norm_track_width,    # 18: Road width (normalised by 20m)
            margin_left,         # 19: Distance to left boundary
            margin_right,        # 20: Distance to right boundary
            d_curvature,         # 21: Curvature derivative (tightening vs opening)
        ], dtype=np.float32)

    if obs_dim == 18:
        if rays is None or radar_200 is None:
            rays, radar_200 = cast_rays(
                car.pos,
                car.heading,
                track.segments,
                max_range,
                midpoints=track.seg_midpoints,
                return_radar=True,
                radar_range=radar_range,
            )
        norm_rays = rays / max_range
        norm_radar = float(radar_200 / radar_range)
        norm_speed = car.speed / 100.0
        lat_pos, angle_rel, curve_50, curve_100, curve_200 = track.get_track_telemetry(
            car.pos, car.heading, lookahead_200=True
        )
        norm_yaw = float(np.clip(car.yaw_rate / 2.0, -1.0, 1.0))
        norm_lat = float(np.clip(car.lat_accel / 40.0, -1.0, 1.0))
        return np.array([
            *norm_rays,       # 0..8: 9 LiDAR rays (100m)
            norm_radar,       # 9: 200m Long-Range Forward Radar
            norm_speed,       # 10: Normalized Speed
            lat_pos,          # 11: Lateral Offset [-1, 1]
            angle_rel,        # 12: Heading Alignment [-1, 1]
            curve_50,         # 13: Curvature 50m ahead
            curve_100,        # 14: Curvature 100m ahead
            curve_200,        # 15: Curvature 200m ahead
            norm_yaw,         # 16: Normalized Yaw Rate
            norm_lat,         # 17: Normalized Lateral Accel
        ], dtype=np.float32)

    # 16-dim or 12-dim fallback
    if rays is None:
        rays = cast_rays(
            car.pos,
            car.heading,
            track.segments,
            max_range,
            midpoints=track.seg_midpoints,
        )
    norm_rays = rays / max_range
    norm_speed = car.speed / 100.0
    lat_pos, angle_rel, curve_50, curve_100 = track.get_track_telemetry(car.pos, car.heading)
    norm_yaw = float(np.clip(car.yaw_rate / 2.0, -1.0, 1.0))
    norm_lat = float(np.clip(car.lat_accel / 40.0, -1.0, 1.0))

    return np.array([
        *norm_rays,
        norm_speed,
        lat_pos,
        angle_rel,
        curve_50,
        curve_100,
        norm_yaw,
        norm_lat,
    ], dtype=np.float32)


class RaceEnv:
    """Headless single-agent racing environment following standard RL conventions.

    Observation space:
        [ray_0..ray_8, speed/100,
         lat_pos, angle_to_tangent, curve_50, curve_100, yaw_rate/2, lat_accel/40]
        (16-dimensional vector with 9 LiDAR rays)

    Action space:
        [throttle, steer] where throttle in [-1, 1], steer in [-1, 1].
    """

    def __init__(self, track=None, dt=0.05, max_range=100.0, max_steps=4000, six_dim=False, obs_dim=None):
        if track is None:
            track = load_circuit("Monza")
        self.track = track
        self.dt = dt
        self.max_range = float(max_range)
        self.max_steps = max_steps
        self.six_dim = six_dim
        self.obs_dim = 6 if six_dim else (int(obs_dim) if obs_dim is not None else MADX_OBS_DIM)

        self.car = spawn_car(self.track)
        self.lap = LapTimer(self.track)
        self.steps = 0
        self.prev_progress = 0.0

    def reset(self, seed=None):
        """Reset the environment to the starting line."""
        self.car = spawn_car(self.track)
        self.lap.start(self.car.pos)
        self.steps = 0
        self.prev_progress = self.track.progress(self.car.pos)

        obs = self._get_obs()
        info = {
            "steps": 0,
            "speed": self.car.speed,
            "progress": self.prev_progress,
            "lap_time": 0.0,
            "laps": 0,
            "best_time": None,
        }
        return obs, info

    def _get_obs(self, rays=None):
        """Compute the normalized observation vector (18-dim default, 16-dim, 12-dim, or 6-dim)."""
        if self.obs_dim == 6 or self.six_dim:
            if rays is not None and len(rays) > 5:
                rays = rays[[1, 3, 4, 5, 7]]
            elif rays is None:
                all_rays = cast_rays(self.car.pos, self.car.heading, self.track.segments, self.max_range, midpoints=self.track.seg_midpoints)
                rays = all_rays[[1, 3, 4, 5, 7]]
            return build_observation(self.car, self.track, rays, self.max_range, six_dim=True, obs_dim=6)
        elif self.obs_dim == 12:
            if rays is not None and len(rays) > 5:
                rays = rays[[1, 3, 4, 5, 7]]
            elif rays is None:
                all_rays = cast_rays(self.car.pos, self.car.heading, self.track.segments, self.max_range, midpoints=self.track.seg_midpoints)
                rays = all_rays[[1, 3, 4, 5, 7]]
            return build_observation(self.car, self.track, rays, self.max_range, obs_dim=12)
        elif self.obs_dim == 16:
            return build_observation(self.car, self.track, rays, self.max_range, obs_dim=16)
        return build_observation(self.car, self.track, rays, self.max_range, obs_dim=18)

    def step(self, action):
        """Execute one simulation step.

        Parameters:
            action: (throttle, steer), each in [-1.0, 1.0].

        Returns:
            obs: np.ndarray shape (6,)
            reward: float
            terminated: bool (crash / off track or lap complete)
            truncated: bool (timeout)
            info: dict
        """
        self.steps += 1
        f, g = float(action[0]), float(action[1])

        # Step vehicle physics
        self.car.update(f, g, self.dt)

        # Query track containment and progress
        on_track, progress, _ = self.track.query(self.car.pos)
        rays = cast_rays(
            self.car.pos,
            self.car.heading,
            self.track.segments,
            self.max_range,
            midpoints=self.track.seg_midpoints,
        )

        # Update lap timer
        lap_completed = self.lap.update(self.car.pos, self.dt, progress=progress)

        # Compute forward progress delta (accounting for track wrap-around)
        length = self.track.length
        dp = progress - self.prev_progress
        if dp < -length / 2:
            dp += length
        elif dp > length / 2:
            dp -= length
        self.prev_progress = progress

        # Reward shaping
        reward = dp  # reward net meters progressed
        terminated = False
        truncated = self.steps >= self.max_steps

        if not on_track:
            terminated = True
            reward -= 50.0  # crash penalty
        elif lap_completed:
            terminated = True
            reward += 100.0  # lap completion bonus

        obs = self._get_obs(rays)
        info = {
            "steps": self.steps,
            "speed": self.car.speed,
            "progress": progress,
            "lap_time": self.lap.lap_time,
            "laps": self.lap.laps,
            "best_time": self.lap.best_time,
            "on_track": on_track,
            "lap_completed": lap_completed,
            "raw_rays": rays,
        }
        return obs, float(reward), terminated, truncated, info


class MultiCarEnv:
    """Headless batch environment simulating N cars in lockstep.

    Designed for fast generational evaluation in genetic algorithms / neuroevolution.

    When `madx_mode=True`, activates the Cubic Time-Attack fitness function:
      - Massive cubic bonuses for fast laps (sub-40s target)
      - Aggressive speed incentives throughout the lap
      - Penalty-free crashing (crash just = no more progress, natural selection)
    """

    def __init__(self, track=None, num_cars=50, dt=0.05, max_range=100.0,
                 max_steps=2000, obs_dim=None, madx_mode=False):
        if track is None:
            track = load_circuit("Monza")
        self.track = track
        self.num_cars = num_cars
        self.dt = dt
        self.max_range = float(max_range)
        self.max_steps = max_steps
        self.madx_mode = madx_mode

        # Observation dimension: explicitly specified, or defaults based on mode
        if obs_dim is not None:
            self.obs_dim = obs_dim
        else:
            self.obs_dim = MADX_OBS_DIM if madx_mode else (NUM_RAYS + 7)

        lat_spread = np.linspace(-0.25, 0.25, num_cars)
        self.cars = [spawn_car(self.track, lat_frac=lat_spread[i]) for i in range(num_cars)]
        self.laps = [LapTimer(self.track) for _ in range(num_cars)]
        self.alive = np.ones(num_cars, dtype=bool)
        self.fitness = np.zeros(num_cars, dtype=float)
        self.prev_progress = np.zeros(num_cars, dtype=float)
        self.prev_steer = np.zeros(num_cars, dtype=float)
        self.steps = 0
        self.best_lap_time = None
        self.total_laps = 0

    def reset(self):
        """Reset all cars to the start line."""
        lat_spread = np.linspace(-0.25, 0.25, self.num_cars)
        self.cars = [spawn_car(self.track, lat_frac=lat_spread[i]) for i in range(self.num_cars)]
        self.laps = [LapTimer(self.track) for _ in range(self.num_cars)]
        self.alive = np.ones(self.num_cars, dtype=bool)
        self.fitness = np.zeros(self.num_cars, dtype=float)
        self.prev_steer = np.zeros(self.num_cars, dtype=float)
        p0 = self.track.progress(self.track.center[0])
        self.prev_progress.fill(p0)
        self.steps = 0
        return self.get_observations()

    def get_observations(self):
        """Return (num_cars, obs_dim) observation matrix for all alive cars."""
        obs = np.zeros((self.num_cars, self.obs_dim), dtype=np.float32)
        for i, car in enumerate(self.cars):
            if not self.alive[i]:
                continue
            obs[i] = build_observation(car, self.track, max_range=self.max_range, obs_dim=self.obs_dim)
        return obs

    def _fitness_standard(self, i, car, dp, lap_done):
        """Standard fitness shaping: progress + speed bonus + lap completion bonus."""
        self.fitness[i] += max(0.0, dp)

        # Speed bonus: reward going fast continuously
        self.fitness[i] += car.speed * 0.002

        # Apex Speed & Grip Bonus:
        # Carrying high speed through corners (rewarding the wide racing line)
        if abs(car.lat_accel) > 12.0:
            self.fitness[i] += car.speed * 0.003

        if lap_done:
            lap_t = self.laps[i].last_time
            # Exponential lap bonus: baseline 500 + speed bonus
            # Reference: ~30s lap scores ~+2000, ~60s lap scores ~+500 (baseline)
            speed_bonus = max(0.0, 1500.0 * np.exp(-lap_t / 40.0))
            self.fitness[i] += 500.0 + speed_bonus

    def _fitness_madx(self, i, car, dp, lap_done, steer, steer_diff, c50, c100=0.0, c200=0.0, lat_pos=0.0, throttle=0.0):
        """MADX Precision Racing Fitness with F1 Racing Line and Trail Braking:
        - Primary: Clean track progression (meters made forward).
        - Straight-Line Stability: On straights (|c50| < 0.10), penalize steering twitching and reward centered wheel.
          In corners (|c50| >= 0.10), permit 100% steering flick agility for linked esses and chicanes without penalty.
        - Corner Apex Momentum: Reward carrying rolling momentum through curved sections instead of over-braking.
        - F1 Geometric Racing Line (Outside-Inside-Outside):
          * Entry: reward positioning toward the outside kerb when approaching a turn (|c100| > 0.20, lat_pos * c100 < -0.12).
          * Apex: reward clipping the inside kerb at peak lateral G (|c50| > 0.18, |lat_accel| > 12, lat_pos * c50 > 0.15).
        - Trail Braking: reward controlled, progressive brake release while turning into high-speed corners.
        - Lap Completion: Massive multiplier for finishing clean fast laps.
        """
        # 1. Forward progression (core driving signal)
        self.fitness[i] += max(0.0, dp)

        # 2. Straight-Line Stability: on straights only, suppress twitching and reward centered wheel
        if abs(c50) < 0.10 and dp > 0.0:
            if steer_diff > 0.30:
                self.fitness[i] -= (steer_diff - 0.30) * 0.05
            self.fitness[i] += max(0.0, 1.0 - abs(steer) * 2.0) * 0.02

        # 3. Corner Momentum & Apex Speed: only when the track is ACTUALLY curved!
        # Prevents cars from swerving on straights, but rewards carrying rolling speed in flowing corners.
        if abs(c50) > 0.15:
            # In flowing curves, reward carrying momentum; in extreme hairpins, cap speed to prioritize grip
            spd_cap = 25.0 if abs(c50) > 0.35 else 50.0
            self.fitness[i] += (min(spd_cap, car.speed) / 40.0) * 0.04
            if abs(car.lat_accel) > 10.0:
                self.fitness[i] += (car.speed / 50.0) * 0.04

        # 4. F1 Racing Line: Geometric Apexing (Outside-Inside-Outside)
        # Entry preparation: wide arc on approach
        if abs(c100) > 0.22 and abs(c50) < 0.15 and car.speed > 25.0:
            if (lat_pos * c100) < -0.12:  # positioned on opposite (outside) kerb
                self.fitness[i] += 0.02
        # Apex clipping: hugging the inside kerb at high lateral G
        if abs(c50) > 0.18 and abs(car.lat_accel) > 12.0:
            if (lat_pos * c50) > 0.15:   # positioned on inside kerb
                self.fitness[i] += min(1.0, abs(car.lat_accel) / 25.0) * 0.03

        # 5. Trail Braking (Controlled Weight Transfer on Corner Entry)
        # When braking into a corner while turning, reward smooth brake-to-apex pressure
        if throttle < -0.05 and abs(car.lat_accel) > 8.0 and car.speed > 18.0:
            trail_bonus = (-throttle) * (abs(car.lat_accel) / 30.0) * 0.025
            self.fitness[i] += trail_bonus

        # 6. G-G Friction Circle Exploitation: Reward operating at 80-99% of maximum tire grip envelope
        g_tot = float(np.hypot(abs(car.accel), abs(car.lat_accel)))
        if car.speed > 16.0 and g_tot > 14.0:
            grip_ratio = min(1.0, g_tot / 32.0)
            self.fitness[i] += grip_ratio * 0.035

        # 7. Corner Exit Rocket Multiplier: Reward pinning 100% throttle immediately as corner opens up
        if abs(c50) < 0.10 and throttle > 0.85 and car.speed > 25.0:
            self.fitness[i] += (car.speed / 50.0) * 0.035

        # 8. Clean Driving Baseline: slight reward for moving forward smoothly
        if dp > 0.0:
            self.fitness[i] += min(1.0, car.speed / 40.0) * 0.01

        # 9. Lap Completion: Massive multiplier for finishing clean laps
        if lap_done:
            lap_t = self.laps[i].last_time
            time_bonus = max(0.0, 3500.0 * np.exp(-lap_t / 40.0))
            self.fitness[i] += 1500.0 + time_bonus

    def step(self, actions):
        """Step all alive cars with their respective (throttle, steer) actions.

        Fitness shaping mode selected by self.madx_mode:
          - False: Standard (progress + speed + exponential lap bonus)
          - True:  MADX Precision Racing (smooth steering + corner-gated apex + F1 line + trail braking)

        actions: array-like of shape (num_cars, 2)
        """
        self.steps += 1
        actions = np.asarray(actions, dtype=float)
        length = self.track.length

        for i, car in enumerate(self.cars):
            if not self.alive[i]:
                continue
            f, g = actions[i, 0], actions[i, 1]

            steer_diff = abs(g - self.prev_steer[i])
            self.prev_steer[i] = g

            # Track curvature and lateral telemetry ahead
            lat_pos, _, c50, c100, c200 = self.track.get_track_telemetry(car.pos, car.heading, lookahead_200=True)

            car.update(f, g, self.dt)

            on_track, progress, _ = self.track.query(car.pos)
            if not on_track:
                self.alive[i] = False
                if self.madx_mode:
                    self.fitness[i] -= 350.0  # Crash penalty: weed out late-brakers that crash into walls
                continue

            dp = progress - self.prev_progress[i]
            if dp < -length / 2:
                dp += length
            elif dp > length / 2:
                dp -= length
            self.prev_progress[i] = progress

            lap_done = self.laps[i].update(car.pos, self.dt, progress=progress)

            # Dispatch to the appropriate fitness function
            if self.madx_mode:
                self._fitness_madx(i, car, dp, lap_done, g, steer_diff, c50, c100, c200, lat_pos, f)
            else:
                self._fitness_standard(i, car, dp, lap_done)

            if lap_done:
                self.total_laps += 1
                lap_t = self.laps[i].last_time
                if self.best_lap_time is None or lap_t < self.best_lap_time:
                    self.best_lap_time = lap_t
                if self.madx_mode:
                    self.alive[i] = False  # Lap finished cleanly; retire car to save compute

        all_dead = not np.any(self.alive)
        truncated = self.steps >= self.max_steps
        done = all_dead or truncated
        return self.get_observations(), self.fitness.copy(), done, {
            "steps": self.steps,
            "best_lap_time": self.best_lap_time,
            "total_laps": self.total_laps,
        }
