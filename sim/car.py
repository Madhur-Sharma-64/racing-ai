import numpy as np


class Car:
    """Simple car. Inputs are throttle f and steer g, both in [-1, 1]:
    f: +1 full gas, -1 full brake     g: +1 right, -1 left
    Units: meters, seconds, radians. Screen y points down, so a positive
    heading change turns the car clockwise on screen (to the right)."""

    WHEELBASE = 3.0        # m
    MAX_STEER = 0.5        # rad, front wheel angle at g = 1
    MAX_ACCEL = 14.0       # m/s^2 at full gas
    MAX_BRAKE = 28.0       # m/s^2 at full brake
    DRAG = 0.005           # quadratic drag -> top speed ~ 53 m/s (190 km/h)
    ROLLING = 0.3          # m/s^2 constant resistance
    MAX_LAT_ACCEL = 25.0   # m/s^2 grip limit, makes the car run wide when too fast

    def __init__(self, pos, heading):
        self.pos = np.array(pos, dtype=float)
        self.heading = float(heading)
        self.speed = 0.0

    def update(self, f, g, dt=0.05):
        f = float(np.clip(f, -1, 1))
        g = float(np.clip(g, -1, 1))

        # longitudinal: gas, brake, drag. No reverse gear.
        accel = f * self.MAX_ACCEL if f > 0 else f * self.MAX_BRAKE
        accel -= self.DRAG * self.speed ** 2 + self.ROLLING
        self.speed = max(0.0, self.speed + accel * dt)

        # steering: bicycle model, yaw rate capped by tire grip
        yaw = self.speed / self.WHEELBASE * np.tan(g * self.MAX_STEER)
        if self.speed > 1e-3:
            cap = self.MAX_LAT_ACCEL / self.speed
            yaw = float(np.clip(yaw, -cap, cap))
        self.heading += yaw * dt

        self.pos += self.speed * dt * np.array([np.cos(self.heading), np.sin(self.heading)])
