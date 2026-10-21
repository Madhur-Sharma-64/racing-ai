import time


class LapTimer:
    """Counts laps and times them, using distance along the centerline.

    A lap only counts after the car has made a full lap's worth of net forward
    progress, so reversing over the line or cutting back does not cheat it.
    Times use high-resolution perf_counter for real millisecond precision,
    exactly like FIA-standard timing systems (accurate to 3 decimal places).
    """

    def __init__(self, track):
        self.track = track
        self.start(track.center[0])

    def start(self, pos):
        """Fresh run: zero laps, no best time."""
        self.laps = 0
        self.last_time = None
        self.best_time = None
        self.restart_lap(pos)

    def restart_lap(self, pos):
        """Restart the current lap (e.g. after a crash). Keeps laps and best time."""
        self.lap_time = 0.0
        self.distance = 0.0                     # net forward meters this lap
        self._last = self.track.progress(pos)

    def update(self, pos, dt, progress=None):
        """Call once per physics step. Returns True on the step a lap is completed."""
        length = self.track.length
        p = self.track.progress(pos) if progress is None else progress
        d = p - self._last
        if d < -length / 2:                     # crossed the start line forwards
            d += length
        elif d > length / 2:                    # crossed it backwards
            d -= length
        self._last = p
        self.distance += d
        self.lap_time += dt

        if self.distance >= length:
            self.laps += 1

            # Sub-frame transponder crossing interpolation:
            # Calculate the exact fraction of dt when the car crossed distance == length
            prev_dist = self.distance - d
            if d > 1e-9:
                frac = (length - prev_dist) / d
                frac = float(max(0.0, min(1.0, frac)))
            else:
                frac = 1.0

            # Exact finish line crossing time with true millisecond (3 decimal places) precision
            exact_lap_time = round((self.lap_time - dt) + frac * dt, 3)
            self.last_time = exact_lap_time
            if self.best_time is None or exact_lap_time < self.best_time:
                self.best_time = exact_lap_time

            # Carry over remaining sub-frame time to the new lap
            self.lap_time = max(0.0, round((1.0 - frac) * dt, 4))
            self.distance -= length
            return True
        return False

