class LapTimer:
    """Counts laps and times them, using distance along the centerline.

    A lap only counts after the car has made a full lap's worth of net forward
    progress, so reversing over the line or cutting back does not cheat it.
    Times are simulated seconds (the sum of the dt you pass in).
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

    def update(self, pos, dt):
        """Call once per physics step. Returns True on the step a lap is completed."""
        length = self.track.length
        p = self.track.progress(pos)
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
            self.last_time = self.lap_time
            if self.best_time is None or self.lap_time < self.best_time:
                self.best_time = self.lap_time
            self.lap_time = 0.0
            self.distance -= length
            return True
        return False
