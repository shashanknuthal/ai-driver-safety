import time


class FPSCounter:
    def __init__(self):
        self.last = time.perf_counter()
        self.fps = 0.0

    def update(self):
        now = time.perf_counter()
        dt = now - self.last
        self.last = now
        if dt > 0:
            instant = 1.0 / dt
            self.fps = 0.9 * self.fps + 0.1 * instant if self.fps else instant
        return self.fps
