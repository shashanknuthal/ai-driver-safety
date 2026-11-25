import time
from collections import deque
try:
    from models.runtime_model import TemporalDrowsinessModel
except Exception:
    TemporalDrowsinessModel = None


class RiskEngine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.t = cfg["thresholds"]
        self.window = cfg["temporal"]["window_seconds"]

        self.eye_closed_since = None
        self.yawn_since = None
        self.events = deque()

        self.prev_time = time.time()
        self.blink_count = 0
        self.yawn_count = 0
        self.prolonged_closures = 0
        self.prev_eye_closed = False
        self.prev_yawn = False
        self.temporal_model = TemporalDrowsinessModel() if TemporalDrowsinessModel else None

    def _cleanup(self, now):
        cutoff = now - self.window
        while self.events and self.events[0][0] < cutoff:
            self.events.popleft()

    def update(self, f):
        now = time.time()

        if not f.get("face"):
            return {
                "drowsiness_score": 0,
                "distraction_score": 0,
                "status": "NO FACE",
                "eye_closed_duration": 0,
                "yawn_duration": 0,
                "blink_count": self.blink_count,
                "yawn_count": self.yawn_count,
                "event": None,
            }

        ear = f["ear"]
        mar = f["mar"]
        yaw = abs(f["yaw"])
        pitch = f["pitch"]

        eye_closed = ear < self.t["ear_closed"]
        yawn_active = mar > self.t["yawn_mar"]
        head_away = yaw > self.t["head_yaw_away"]
        head_down = pitch > self.t["head_pitch_down"]

        if eye_closed and self.eye_closed_since is None:
            self.eye_closed_since = now
        if not eye_closed and self.eye_closed_since is not None:
            duration = now - self.eye_closed_since
            if duration < self.t["eye_closure_seconds"]:
                self.blink_count += 1
            self.eye_closed_since = None

        if yawn_active and self.yawn_since is None:
            self.yawn_since = now
        if not yawn_active and self.yawn_since is not None:
            duration = now - self.yawn_since
            if duration >= self.t["yawn_seconds"]:
                self.yawn_count += 1
                self.events.append((now, "yawn"))
            self.yawn_since = None

        eye_duration = 0 if self.eye_closed_since is None else now - self.eye_closed_since
        yawn_duration = 0 if self.yawn_since is None else now - self.yawn_since

        if eye_duration >= self.t["eye_closure_seconds"] and not self.prev_eye_closed:
            self.prolonged_closures += 1
            self.events.append((now, "prolonged_eye_closure"))

        self.prev_eye_closed = eye_closed
        self.prev_yawn = yawn_active
        self._cleanup(now)

        # Heuristic MVP score. The later ML model should replace/calibrate this.
        eye_component = min(50.0, (eye_duration / 2.0) * 50.0)
        yawn_component = 20.0 if yawn_duration >= self.t["yawn_seconds"] else 0.0
        head_component = 20.0 if head_down else 0.0
        repetition_component = min(10.0, self.prolonged_closures * 2.0)

        drowsiness = min(
            100.0,
            eye_component + yawn_component + head_component + repetition_component
        )

        distraction = 80.0 if head_away else 0.0
        if head_away and eye_closed:
            distraction = 90.0

        if drowsiness >= self.t["risk_critical"]:
            status = "CRITICAL"
        elif drowsiness >= self.t["risk_warning"]:
            status = "DROWSY"
        elif drowsiness >= 20:
            status = "FATIGUE"
        elif distraction >= 60:
            status = "DISTRACTED"
        else:
            status = "ALERT"

        ml = None
        if self.temporal_model and f.get("face"):
            ml = self.temporal_model.predict([ear, mar, yaw, pitch, eye_duration, yawn_duration, self.blink_count / max(1.0, (now - self.prev_time) / 60.0), self.yawn_count / max(1.0, (now - self.prev_time) / 60.0)])
        event = None
        if status == "CRITICAL":
            event = "critical_drowsiness"
        elif status == "DROWSY":
            event = "drowsiness"
        elif status == "DISTRACTED":
            event = "distraction"

        return {
            "drowsiness_score": drowsiness,
            "distraction_score": distraction,
            "status": status,
            "eye_closed_duration": eye_duration,
            "yawn_duration": yawn_duration,
            "blink_count": self.blink_count,
            "yawn_count": self.yawn_count,
            "event": event,
            "ml_prediction": ml,
        }
