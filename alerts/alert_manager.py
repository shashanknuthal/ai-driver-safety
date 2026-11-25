import time
import threading


class AlertManager:
    def __init__(self, cfg):
        self.cfg = cfg
        self.last_alert = 0.0
        self.cooldown = cfg["alerts"]["cooldown_seconds"]
        self.voice_enabled = cfg["alerts"]["enable_voice"]
        self.beep_enabled = cfg["alerts"]["enable_beep"]

    def _voice(self, message):
        if not self.voice_enabled:
            return
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.say(message)
            engine.runAndWait()
        except Exception as e:
            print(f"[voice unavailable] {e}")

    def _beep(self):
        if not self.beep_enabled:
            return
        try:
            import winsound
            winsound.Beep(1000, 500)
        except Exception:
            print("\a", end="", flush=True)

    def update(self, risk):
        now = time.time()
        if now - self.last_alert < self.cooldown:
            return

        status = risk["status"]
        if status == "CRITICAL":
            self.last_alert = now
            print("CRITICAL DROWSINESS")
            threading.Thread(
                target=self._voice,
                args=("Critical drowsiness detected. Please stop and take a break.",),
                daemon=True
            ).start()
            self._beep()

        elif status == "DROWSY":
            self.last_alert = now
            print("DROWSINESS DETECTED")
            threading.Thread(
                target=self._voice,
                args=("You appear drowsy. Please stay alert.",),
                daemon=True
            ).start()
            self._beep()

        elif status == "DISTRACTED":
            self.last_alert = now
            print("DRIVER DISTRACTION DETECTED")
            threading.Thread(
                target=self._voice,
                args=("Please keep your attention on the road.",),
                daemon=True
            ).start()
