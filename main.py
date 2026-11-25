import time
import cv2
import yaml

from features.face_processor import FaceProcessor
from features.risk_engine import RiskEngine
from alerts.alert_manager import AlertManager
from database.db import SafetyDB
from utils.fps import FPSCounter


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    cap = cv2.VideoCapture(cfg["camera"]["index"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["camera"]["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["camera"]["height"])

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check camera permissions/index.")

    processor = FaceProcessor(cfg)
    risk_engine = RiskEngine(cfg)
    alerts = AlertManager(cfg)
    db = SafetyDB(cfg["database"]["path"])
    fps_counter = FPSCounter()

    print("Driver safety system started. Press 'q' to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera frame unavailable.")
                break

            frame = cv2.flip(frame, 1)
            result = processor.process(frame)
            risk = risk_engine.update(result)

            alerts.update(risk)

            db.log_if_event(risk)

            annotated = processor.draw(frame, result, risk)
            fps = fps_counter.update()
            cv2.putText(
                annotated, f"FPS: {fps:.1f}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2
            )

            cv2.imshow("AI Driver Safety System", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        processor.close()
        db.close()


if __name__ == "__main__":
    main()
