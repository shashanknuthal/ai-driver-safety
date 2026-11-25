import math
import cv2
import mediapipe as mp
import numpy as np


# MediaPipe Face Mesh landmark indices used by this prototype.
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 291, 13, 14, 78, 308]


def euclidean(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def eye_aspect_ratio(points):
    # p1,p2,p3,p4,p5,p6 around eye.
    p1, p2, p3, p4, p5, p6 = points
    return (euclidean(p2, p6) + euclidean(p3, p5)) / max(
        2.0 * euclidean(p1, p4), 1e-6
    )


def mouth_aspect_ratio(points):
    # Vertical opening relative to mouth width.
    left, right, top, bottom, upper_side, lower_side = points
    vertical1 = euclidean(top, bottom)
    vertical2 = euclidean(upper_side, lower_side)
    horizontal = euclidean(left, right)
    return (vertical1 + vertical2) / max(2.0 * horizontal, 1e-6)


class FaceProcessor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def _points(self, landmarks, indices, w, h):
        return [(landmarks[i].x * w, landmarks[i].y * h) for i in indices]

    def _head_pose(self, lm, w, h):
        # Approximate pose using normalized landmark geometry.
        # Values are heuristic screen-relative angles for this MVP.
        nose = np.array([lm[1].x * w, lm[1].y * h])
        chin = np.array([lm[152].x * w, lm[152].y * h])
        left_face = np.array([lm[234].x * w, lm[234].y * h])
        right_face = np.array([lm[454].x * w, lm[454].y * h])

        face_center = (left_face + right_face) / 2
        dx = nose[0] - face_center[0]
        face_width = max(np.linalg.norm(right_face - left_face), 1.0)
        yaw = float(np.degrees(np.arctan2(dx, face_width)))

        dy = chin[1] - nose[1]
        face_height = max(abs(dy), 1.0)
        pitch = float(np.degrees(np.arctan2(dy - 0.55 * h, face_height)))

        return yaw, pitch

    def process(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.mesh.process(rgb)

        if not result.multi_face_landmarks:
            return {"face": False}

        lm = result.multi_face_landmarks[0].landmark
        le = self._points(lm, LEFT_EYE, w, h)
        re = self._points(lm, RIGHT_EYE, w, h)
        mouth = self._points(lm, MOUTH, w, h)

        left_ear = eye_aspect_ratio(le)
        right_ear = eye_aspect_ratio(re)
        ear = (left_ear + right_ear) / 2.0
        mar = mouth_aspect_ratio(mouth)
        yaw, pitch = self._head_pose(lm, w, h)

        return {
            "face": True,
            "ear": ear,
            "mar": mar,
            "yaw": yaw,
            "pitch": pitch,
        }

    def draw(self, frame, result, risk):
        out = frame.copy()

        if not result.get("face"):
            cv2.putText(
                out, "NO FACE DETECTED", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2
            )
            return out

        lines = [
            f"EAR: {result['ear']:.3f}",
            f"MAR: {result['mar']:.3f}",
            f"Yaw: {result['yaw']:.1f}",
            f"Pitch: {result['pitch']:.1f}",
            f"Drowsiness risk: {risk['drowsiness_score']:.0f}%",
            f"Distraction: {risk['distraction_score']:.0f}%",
            f"Status: {risk['status']}",
        ]

        y = 70
        for line in lines:
            cv2.putText(
                out, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.72, (255, 255, 255), 2
            )
            y += 32

        if risk["status"] in ("DROWSY", "CRITICAL"):
            cv2.rectangle(out, (0, 0), (out.shape[1] - 1, out.shape[0] - 1),
                           (0, 0, 255), 8)

        return out

    def close(self):
        self.mesh.close()
