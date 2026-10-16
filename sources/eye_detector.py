import os
import cv2
import joblib
import numpy as np

from sources.features import extract_pixel_features


class EyeDetector:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, ".."))
        model_path = os.path.join(project_root, "models", "best_model.pkl")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        self.model = joblib.load(model_path)

        face_xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_detector = cv2.CascadeClassifier(face_xml)
        if self.face_detector.empty():
            raise RuntimeError(f"Failed to load cascade classifier from {face_xml}")

        self.image_size = (32, 32)
        self.last_face = None
        self.lost = 0

    def _find_face(self, gray):
        faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))
        if len(faces) > 0:
            self.last_face = max(faces, key=lambda f: f[2] * f[3])
            self.lost = 0
        else:
            self.lost += 1
            if self.lost > 10:
                self.last_face = None
        return self.last_face

    @staticmethod
    def _eye_boxes(face):
        x, y, w, h = face
        ew, eh = int(0.36 * w), int(0.28 * h)
        ey = y + int(0.20 * h)
        return [
            (x + int(0.12 * w), ey, ew, eh),
            (x + int(0.52 * w), ey, ew, eh)
        ]

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = self._find_face(gray)
        detected_states = []

        if face is None:
            return frame, detected_states

        for (x, y, w, h) in self._eye_boxes(face):
            if y < 0 or x < 0 or (y + h) > frame.shape[0] or (x + w) > frame.shape[1]:
                continue

            eye = gray[y:y + h, x:x + w]
            if eye.size == 0:
                continue

            eye_resized = cv2.resize(eye, self.image_size)
            features = extract_pixel_features(np.expand_dims(eye_resized, axis=0))
            prediction = self.model.predict(features)[0]
            detected_states.append(prediction)

            label = "Open" if prediction == 1 else "Closed"
            color = (0, 255, 0) if prediction == 1 else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return frame, detected_states