import os
import cv2
import joblib
import numpy as np
from collections import deque

from sources.features import extract_pixel_features


class EyeStateSmoother:
    def __init__(self, window=5, close_th=0.45, open_th=0.55):
        self.probs = deque(maxlen=window)
        self.close_th = close_th
        self.open_th = open_th
        self.state = 1

    def update(self, p_open):
        self.probs.append(p_open)
        avg = sum(self.probs) / len(self.probs)
        if self.state == 1 and avg < self.close_th:
            self.state = 0
        elif self.state == 0 and avg > self.open_th:
            self.state = 1
        return self.state, avg


class EyeDetector:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, ".."))
        model_path = os.path.join(project_root, "models", "best_model.pkl")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        self.model = joblib.load(model_path)
        self.open_idx = list(self.model.classes_).index(1)

        face_xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_detector = cv2.CascadeClassifier(face_xml)
        if self.face_detector.empty():
            raise RuntimeError(f"Failed to load cascade classifier from {face_xml}")

        self.image_size = (32, 32)
        self.last_face = None
        self.smoother = EyeStateSmoother(window=5, close_th=0.45, open_th=0.55)

    def _find_face(self, gray):
        faces = self.face_detector.detectMultiScale(
            gray, scaleFactor=1.15, minNeighbors=7, minSize=(120, 120)
        )
        if len(faces) > 0:
            new_face = max(faces, key=lambda f: f[2] * f[3])
            if self.last_face is None:
                self.last_face = new_face
            else:
                prev = np.array(self.last_face, dtype=float)
                curr = np.array(new_face, dtype=float)
                smoothed = (0.65 * prev) + (0.35 * curr)
                self.last_face = tuple(smoothed.astype(int))
            return self.last_face
        else:
            self.last_face = None
            return None

    @staticmethod
    def _eye_boxes(face):
        x, y, w, h = face
        ew = int(0.24 * w)
        eh = int(0.16 * h)
        ey = y + int(0.34 * h)
        return [
            (x + int(0.19 * w), ey, ew, eh),
            (x + int(0.57 * w), ey, ew, eh),
        ]

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = self._find_face(gray)
        detected_states = []

        if face is None:
            return frame, detected_states

        p_list, boxes = [], []
        for (x, y, w, h) in self._eye_boxes(face):
            if y < 0 or x < 0 or (y + h) > frame.shape[0] or (x + w) > frame.shape[1]:
                continue
            eye = gray[y:y + h, x:x + w]
            if eye.size == 0:
                continue

            eye_resized = cv2.resize(eye, self.image_size)
            features = extract_pixel_features(np.expand_dims(eye_resized, axis=0))
            prob_open = self.model.predict_proba(features)[0][self.open_idx]
            p_list.append(prob_open)
            boxes.append((x, y, w, h))

        if not p_list:
            return frame, detected_states

        state, avg_prob = self.smoother.update(sum(p_list) / len(p_list))
        detected_states = [state] * len(boxes)

        label = f"{'Open' if state == 1 else 'Closed'} ({avg_prob:.2f})"
        color = (0, 255, 0) if state == 1 else (0, 0, 255)

        for (x, y, w, h) in boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        if boxes:
            text_x = boxes[0][0]
            text_y = boxes[0][1] + boxes[0][3] + 22
            cv2.putText(frame, label, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        return frame, detected_states