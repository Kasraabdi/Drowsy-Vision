import os
import cv2 as cv
import joblib
import numpy as np

from sources.config import MODELS_DIR, IMAGE_SIZE
from sources.features import extract_pixel_features


class EyeDetector:
    def __init__(self):
        model_path = os.path.join(MODELS_DIR, "best_model.pkl")
        self.model = joblib.load(model_path)

        cascade_path = "haarcascade_eye.xml"
        if not os.path.exists(cascade_path):
            cascade_path = cv.data.haarcascades + "haarcascade_eye.xml"

        self.eye_detector = cv.CascadeClassifier(cascade_path)

    def process_frame(self, frame):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        eyes = self.eye_detector.detectMultiScale(gray, 1.3, 5)

        detected_states = []

        for (x, y, w, h) in eyes:
            eye_detected = frame[y:y + h, x:x + w]
            eye_resized = cv.resize(eye_detected, IMAGE_SIZE)
            eye_gray = cv.cvtColor(eye_resized, cv.COLOR_BGR2GRAY)

            features = extract_pixel_features(np.expand_dims(eye_gray, axis=0))
            prediction = self.model.predict(features)[0]
            detected_states.append(prediction)

            label_text = "Open" if prediction == 1 else "Closed"
            frame = cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv.putText(frame, label_text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return frame, detected_states