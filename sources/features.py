import cv2
import numpy as np


def extract_pixel_features(images):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    features = []

    for img in images:
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img_eq = clahe.apply(img.astype(np.uint8))
        img_norm = img_eq.astype(np.float32) / 255.0
        features.append(img_norm.flatten())

    return np.array(features)