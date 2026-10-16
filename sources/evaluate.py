
import os
import cv2
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from features import extract_pixel_features


def load_test_images(test_dir, img_size=(32, 32)):
    images = []
    labels = []
    categories = {"closed": 0, "open": 1}

    for category, label in categories.items():
        category_dir = os.path.join(test_dir, category)
        if not os.path.exists(category_dir):
            continue

        for filename in os.listdir(category_dir):
            filepath = os.path.join(category_dir, filename)
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img_resized = cv2.resize(img, img_size)
                images.append(img_resized)
                labels.append(label)

    return np.array(images), np.array(labels)


def evaluate_saved_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    model_path = os.path.join(project_root, "models", "best_model.pkl")
    test_dir = os.path.join(project_root, "data", "open-closed-eyes-dataset", "test")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    print(f"Loading trained model from: {model_path}")
    model = joblib.load(model_path)

    print(f"Loading test images from: {test_dir}")
    X_test_raw, y_test = load_test_images(test_dir)

    if len(X_test_raw) == 0:
        raise ValueError("Test images could not be loaded. Check folder path.")

    print(f"Loaded {len(X_test_raw)} test images.")
    print("Extracting features for test set...")
    X_test = extract_pixel_features(X_test_raw)

    print("Generating predictions...")
    y_pred = model.predict(X_test)

    target_names = ["Closed", "Open"]
    print("\n" + "=" * 50)
    print("TEST EVALUATION REPORT:")
    print("=" * 50)
    print(classification_report(y_test, y_pred, target_names=target_names))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)

    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title("Confusion Matrix - Test Set")

    reports_dir = os.path.join(project_root, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    matrix_save_path = os.path.join(reports_dir, "confusion_matrix.png")
    plt.savefig(matrix_save_path)
    print(f"Confusion matrix saved to: {matrix_save_path}")
    plt.close()


if __name__ == "__main__":
    evaluate_saved_model()