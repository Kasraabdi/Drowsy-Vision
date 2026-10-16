
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from sources.config import MODELS_DIR, RANDOM_SEED, MAX_SAMPLES_PER_CLASS
from sources.data_loader import load_images_from_folder
from sources.features import extract_pixel_features


def get_trained_models(X_train, y_train):
    models = {
        "Logistic_Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
        "SVM": SVC(kernel="rbf", C=1.0, probability=True, random_state=RANDOM_SEED),
        "Random_Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
    }

    trained_models = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model

    return trained_models


def evaluate_and_save_best(models, X_val, y_val):
    os.makedirs(MODELS_DIR, exist_ok=True)
    best_name = None
    best_accuracy = 0.0
    best_model = None

    print("\nValidation Results:")
    for name, model in models.items():
        preds = model.predict(X_val)
        acc = accuracy_score(y_val, preds)
        print(f"{name} Accuracy: {acc:.4f}")

        if acc > best_accuracy:
            best_accuracy = acc
            best_name = name
            best_model = model

    best_model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    joblib.dump(best_model, best_model_path)
    print(f"\nBest Model: {best_name} with Accuracy {best_accuracy:.4f}")
    print(f"Saved to: {best_model_path}")

    return best_name, best_accuracy

if __name__ == "__main__":
    print("Loading datasets...")
    X_train_raw, y_train = load_images_from_folder(split="train", max_samples=MAX_SAMPLES_PER_CLASS)
    X_val_raw, y_val = load_images_from_folder(split="val", max_samples=1000)

    print("Extracting features...")
    X_train = extract_pixel_features(X_train_raw)
    X_val = extract_pixel_features(X_val_raw)

    trained_models = get_trained_models(X_train, y_train)
    evaluate_and_save_best(trained_models, X_val, y_val)