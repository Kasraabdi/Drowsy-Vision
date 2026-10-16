import numpy as np


def normalize_pixels(X):
    return X / 255.0


def flatten_images(X):
    num_samples = X.shape[0]
    return X.reshape(num_samples, -1)


def extract_pixel_features(X):
    X_norm = normalize_pixels(X)
    return flatten_images(X_norm)

if __name__ == "__main__":
    dummy_data = np.random.randint(0, 256, size=(10, 32, 32), dtype=np.uint8)
    features = extract_pixel_features(dummy_data)
    print(f"Original shape: {dummy_data.shape}")
    print(f"Extracted features shape: {features.shape}")
    print(f"Min value: {np.min(features)}, Max value: {np.max(features)}")