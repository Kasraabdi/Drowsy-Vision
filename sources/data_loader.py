import os
import cv2
import numpy as np
from sources.config import DATA_DIR, IMAGE_SIZE, RANDOM_SEED


def load_images_from_folder(split="train", max_samples=None):
    images = []
    labels = []
    categories = {"closed": 0, "open": 1}
    split_dir = os.path.join(DATA_DIR, split)

    np.random.seed(RANDOM_SEED)

    for category, label in categories.items():
        category_dir = os.path.join(split_dir, category)
        if not os.path.exists(category_dir):
            print(f"Warning: Directory not found - {category_dir}")
            continue

        file_names = sorted(os.listdir(category_dir))
        np.random.shuffle(file_names)

        if max_samples is not None and len(file_names) > max_samples:
            file_names = file_names[:max_samples]

        count = 0
        for fname in file_names:
            img_path = os.path.join(category_dir, fname)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img_resized = cv2.resize(img, IMAGE_SIZE)
                images.append(img_resized)
                labels.append(label)
                count += 1

        print(f"Loaded {count} images for class '{category}' from '{split}'")

    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)

    indices = np.arange(len(y))
    np.random.shuffle(indices)

    return X[indices], y[indices]


def load_data_in_batches(split="train", batch_size=3000, total_samples=None):
    X, y = load_images_from_folder(split=split, max_samples=total_samples // 2 if total_samples else None)
    total_len = len(y)

    for start_idx in range(0, total_len, batch_size):
        end_idx = min(start_idx + batch_size, total_len)
        yield X[start_idx:end_idx], y[start_idx:end_idx]


if __name__ == "__main__":
    print("Testing batch data loader...")
    batch_count = 0
    for X_batch, y_batch in load_data_in_batches(split="train", batch_size=3000, total_samples=6000):
        batch_count += 1
        print(f"Batch {batch_count} - Features shape: {X_batch.shape}, Labels shape: {y_batch.shape}")
        print(f"Closed: {np.sum(y_batch == 0)}, Open: {np.sum(y_batch == 1)}")