import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "open-closed-eyes-dataset")

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

IMAGE_SIZE = (32, 32)

RANDOM_SEED = 42

MAX_SAMPLES_PER_CLASS = 3000