"""Shared configuration and paths for the BiteID food classifier."""

import json
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# TensorFlow Datasets creates intermediate extraction paths that include the
# source archive name. On Windows, keeping TFDS under a deeply nested Git
# checkout can therefore exceed the traditional MAX_PATH limit (WinError 206).
# Allow an explicit override, but default to a short Windows path so the
# project works without requiring users to change Windows policy settings.
if os.environ.get("TFDS_DATA_DIR"):
    TFDS_DATA_DIR = os.environ["TFDS_DATA_DIR"]
elif os.name == "nt":
    TFDS_DATA_DIR = r"C:\tfds_food101"
else:
    TFDS_DATA_DIR = os.path.join(DATA_DIR, "tfds")

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
EVAL_DIR = os.path.join(PROJECT_ROOT, "evaluation_results")

CLASS_NAMES_PATH = os.path.join(MODELS_DIR, "class_names.json")
BASELINE_MODEL_PATH = os.path.join(MODELS_DIR, "best_baseline_model.keras")
TRANSFER_MODEL_PATH = os.path.join(MODELS_DIR, "best_transfer_model.keras")

APP_DIR = os.path.join(PROJECT_ROOT, "app")
SAMPLE_IMAGES_DIR = os.path.join(APP_DIR, "sample_images")

IMAGE_SIZE = (224, 224)
NUM_CLASSES = 101
SLICE_CLASS_COUNT = 10

SEED = 42


def slice_mode_enabled():
    """Return True when the SLICE_MODE environment toggle is active."""
    return os.environ.get("SLICE_MODE", "False").strip().lower() in {"1", "true", "yes"}


def ensure_dir(path):
    """Create a directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)
    return path


def save_class_names(class_names):
    """Persist class names so inference does not require the dataset."""
    ensure_dir(MODELS_DIR)
    with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as handle:
        json.dump(list(class_names), handle, indent=2)
    return CLASS_NAMES_PATH


def load_class_names():
    """Load class names persisted by the training pipeline.

    Inference should not download or inspect the training dataset. A missing
    label artifact is therefore treated as a deployment error rather than
    silently triggering a multi-gigabyte TFDS download.
    """
    if not os.path.exists(CLASS_NAMES_PATH):
        raise FileNotFoundError(
            f"Class-name artifact not found at {CLASS_NAMES_PATH}. "
            "Run a training command before starting the inference app."
        )

    with open(CLASS_NAMES_PATH, encoding="utf-8") as handle:
        class_names = json.load(handle)

    if not isinstance(class_names, list) or not class_names:
        raise ValueError(f"Invalid class-name artifact: {CLASS_NAMES_PATH}")
    return class_names
