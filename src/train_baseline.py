"""Baseline CNN training pipeline for Food-101 classification.

Owners: Srikar Josyam (Primary), Madhur Modi (Co-owner)
Purpose: establish a from-scratch convolutional reference point that the
transfer learning model must beat before the capstone comparison holds.
"""

import argparse
import json
import os
import sys

import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import BASELINE_MODEL_PATH, MODELS_DIR, ensure_dir
from src.data_loader import get_food101_dataset


def build_baseline_model(
    num_classes, input_shape=(224, 224, 3), learning_rate=1e-3
):
    """Build the custom CNN baseline: 3 conv blocks plus a dense head.

    The pipeline emits raw [0, 255] float tensors, so the model rescales its
    own input. Without this layer the flatten -> dense stack sees activations
    three orders of magnitude too large and the loss diverges.
    """
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Rescaling(scale=1.0 / 255.0, name="rescale")(inputs)

    x = tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_callbacks(checkpoint_path):
    """Early stopping plus best-weight checkpointing."""
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
        ),
    ]


def save_history(history, path):
    """Persist the training history as JSON."""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                key: [float(value) for value in values]
                for key, values in history.items()
            },
            handle,
            indent=2,
        )


def train_baseline(
    epochs=20,
    subset_classes_count=None,
    batch_size=32,
    learning_rate=1e-3,
    steps_per_epoch=None,
    validation_steps=None,
    cache_mode="auto",
):
    """Train the baseline CNN and checkpoint the best weights.

    Args:
        epochs: Number of passes over the training data.
        subset_classes_count: Restrict to the first N classes when set.
        batch_size: Batch size for both splits.
        learning_rate: Adam learning rate.
        steps_per_epoch: Cap batches per epoch. Use for quick smoke tests.
        validation_steps: Cap validation batches per epoch.
        cache_mode: Dataset cache strategy.

    Returns:
        The Keras ``History`` object.
    """
    ensure_dir(MODELS_DIR)

    train_ds, val_ds, class_names, num_classes = get_food101_dataset(
        subset_classes_count=subset_classes_count,
        batch_size=batch_size,
        cache_mode=cache_mode,
    )

    print(f"Training baseline CNN on {num_classes} classes")
    print(f"Classes: {', '.join(class_names)}")
    model = build_baseline_model(
        num_classes=num_classes, learning_rate=learning_rate
    )
    model.summary()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=build_callbacks(BASELINE_MODEL_PATH),
        verbose=1,
    )

    save_history(history.history, f"{MODELS_DIR}/baseline_history.json")
    print(f"Best weights saved to {BASELINE_MODEL_PATH}")
    return history


def main():
    """Command line entry point."""
    parser = argparse.ArgumentParser(description="Train the Food-101 baseline CNN")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=None,
        help="Cap batches per epoch for quick smoke tests",
    )
    parser.add_argument("--validation-steps", type=int, default=None)
    parser.add_argument(
        "--cache-mode", choices=["auto", "memory", "disk", "none"], default="auto"
    )
    args = parser.parse_args()

    train_baseline(
        epochs=args.epochs,
        subset_classes_count=args.subset,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        steps_per_epoch=args.steps_per_epoch,
        validation_steps=args.validation_steps,
        cache_mode=args.cache_mode,
    )


if __name__ == "__main__":
    main()
