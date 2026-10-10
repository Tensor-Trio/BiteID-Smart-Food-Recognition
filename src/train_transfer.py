"""Transfer learning training pipeline for Food-101 classification.

Owners: Srikar Josyam (Primary), Madhur Modi (Co-owner)
Two-pass strategy: frozen-backbone feature extraction, then low-learning-rate
fine-tuning of the upper backbone blocks.
"""

import argparse
import os
import shutil
import sys

import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import MODELS_DIR, TRANSFER_MODEL_PATH, ensure_dir
from src.data_loader import get_food101_dataset
from src.train_baseline import build_callbacks, save_history

BACKBONES = {
    "EfficientNetB0": tf.keras.applications.EfficientNetB0,
    "MobileNetV2": tf.keras.applications.MobileNetV2,
}

# The data pipeline emits float32 tensors on the raw [0, 255] pixel scale and
# leaves backbone-specific normalization to the model. The two backbones do
# not agree on an input convention, so the conversion is declared here:
#
#   EfficientNetB0 - ships with an internal Rescaling(1/255) + Normalization,
#                    so raw [0, 255] input is already correct.
#   MobileNetV2    - ships with no preprocessing and expects [-1, 1], so a
#                    Rescaling layer is added in front of the backbone.
#
# Feeding a single shared normalization to both would silently corrupt one of
# them, so assert_backbone_input_convention() guards against a future Keras
# release changing these defaults.
BACKBONE_HAS_INTERNAL_RESCALING = {
    "EfficientNetB0": True,
    "MobileNetV2": False,
}

FINE_TUNE_LAYERS = 20
FINE_TUNE_LEARNING_RATE = 1e-5


def load_backbone(name, input_shape):
    """Instantiate a pretrained backbone without its classification head."""
    if name not in BACKBONES:
        raise ValueError(
            f"Unsupported backbone '{name}'. Choose from {sorted(BACKBONES)}."
        )
    return BACKBONES[name](
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )


def assert_backbone_input_convention(backbone, base_model):
    """Fail loudly if Keras changed a backbone's built-in preprocessing."""
    has_rescaling = any(
        isinstance(layer, tf.keras.layers.Rescaling) for layer in base_model.layers
    )
    expected = BACKBONE_HAS_INTERNAL_RESCALING[backbone]
    if has_rescaling != expected:
        raise RuntimeError(
            f"{backbone} input convention changed in this Keras version: "
            f"internal Rescaling layer present={has_rescaling}, expected "
            f"{expected}. Update BACKBONE_HAS_INTERNAL_RESCALING and the "
            f"preprocessing in build_transfer_model() to match."
        )
    return has_rescaling


def build_transfer_model(
    num_classes, input_shape=(224, 224, 3), backbone="EfficientNetB0"
):
    """Build the transfer classifier on a frozen ImageNet backbone.

    Returns:
        Tuple of ``(model, base_model)``.
    """
    base_model = load_backbone(backbone, input_shape)
    assert_backbone_input_convention(backbone, base_model)
    base_model.trainable = False

    inputs = tf.keras.Input(shape=input_shape, name="image")

    x = inputs
    if not BACKBONE_HAS_INTERNAL_RESCALING[backbone]:
        # Map [0, 255] to [-1, 1] for backbones that expect that convention.
        x = tf.keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1.0, name="rescale")(
            x
        )

    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_pool")(x)
    x = tf.keras.layers.Dropout(0.3, name="dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="scores")(
        x
    )

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base_model


def unfreeze_top_layers(base_model, num_layers=FINE_TUNE_LAYERS):
    """Freeze every layer except the top ``num_layers``.

    Returns:
        Number of layers left trainable.
    """
    if num_layers <= 0:
        base_model.trainable = False
        return 0

    num_layers = min(num_layers, len(base_model.layers))
    base_model.trainable = True
    for layer in base_model.layers[:-num_layers]:
        layer.trainable = False
    return num_layers


def train_transfer(
    epochs_feature_extraction=15,
    epochs_fine_tuning=10,
    subset_classes_count=None,
    batch_size=32,
    backbone="EfficientNetB0",
    fine_tune_layers=FINE_TUNE_LAYERS,
    steps_per_epoch=None,
    validation_steps=None,
    cache_mode="auto",
):
    """Run feature extraction then fine-tuning, checkpointing the best weights.

    Args:
        cache_mode: Dataset cache strategy.

    Returns:
        The Keras ``History`` object covering the final phase.
    """
    ensure_dir(MODELS_DIR)

    train_ds, val_ds, class_names, num_classes = get_food101_dataset(
        subset_classes_count=subset_classes_count,
        batch_size=batch_size,
        cache_mode=cache_mode,
    )

    print(f"Training {backbone} on {num_classes} classes")
    print(f"Classes: {', '.join(class_names)}")
    model, base_model = build_transfer_model(num_classes=num_classes, backbone=backbone)
    model.summary()

    feature_model_path = os.path.join(
        MODELS_DIR, "best_transfer_feature_extraction.keras"
    )

    print("=" * 60)
    print("Phase 1/2: feature extraction (backbone frozen)")
    print("=" * 60)
    history_fe = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_feature_extraction,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=build_callbacks(feature_model_path),
        verbose=1,
    )

    completed_epochs = len(history_fe.history["loss"])

    print("=" * 60)
    print(
        f"Phase 2/2: fine-tuning (top {fine_tune_layers} layers, "
        f"lr={FINE_TUNE_LEARNING_RATE})"
    )
    print("=" * 60)

    trainable = unfreeze_top_layers(base_model, fine_tune_layers)
    if trainable == 0:
        print("WARNING: nothing left trainable - skipping fine-tuning phase")
        history_ft = None
        shutil.copy2(feature_model_path, TRANSFER_MODEL_PATH)
    else:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_LEARNING_RATE),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        fine_tune_model_path = os.path.join(
            MODELS_DIR, "best_transfer_fine_tuning.keras"
        )
        history_ft = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=completed_epochs + epochs_fine_tuning,
            initial_epoch=completed_epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=build_callbacks(fine_tune_model_path),
            verbose=1,
        )

        best_fe = max(history_fe.history.get("val_accuracy", [float("-inf")]))
        best_ft = max(history_ft.history.get("val_accuracy", [float("-inf")]))
        best_path = (
            fine_tune_model_path if best_ft >= best_fe else feature_model_path
        )
        shutil.copy2(best_path, TRANSFER_MODEL_PATH)

    combined = dict(history_fe.history)
    if history_ft is not None:
        for key, values in history_ft.history.items():
            combined[key] = list(combined.get(key, [])) + list(values)

    save_history(combined, f"{MODELS_DIR}/transfer_history.json")
    print(f"Best weights saved to {TRANSFER_MODEL_PATH}")
    return history_ft if history_ft is not None else history_fe


def main():
    """Command line entry point."""
    parser = argparse.ArgumentParser(description="Train the Food-101 transfer model")
    parser.add_argument("--epochs-fe", type=int, default=15)
    parser.add_argument("--epochs-ft", type=int, default=10)
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--backbone", type=str, default="EfficientNetB0", choices=sorted(BACKBONES)
    )
    parser.add_argument("--fine-tune-layers", type=int, default=FINE_TUNE_LAYERS)
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

    train_transfer(
        epochs_feature_extraction=args.epochs_fe,
        epochs_fine_tuning=args.epochs_ft,
        subset_classes_count=args.subset,
        batch_size=args.batch_size,
        backbone=args.backbone,
        fine_tune_layers=args.fine_tune_layers,
        steps_per_epoch=args.steps_per_epoch,
        validation_steps=args.validation_steps,
        cache_mode=args.cache_mode,
    )


if __name__ == "__main__":
    main()
