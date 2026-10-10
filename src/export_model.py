"""TFLite export and quantization for the Food-101 classifier.

Owners: Srikar Josyam (Primary), Samuel Pradeep (Co-owner)
Produces mobile-friendly artifacts from the trained Keras model.
"""

import argparse
import os
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import MODELS_DIR, TRANSFER_MODEL_PATH, ensure_dir
from src.data_loader import get_food101_dataset

QUANTIZATION_MODES = ("none", "dynamic", "float16", "int8")


def _representative_dataset(num_samples=200, subset_classes_count=None):
    """Yield real preprocessed images for static-range calibration."""

    def generator():
        train_ds, _, _, _ = get_food101_dataset(
            subset_classes_count=subset_classes_count,
            batch_size=1,
            persist_class_names=False,
            cache_mode="none",
            augment_training=False,
        )
        for index, (images, _) in enumerate(train_ds.take(num_samples)):
            if index >= num_samples:
                break
            yield [images.numpy()]

    return generator


def build_converter(model, quantization, subset_classes_count=None):
    """Configure a TFLite converter for the requested quantization mode."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantization == "none":
        return converter

    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if quantization == "float16":
        converter.target_spec.supported_types = [tf.float16]
    elif quantization == "dynamic":
        # Dynamic-range quantization compresses weights without changing the
        # model input/output dtype and does not require calibration data.
        pass
    elif quantization == "int8":
        converter.representative_dataset = _representative_dataset(
            subset_classes_count=subset_classes_count
        )
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    else:
        raise ValueError(
            f"Unsupported quantization mode '{quantization}'. "
            f"Choose from {QUANTIZATION_MODES}."
        )

    return converter


def convert_to_tflite(
    model_path=TRANSFER_MODEL_PATH,
    output_path=None,
    quantization="none",
    subset_classes_count=None,
):
    """Convert a Keras model to TFLite.

    Returns:
        The serialized TFLite model bytes.
    """
    if quantization not in QUANTIZATION_MODES:
        raise ValueError(
            f"Unsupported quantization mode '{quantization}'. "
            f"Choose from {QUANTIZATION_MODES}."
        )

    ensure_dir(MODELS_DIR)
    output_path = output_path or os.path.join(
        MODELS_DIR, f"food101_{quantization}.tflite"
    )

    model = tf.keras.models.load_model(model_path)
    tflite_bytes = build_converter(
        model, quantization, subset_classes_count=subset_classes_count
    ).convert()

    with open(output_path, "wb") as handle:
        handle.write(tflite_bytes)

    size_mb = len(tflite_bytes) / (1024 * 1024)
    print(f"Saved {output_path} ({size_mb:.2f} MB, quantization={quantization})")
    return tflite_bytes


def evaluate_tflite(tflite_path, subset_classes_count=None, max_samples=250):
    """Score a TFLite model on the validation split.

    Returns:
        A dictionary with accuracy and the number of samples scored.
    """
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    _, val_ds, _, num_classes = get_food101_dataset(
        subset_classes_count=subset_classes_count,
        batch_size=1,
        persist_class_names=False,
        cache_mode="none",
        augment_training=False,
    )

    correct = 0
    total = 0

    for images, labels in val_ds.take(max_samples):
        tensor = images.numpy()
        if np.issubdtype(input_details["dtype"], np.integer):
            scale, zero_point = input_details["quantization"]
            if scale == 0:
                raise ValueError("TFLite input tensor has an invalid quantization scale")
            tensor = np.round(tensor / scale + zero_point)
            info = np.iinfo(input_details["dtype"])
            tensor = np.clip(tensor, info.min, info.max).astype(input_details["dtype"])
        else:
            tensor = tensor.astype(input_details["dtype"])

        if tuple(tensor.shape) != tuple(input_details["shape"]):
            tensor = tensor.reshape(input_details["shape"])

        interpreter.set_tensor(input_details["index"], tensor)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details["index"])

        predicted = int(np.argmax(output, axis=-1).reshape(-1)[0])
        if predicted == int(labels.numpy()[0]):
            correct += 1
        total += 1

    accuracy = correct / total if total else float("nan")
    print(
        f"TFLite accuracy: {accuracy:.4f} ({correct}/{total}, "
        f"{num_classes} classes)"
    )
    return {"accuracy": accuracy, "correct": correct, "total": total}


def main():
    """Command line entry point."""
    parser = argparse.ArgumentParser(description="Export a Food-101 model to TFLite")
    parser.add_argument("--model", default=TRANSFER_MODEL_PATH)
    parser.add_argument("--output", default=None)
    parser.add_argument("--quantization", choices=QUANTIZATION_MODES, default="none")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=250)
    args = parser.parse_args()

    convert_to_tflite(
        args.model,
        args.output,
        args.quantization,
        subset_classes_count=args.subset,
    )

    if args.evaluate:
        path = args.output or os.path.join(
            MODELS_DIR, f"food101_{args.quantization}.tflite"
        )
        evaluate_tflite(
            path,
            subset_classes_count=args.subset,
            max_samples=args.max_samples,
        )


if __name__ == "__main__":
    main()
