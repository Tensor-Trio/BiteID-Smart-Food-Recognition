"""Evaluation engine for Food-101 classifiers.

Owners: Samuel Pradeep (Primary), Srikar Josyam (Co-owner)
Reports Top-1 / Top-5 accuracy, macro F1, per-class breakdowns, a confusion
matrix, high-confidence failure cases, and inference latency.
"""

import argparse
import json
import os
import sys
import time

import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    top_k_accuracy_score,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import EVAL_DIR, IMAGE_SIZE, ensure_dir
from src.data_loader import get_food101_dataset

HIGH_CONFIDENCE_THRESHOLD = 0.85
TOP_CONFUSED_PAIRS = 5
MAX_FAILURE_IMAGES = 25


def collect_predictions(model, val_ds, failure_threshold=HIGH_CONFIDENCE_THRESHOLD):
    """Run inference while retaining images only for high-confidence failures.

    The Food-101 validation split contains 25,250 images. Keeping every
    224x224 RGB image in RAM can consume several gigabytes, so evaluation
    stores only the small set of failure images needed for error analysis.
    """
    y_true, y_pred, y_proba = [], [], []
    failure_images = {}
    sample_offset = 0

    for batch_images, batch_labels in val_ds:
        probs = model.predict(batch_images, verbose=0)
        labels = batch_labels.numpy()
        predictions = np.argmax(probs, axis=1)
        confidences = probs.max(axis=1)

        y_true.extend(labels)
        y_pred.extend(predictions)
        y_proba.append(probs)

        if len(failure_images) < MAX_FAILURE_IMAGES:
            failure_mask = (labels != predictions) & (confidences > failure_threshold)
            for local_index in np.where(failure_mask)[0]:
                if len(failure_images) >= MAX_FAILURE_IMAGES:
                    break
                global_index = sample_offset + int(local_index)
                failure_images[global_index] = (
                    batch_images[local_index].numpy().clip(0, 255).astype(np.uint8)
                )

        sample_offset += len(labels)

    return (
        np.asarray(y_true),
        np.asarray(y_pred),
        np.concatenate(y_proba, axis=0),
        failure_images,
    )


def compute_metrics(y_true, y_pred, y_proba, num_classes):
    """Compute accuracy, macro F1, and per-class precision/recall/F1."""
    top1_accuracy = accuracy_score(y_true, y_pred)

    k = min(5, num_classes, y_proba.shape[1])
    top5_accuracy = (
        top_k_accuracy_score(y_true, y_proba, k=k) if k >= 1 else float("nan")
    )

    macro_f1 = f1_score(
        y_true, y_pred, average="macro", labels=range(num_classes), zero_division=0
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(num_classes), average=None, zero_division=0
    )

    class_metrics = pd.DataFrame(
        {"precision": precision, "recall": recall, "f1": f1, "support": support},
        index=pd.RangeIndex(num_classes, name="class_index"),
    )

    return {
        "top1_accuracy": float(top1_accuracy),
        "top5_accuracy": float(top5_accuracy),
        "macro_f1": float(macro_f1),
        "class_metrics": class_metrics,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=range(num_classes)),
    }


def find_high_confidence_failures(y_true, y_pred, y_proba, class_names, threshold=None):
    """Return mispredictions the model was still highly confident about."""
    threshold = HIGH_CONFIDENCE_THRESHOLD if threshold is None else threshold

    failures = []
    confidences = y_proba.max(axis=1)
    incorrect = y_true != y_pred

    for index in np.where(incorrect & (confidences > threshold))[0]:
        failures.append(
            {
                "index": int(index),
                "true_index": int(y_true[index]),
                "true_class": class_names[y_true[index]],
                "pred_index": int(y_pred[index]),
                "pred_class": class_names[y_pred[index]],
                "confidence": float(confidences[index]),
            }
        )

    return failures


def summarize_confusion_pairs(y_true, y_pred, class_names, limit=TOP_CONFUSED_PAIRS):
    """Count the most frequent true-to-predicted errors across all samples."""
    counts = {}
    for true_index, pred_index in zip(y_true, y_pred):
        if true_index == pred_index:
            continue
        key = (class_names[true_index], class_names[pred_index])
        counts[key] = counts.get(key, 0) + 1

    ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:limit]
    return [
        {"true_class": true_class, "pred_class": pred_class, "count": count}
        for (true_class, pred_class), count in ranked
    ]


def benchmark_latency(model, batch_size=1, warmup=10, runs=50):
    """Measure average single-inference latency in milliseconds."""
    height, width = IMAGE_SIZE
    sample = np.random.random((batch_size, height, width, 3)).astype(np.float32) * 255.0

    for _ in range(warmup):
        model.predict(sample, verbose=0)

    start = time.perf_counter()
    for _ in range(runs):
        model.predict(sample, verbose=0)
    elapsed = time.perf_counter() - start

    return elapsed / runs * 1000.0


def save_confusion_matrix(matrix, class_names, output_path):
    """Write a confusion matrix heatmap to disk."""
    size = max(6, min(24, len(class_names) * 0.35))
    figure, axes = plt.subplots(figsize=(size, size))

    sns.heatmap(
        matrix,
        cmap="Blues",
        square=True,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"shrink": 0.7},
        figure=figure,
    )
    axes.set_xlabel("Predicted")
    axes.set_ylabel("True")
    axes.set_title("Food-101 Confusion Matrix")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_failure_images(failures, failure_images, output_dir, limit=MAX_FAILURE_IMAGES):
    """Persist the retained high-confidence failure images for inspection."""
    image_dir = ensure_dir(os.path.join(output_dir, "images"))

    saved = 0
    for failure in failures:
        if saved >= limit:
            break
        index = failure["index"]
        image = failure_images.get(index)
        if image is None:
            continue
        safe_true = failure["true_class"].replace(" ", "_")
        safe_pred = failure["pred_class"].replace(" ", "_")
        filename = f"{index:06d}_{safe_true}_as_{safe_pred}.jpg"
        plt.imsave(os.path.join(image_dir, filename), image)
        saved += 1


def evaluate_model(
    model_path,
    subset_classes_count=None,
    batch_size=32,
    output_dir=None,
    cache_mode="auto",
):
    """Evaluate a saved model and write every artifact to ``output_dir``.

    Returns:
        A summary dictionary of headline metrics.
    """
    output_dir = ensure_dir(output_dir or EVAL_DIR)
    model = tf.keras.models.load_model(model_path)

    _, val_ds, class_names, num_classes = get_food101_dataset(
        subset_classes_count=subset_classes_count,
        batch_size=batch_size,
        persist_class_names=False,
        augment_training=False,
        cache_mode=cache_mode,
    )

    y_true, y_pred, y_proba, failure_images = collect_predictions(model, val_ds)
    metrics = compute_metrics(y_true, y_pred, y_proba, num_classes)

    failures = find_high_confidence_failures(y_true, y_pred, y_proba, class_names)
    summary_pairs = summarize_confusion_pairs(y_true, y_pred, class_names)
    latency_ms = benchmark_latency(model)

    metrics["class_metrics"].insert(0, "class", class_names)
    metrics["class_metrics"].to_csv(
        os.path.join(output_dir, "class_metrics.csv"), index=False
    )

    save_confusion_matrix(
        metrics["confusion_matrix"],
        class_names,
        os.path.join(output_dir, "confusion_matrix.png"),
    )

    failure_dir = ensure_dir(os.path.join(output_dir, "failures"))
    with open(
        os.path.join(failure_dir, "high_confidence_failures.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(failures, handle, indent=2)
    save_failure_images(failures, failure_images, failure_dir)

    summary = {
        "model_path": os.path.abspath(model_path),
        "num_classes": num_classes,
        "top1_accuracy": metrics["top1_accuracy"],
        "top5_accuracy": metrics["top5_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "latency_ms_batch1": latency_ms,
        "high_confidence_failure_count": len(failures),
        "top_confused_pairs": summary_pairs,
    }

    with open(
        os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)
    np.save(
        os.path.join(output_dir, "confusion_matrix.npy"), metrics["confusion_matrix"]
    )

    print(f"Top-1 accuracy : {summary['top1_accuracy']:.4f}")
    print(f"Top-5 accuracy : {summary['top5_accuracy']:.4f}")
    print(f"Macro F1       : {summary['macro_f1']:.4f}")
    print(f"Latency (b=1)  : {summary['latency_ms_batch1']:.2f} ms")
    print(f"High-conf fails: {summary['high_confidence_failure_count']}")
    for pair in summary_pairs:
        print(f"  {pair['true_class']} -> {pair['pred_class']}: {pair['count']}")
    print(f"Artifacts written to {output_dir}")

    return summary


def main():
    """Command line entry point."""
    parser = argparse.ArgumentParser(description="Evaluate a Food-101 model")
    parser.add_argument("--model", required=True, help="Path to a .keras model file")
    parser.add_argument("--output", default=EVAL_DIR)
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--cache-mode", choices=["auto", "memory", "disk", "none"], default="auto"
    )
    args = parser.parse_args()

    evaluate_model(
        args.model,
        subset_classes_count=args.subset,
        batch_size=args.batch_size,
        output_dir=args.output,
        cache_mode=args.cache_mode,
    )


if __name__ == "__main__":
    main()
