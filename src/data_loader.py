"""Data loading and preprocessing for the Food-101 dataset.

Owners: Madhur Modi (Primary), Srikar Josyam (Co-owner)
Scope note: this module only ingests and preprocesses images. It never
computes nutritional values, calorie estimates, or health claims.
"""

import os
import sys

import tensorflow as tf
import tensorflow_datasets as tfds

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    DATA_DIR,
    IMAGE_SIZE,
    SEED,
    SLICE_CLASS_COUNT,
    TFDS_DATA_DIR,
    ensure_dir,
    save_class_names,
    slice_mode_enabled,
)

tf.keras.utils.set_random_seed(SEED)


def build_preprocess_fn():
    """Resize to 224x224 and cast to float32 on the raw [0, 255] scale.

    Normalization is deliberately left to the model. The baseline CNN learns
    its own scaling, and the transfer backbones apply their own input
    conventions (see ``src.train_transfer``), so normalizing here would make
    the same tensor correct for one backbone and wrong for the other.
    """
    target_height, target_width = IMAGE_SIZE

    def preprocess(image, label):
        image = tf.image.resize(image, (target_height, target_width))
        return tf.cast(image, tf.float32), label

    return preprocess


def build_augment_fn():
    """Apply augmentation layers used for the training stream only."""
    flip = tf.keras.layers.RandomFlip("horizontal")
    rotate = tf.keras.layers.RandomRotation(0.1)
    zoom = tf.keras.layers.RandomZoom(0.1)

    def augment(image, label):
        image = flip(image)
        image = rotate(image)
        image = zoom(image)
        return image, label

    return augment


def subset_class_names(class_names, subset_classes_count):
    """Return the first ``subset_classes_count`` class names."""
    return list(class_names[:subset_classes_count])


def filter_dataset(dataset, class_ids):
    """Keep only examples whose label is in ``class_ids``."""
    class_tensor = tf.constant(class_ids, dtype=tf.int64)

    def filter_classes(image, label):
        return tf.math.reduce_any(tf.math.equal(label, class_tensor))

    return dataset.filter(filter_classes)


def remap_labels(dataset, class_ids):
    """Reindex labels to contiguous values starting at zero."""
    class_tensor = tf.constant(class_ids, dtype=tf.int64)

    def remap_label(image, label):
        matches = tf.cast(tf.math.equal(label, class_tensor), tf.int32)
        return image, tf.argmax(matches)

    return dataset.map(remap_label, num_parallel_calls=tf.data.AUTOTUNE)


def get_food101_dataset(
    subset_classes_count=None,
    batch_size=32,
    shuffle_buffer=2000,
    persist_class_names=True,
    cache_mode="auto",
    augment_training=True,
):
    """Load Food-101 and return prepared train/validation pipelines.

    Args:
        subset_classes_count: Number of leading classes to keep. When None
            the full 101-class dataset is used unless ``SLICE_MODE`` is set.
        batch_size: Batch size for both pipelines.
        shuffle_buffer: Shuffle buffer size for the training stream.
        persist_class_names: Persist the labels for the inference application.
        cache_mode: ``"auto"`` uses memory cache for subsets and disables the
            large full-dataset cache; ``"memory"`` always uses memory; ``"disk"``
            uses a project-local disk cache; ``"none"`` disables caching.
        augment_training: Apply the training-only augmentation pipeline.

    Returns:
        Tuple of ``(train_ds, val_ds, class_names, num_classes)``.
    """
    if slice_mode_enabled() and subset_classes_count is None:
        subset_classes_count = SLICE_CLASS_COUNT
        print(
            f"WARNING: SLICE_MODE active - training on a "
            f"{subset_classes_count}-class subset."
        )

    tfds_data_dir = TFDS_DATA_DIR
    ensure_dir(tfds_data_dir)

    # TFDS can leave a partially extracted Food-101 archive behind if a
    # previous download/extraction is interrupted. A subsequent tfds.load()
    # may then find the archive but fail later while generating examples
    # (for example, when food-101/meta/train.json is missing). On Windows this
    # is particularly easy to encounter after a long extraction. Retry once
    # from a clean TFDS directory instead of requiring manual cache repair.
    try:
        dataset, info = tfds.load(
            "food101",
            data_dir=tfds_data_dir,
            as_supervised=True,
            with_info=True,
        )
    except FileNotFoundError as exc:
        missing_path = str(exc)
        if "food-101" not in missing_path or "train.json" not in missing_path:
            raise
        print(
            "WARNING: Food-101 TFDS cache appears incomplete "
            f"(missing: {missing_path}). Clearing {tfds_data_dir!r} "
            "and retrying the download once."
        )
        import shutil

        shutil.rmtree(tfds_data_dir, ignore_errors=True)
        ensure_dir(tfds_data_dir)
        dataset, info = tfds.load(
            "food101",
            data_dir=tfds_data_dir,
            as_supervised=True,
            with_info=True,
        )
    all_class_names = list(info.features["label"].names)
    total_classes = info.features["label"].num_classes

    train_split = dataset["train"]
    val_split = dataset["validation"]

    if subset_classes_count is not None:
        if not 0 < subset_classes_count <= total_classes:
            raise ValueError(
                f"subset_classes_count must be between 1 and {total_classes}, "
                f"got {subset_classes_count}"
            )
        class_ids = list(range(subset_classes_count))
        train_split = filter_dataset(train_split, class_ids)
        val_split = filter_dataset(val_split, class_ids)
        train_split = remap_labels(train_split, class_ids)
        val_split = remap_labels(val_split, class_ids)
        class_names = subset_class_names(all_class_names, subset_classes_count)
    else:
        class_names = all_class_names

    num_classes = len(class_names)
    if persist_class_names:
        save_class_names(class_names)

    preprocess = build_preprocess_fn()
    train_ds = train_split.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_split.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    if cache_mode not in {"auto", "memory", "disk", "none"}:
        raise ValueError(
            "cache_mode must be one of 'auto', 'memory', 'disk', or 'none'"
        )

    if cache_mode == "auto":
        effective_cache_mode = "memory" if subset_classes_count is not None else "none"
    else:
        effective_cache_mode = cache_mode

    if effective_cache_mode == "memory":
        train_ds = train_ds.cache()
        val_ds = val_ds.cache()
    elif effective_cache_mode == "disk":
        cache_root = ensure_dir(os.path.join(DATA_DIR, "pipeline_cache"))
        cache_suffix = f"_{subset_classes_count or total_classes}"
        train_ds = train_ds.cache(os.path.join(cache_root, f"train{cache_suffix}"))
        val_ds = val_ds.cache(os.path.join(cache_root, f"validation{cache_suffix}"))

    train_ds = train_ds.shuffle(
        shuffle_buffer, seed=SEED, reshuffle_each_iteration=True
    )
    if augment_training:
        train_ds = train_ds.map(build_augment_fn(), num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, class_names, num_classes


def get_unbatched_validation(subset_classes_count=None):
    """Return validation images and labels for targeted visual inspection.

    Food-101 TFDS does not expose original local image file paths through the
    ``as_supervised=True`` API, so this function returns image arrays instead of
    pretending a source path exists.
    """
    tfds_data_dir = TFDS_DATA_DIR
    ensure_dir(tfds_data_dir)
    try:
        dataset, info = tfds.load(
            "food101",
            data_dir=tfds_data_dir,
            as_supervised=True,
            with_info=True,
        )
    except FileNotFoundError as exc:
        missing_path = str(exc)
        if "food-101" not in missing_path or "train.json" not in missing_path:
            raise
        print(
            "WARNING: Food-101 TFDS cache appears incomplete "
            f"(missing: {missing_path}). Clearing {tfds_data_dir!r} "
            "and retrying the download once."
        )
        import shutil

        shutil.rmtree(tfds_data_dir, ignore_errors=True)
        ensure_dir(tfds_data_dir)
        dataset, info = tfds.load(
            "food101",
            data_dir=tfds_data_dir,
            as_supervised=True,
            with_info=True,
        )
    val_split = dataset["validation"]

    if subset_classes_count is not None:
        total_classes = info.features["label"].num_classes
        if not 0 < subset_classes_count <= total_classes:
            raise ValueError(
                f"subset_classes_count must be between 1 and {total_classes}, "
                f"got {subset_classes_count}"
            )
        class_ids = list(range(subset_classes_count))
        val_split = remap_labels(filter_dataset(val_split, class_ids), class_ids)
        class_names = subset_class_names(
            list(info.features["label"].names), subset_classes_count
        )
    else:
        class_names = list(info.features["label"].names)

    preprocess = build_preprocess_fn()
    val_ds = val_split.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    records = []
    for image, label in tfds.as_numpy(val_ds):
        records.append({"image": image, "label": int(label)})

    return records, class_names


def prepare_cache_dir(path):
    """Create and return the local TFDS download directory."""
    return ensure_dir(path)


if __name__ == "__main__":
    from src.config import DATA_DIR

    prepare_cache_dir(DATA_DIR)

    print("Verifying the Food-101 data pipeline...")
    pipeline_train, pipeline_val, names, count = get_food101_dataset(
        subset_classes_count=SLICE_CLASS_COUNT
    )

    print(f"Classes in use: {count}")
    print(f"Class names: {names}")

    for batch_images, batch_labels in pipeline_train.take(1):
        print(f"Train batch shape: {batch_images.shape}")
        print(f"Train labels shape: {batch_labels.shape}")
        print(
            f"Label range: [{int(batch_labels.numpy().min())}, "
            f"{int(batch_labels.numpy().max())}]"
        )
        values = batch_images.numpy()
        print(f"Pixel range: [{values.min():.1f}, {values.max():.1f}]")

    for batch_images, _ in pipeline_val.take(1):
        print(f"Validation batch shape: {batch_images.shape}")

    print("Data loader verification passed.")
