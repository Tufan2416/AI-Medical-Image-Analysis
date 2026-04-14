# ============================================================

# src/preprocess.py — Image Preprocessing Pipeline

# ============================================================

import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.utils.class_weight import compute_class_weight
from src.utils import get_logger, load_config

logger = get_logger(__name__)
# ── Constants from Config ────────────────────────────────────

def load_settings():
    cfg = load_config()
    IMG_SIZE = tuple(cfg["dataset"]["image_size"])
    BATCH = cfg["training"]["batch_size"]
    return cfg, IMG_SIZE, BATCH

def create_data_generators(data_dir: str):

    cfg, IMG_SIZE, BATCH = load_settings()

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    eval_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_gen = train_datagen.flow_from_directory(
        directory=os.path.join(data_dir, "train"),
        target_size=IMG_SIZE,
        batch_size=BATCH,
        class_mode="binary",
        shuffle=True
    )

    val_gen = eval_datagen.flow_from_directory(
        directory=os.path.join(data_dir, "val"),
        target_size=IMG_SIZE,
        batch_size=BATCH,
        class_mode="binary",
        shuffle=False
    )

    test_gen = eval_datagen.flow_from_directory(
        directory=os.path.join(data_dir, "test"),
        target_size=IMG_SIZE,
        batch_size=BATCH,
        class_mode="binary",
        shuffle=False
    )

    logger.info(f"Train: {train_gen.samples} | Val: {val_gen.samples} | Test: {test_gen.samples}")
    logger.info(f"Classes: {train_gen.class_indices}")

    return train_gen, val_gen, test_gen



def compute_weights(train_gen) -> dict:

    labels = train_gen.classes
    classes = np.unique(labels)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels
    )

    weight_dict = dict(zip(classes, weights))
    logger.info(f"Class weights: {weight_dict}")

    return weight_dict

# ── Single Image Preprocessor ─────────────────────────────────

def preprocess_single_image(image_path: str, img_size: tuple = (224, 224)) -> np.ndarray:
    img = load_img(image_path, target_size=img_size, color_mode="rgb")
    arr = img_to_array(img)
    arr = arr / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

# ── PIL Image Preprocessor ────────────────────────────────────

def preprocess_pil_image(pil_image, img_size: tuple = (224, 224)) -> np.ndarray:
    from PIL import Image


    pil_image = pil_image.convert("RGB").resize(img_size)
    arr = np.array(pil_image, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr