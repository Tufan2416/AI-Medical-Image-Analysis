# ============================================================

# src/model.py — FINAL CLEAN WORKING VERSION

# ============================================================

import os
import tensorflow as tf
from tensorflow.keras import layers, optimizers
from tensorflow.keras.applications import MobileNetV2
from src.utils import get_logger, load_config

logger = get_logger(__name__)

# ── Backbone Dictionary ──────────────────────────────────────

BACKBONES = {
"MobileNetV2": MobileNetV2
}

# ── Build Model ──────────────────────────────────────────────

def build_model(
    num_classes=2,
    img_size=(224, 224),
    backbone_name="MobileNetV2",
    dropout_rate=0.3,
    dense_units=128,
    learning_rate=0.0001
):

    cfg = load_config()
    input_shape = (*img_size, 3)

    logger.info(f"Loading {backbone_name} backbone (imagenet weights)...")

    BackboneClass = BACKBONES.get(backbone_name, MobileNetV2)

    base_model = BackboneClass(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )

    base_model.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    logger.info("Model built successfully")
    return model

# ── Callbacks ────────────────────────────────────────────────

def get_callbacks(model_save_path="models/best_model.keras", log_dir="logs"):

    import os
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3
        ),

        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_save_path,
            save_best_only=True,
            monitor="val_accuracy"
        ),

        tf.keras.callbacks.TensorBoard(
            log_dir=log_dir
        )
    ]


# ── Model Summary ────────────────────────────────────────────

def print_model_summary(model):
    print("\n" + "=" * 50)
    model.summary()
    print("=" * 50)

# ── Load Model ───────────────────────────────────────────────

def load_trained_model(model_path):


    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not found. Run training first.")

    return tf.keras.models.load_model(model_path)

