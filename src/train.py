# ============================================================
# src/train.py — Model Training Pipeline
# AI-Powered Medical Image Analysis System
# ============================================================

import os
import tensorflow as tf
from src.preprocess import create_data_generators, compute_weights
from src.model import build_model, get_callbacks, print_model_summary
from src.utils import (get_logger, load_config, ensure_dirs,
                        plot_training_history, print_dataset_stats,
                        plot_sample_images)

logger = get_logger(__name__)


def train(data_dir: str = "data/raw",
          model_save_path: str = "models/best_model.keras"):
    """
    Full training pipeline:
      1. Load config
      2. Create data generators
      3. Build model
      4. Train with callbacks
      5. Save training plots

    Args:
        data_dir:        Root directory of the dataset
        model_save_path: Where to save the best trained model

    Returns:
        model, history
    """
    # ── Load Config ───────────────────────────────────────
    cfg = load_config()
    IMG_SIZE     = tuple(cfg["dataset"]["image_size"])
    NUM_CLASSES  = cfg["dataset"]["num_classes"]
    EPOCHS       = cfg["training"]["epochs"]
    LR           = cfg["training"]["learning_rate"]
    BACKBONE     = cfg["model"]["backbone"]
    DROPOUT      = cfg["model"]["dropout_rate"]
    DENSE_UNITS  = cfg["model"]["dense_units"]
    BATCH_SIZE   = cfg["training"]["batch_size"]
    CLASS_LABELS = cfg["dataset"]["classes"]

    ensure_dirs("models", "outputs/plots", "outputs/predictions", "logs")

    logger.info("=" * 55)
    logger.info("   STARTING AI MEDICAL IMAGE ANALYSIS TRAINING")
    logger.info("=" * 55)
    logger.info(f"  Backbone    : {BACKBONE}")
    logger.info(f"  Image size  : {IMG_SIZE}")
    logger.info(f"  Num classes : {NUM_CLASSES}")
    logger.info(f"  Epochs      : {EPOCHS}")
    logger.info(f"  Batch size  : {BATCH_SIZE}")
    logger.info(f"  Dataset     : {data_dir}")
    logger.info("=" * 55)

    # ── Step 1: Data Generators ───────────────────────────
    logger.info("Step 1/5 → Creating data generators...")
    train_gen, val_gen, test_gen = create_data_generators(data_dir)
    print_dataset_stats(train_gen, val_gen, test_gen)
    plot_sample_images(train_gen, CLASS_LABELS)

    # ── Step 2: Class Weights ─────────────────────────────
    logger.info("Step 2/5 → Computing class weights...")
    class_weights = compute_weights(train_gen)

    # ── Step 3: Build Model ───────────────────────────────
    logger.info("Step 3/5 → Building model...")
    model = build_model(
        num_classes=NUM_CLASSES,
        img_size=IMG_SIZE,
        backbone_name=BACKBONE,
        dropout_rate=DROPOUT,
        dense_units=DENSE_UNITS,
        learning_rate=LR
    )
    print_model_summary(model)

    # ── Step 4: Train ─────────────────────────────────────
    logger.info("Step 4/5 → Training...")
    callbacks = get_callbacks(model_save_path=model_save_path)

    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    # ── Step 5: Save Plots ────────────────────────────────
    logger.info("Step 5/5 → Saving training plots...")
    plot_training_history(history)

    # Print final metrics
    final_train_acc = history.history["accuracy"][-1]
    final_val_acc   = history.history["val_accuracy"][-1]
    best_val_acc    = max(history.history["val_accuracy"])

    logger.info("=" * 55)
    logger.info("           TRAINING COMPLETE")
    logger.info("=" * 55)
    logger.info(f"  Final Train Accuracy : {final_train_acc:.4f}")
    logger.info(f"  Final Val Accuracy   : {final_val_acc:.4f}")
    logger.info(f"  Best Val Accuracy    : {best_val_acc:.4f}")
    logger.info(f"  Model saved to       : {model_save_path}")
    logger.info("=" * 55)

    return model, history, test_gen
