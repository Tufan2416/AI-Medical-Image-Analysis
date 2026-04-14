# ============================================================

# src/utils.py — Utility Functions

# AI-Powered Medical Image Analysis System

# ============================================================

import os
import yaml
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use

# ── Logger Setup ────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger.
    Use: logger = get_logger(**name**)
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger(name)

# ── Config Loader ────────────────────────────────────────────

def load_config(config_path: str = "config.yaml") -> dict:
    """
    Loads the project YAML config file.
    Returns a dict with all settings.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

# ── Directory Creator ────────────────────────────────────────

def ensure_dirs(*paths):
    """
    Creates directories if they don't exist.
    Usage: ensure_dirs("models/", "outputs/plots/")
    """
    for path in paths:
        os.makedirs(path, exist_ok=True)

# ── Class Label Mapper ───────────────────────────────────────

def get_class_labels(config: dict) -> list:
    """
    Returns list of class labels from config.
    Example: ["NORMAL", "PNEUMONIA"]
    """
    return config["dataset"]["classes"]

# ── Image Stats ──────────────────────────────────────────────

def print_dataset_stats(train_gen, val_gen, test_gen=None):
    """
    Prints quick stats about your dataset generators.
    """
    print("\n" + "=" * 50)
    print("       DATASET STATISTICS")
    print("=" * 50)
    print(f"  Training samples   : {train_gen.samples}")
    print(f"  Validation samples : {val_gen.samples}")
    if test_gen:
        print(f"  Test samples       : {test_gen.samples}")
    print(f"  Classes            : {list(train_gen.class_indices.keys())}")
    print(f"  Image size         : {train_gen.image_shape}")
    print(f"  Batch size         : {train_gen.batch_size}")
    print("=" * 50 + "\n")

# ── Training History Plotter ─────────────────────────────────

def plot_training_history(history, save_path: str = "outputs/plots/training_history.png"):
    """
    Plots accuracy and loss curves from training history.
    Saves the figure to save_path.
    """
    ensure_dirs(os.path.dirname(save_path))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Model Training History", fontsize=16, fontweight='bold')
    
    # ── Accuracy Plot ──
    axes[0].plot(history.history["accuracy"],
                 label="Train Accuracy", color="#0066CC", linewidth=2)
    axes[0].plot(history.history["val_accuracy"],
                 label="Val Accuracy", color="#FF6600", linewidth=2, linestyle="--")
    axes[0].set_title("Accuracy Over Epochs", fontsize=13)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1])
    
    # ── Loss Plot ──
    axes[1].plot(history.history["loss"],
                 label="Train Loss", color="#0066CC", linewidth=2)
    axes[1].plot(history.history["val_loss"],
                 label="Val Loss", color="#FF6600", linewidth=2, linestyle="--")
    axes[1].set_title("Loss Over Epochs", fontsize=13)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Training history saved → {save_path}")

# ── Sample Image Grid ─────────────────────────────────────────

def plot_sample_images(generator, class_labels: list,
                       save_path: str = "outputs/plots/sample_images.png",
                       num_images: int = 12):
    """
    Plots a grid of sample images from a data generator.
    """
    ensure_dirs(os.path.dirname(save_path))
    
    images, labels = next(generator)
    n = min(num_images, len(images))
    
    cols = 4
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    fig.suptitle("Sample Medical Images", fontsize=15, fontweight='bold')
    axes = axes.flatten()
    
    for i in range(n):
        axes[i].imshow(images[i])
        label_idx = int(np.argmax(labels[i])) if labels[i].ndim > 0 else int(labels[i])
        axes[i].set_title(class_labels[label_idx], fontsize=10,
                          color="green" if class_labels[label_idx] == "NORMAL" else "red")
        axes[i].axis("off")
    
    for j in range(n, len(axes)):
        axes[j].axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Sample images saved → {save_path}")


def plot_training_history(history, save_path: str = "outputs/plots/training_history.png"):
    """
    Plots accuracy and loss curves from training history.
    Saves the figure to save_path.
    """
    ensure_dirs(os.path.dirname(save_path))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Model Training History", fontsize=16, fontweight='bold')
    
    # ── Accuracy Plot ──
    axes[0].plot(history.history["accuracy"],
                 label="Train Accuracy", color="#0066CC", linewidth=2)
    axes[0].plot(history.history["val_accuracy"],
                 label="Val Accuracy", color="#FF6600", linewidth=2, linestyle="--")
    axes[0].set_title("Accuracy Over Epochs", fontsize=13)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1])
    
    # ── Loss Plot ──
    axes[1].plot(history.history["loss"],
                 label="Train Loss", color="#0066CC", linewidth=2)
    axes[1].plot(history.history["val_loss"],
                 label="Val Loss", color="#FF6600", linewidth=2, linestyle="--")
    axes[1].set_title("Loss Over Epochs", fontsize=13)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Training history saved → {save_path}")


def plot_training_history(history, save_path: str = "outputs/plots/training_history.png"):
    """
    Plots accuracy and loss curves from training history.
    Saves the figure to save_path.
    """
    ensure_dirs(os.path.dirname(save_path))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Model Training History", fontsize=16, fontweight='bold')
    
    # ── Accuracy Plot ──
    axes[0].plot(history.history["accuracy"],
                 label="Train Accuracy", color="#0066CC", linewidth=2)
    axes[0].plot(history.history["val_accuracy"],
                 label="Val Accuracy", color="#FF6600", linewidth=2, linestyle="--")
    axes[0].set_title("Accuracy Over Epochs", fontsize=13)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1])
    
    # ── Loss Plot ──
    axes[1].plot(history.history["loss"],
                 label="Train Loss", color="#0066CC", linewidth=2)
    axes[1].plot(history.history["val_loss"],
                 label="Val Loss", color="#FF6600", linewidth=2, linestyle="--")
    axes[1].set_title("Loss Over Epochs", fontsize=13)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Training history saved → {save_path}")


# ── Sample Image Grid ─────────────────────────────────────────

def plot_sample_images(generator, class_labels: list,
                       save_path: str = "outputs/plots/sample_images.png",
                       num_images: int = 12):
    """
    Plots a grid of sample images from a data generator.
    """
    ensure_dirs(os.path.dirname(save_path))
    
    images, labels = next(generator)
    n = min(num_images, len(images))
    
    cols = 4
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    fig.suptitle("Sample Medical Images", fontsize=15, fontweight='bold')
    axes = axes.flatten()
    
    for i in range(n):
        axes[i].imshow(images[i])
        label_idx = int(np.argmax(labels[i])) if labels[i].ndim > 0 else int(labels[i])
        axes[i].set_title(class_labels[label_idx], fontsize=10,
                          color="green" if class_labels[label_idx] == "NORMAL" else "red")
        axes[i].axis("off")
    
    for j in range(n, len(axes)):
        axes[j].axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Sample images saved → {save_path}")


def plot_sample_images(generator, class_labels: list,
                       save_path: str = "outputs/plots/sample_images.png",
                       num_images: int = 12):
    """
    Plots a grid of sample images from a data generator.
    """
    ensure_dirs(os.path.dirname(save_path))
    
    images, labels = next(generator)
    n = min(num_images, len(images))
    
    cols = 4
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    fig.suptitle("Sample Medical Images", fontsize=15, fontweight='bold')
    axes = axes.flatten()
    
    for i in range(n):
        axes[i].imshow(images[i])
        label_idx = int(np.argmax(labels[i])) if labels[i].ndim > 0 else int(labels[i])
        axes[i].set_title(class_labels[label_idx], fontsize=10,
                          color="green" if class_labels[label_idx] == "NORMAL" else "red")
        axes[i].axis("off")
    
    for j in range(n, len(axes)):
        axes[j].axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Sample images saved → {save_path}")

