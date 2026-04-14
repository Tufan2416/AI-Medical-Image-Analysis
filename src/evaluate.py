# ============================================================
# src/evaluate.py — Professional Model Evaluation
# AI-Powered Medical Image Analysis System
# ============================================================
#
# METRICS EXPLAINED (for your README / interviews):
#
#   Accuracy  = (TP + TN) / Total          → overall correctness
#   Precision = TP / (TP + FP)             → of all predicted positive, how many are truly positive?
#   Recall    = TP / (TP + FN)             → of all actual positive, how many did we find?
#   F1 Score  = 2 × (P × R) / (P + R)     → harmonic mean of precision & recall
#   AUC-ROC   = area under ROC curve       → discrimination ability (0.5 = random, 1.0 = perfect)
#
#   In healthcare, RECALL is critical!
#   Missing a real pneumonia case (False Negative) is worse than
#   a false alarm (False Positive). We optimize for recall.
# ============================================================

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve,
    accuracy_score, f1_score, precision_score, recall_score
)
from src.utils import get_logger, load_config, ensure_dirs

logger = get_logger(__name__)


# ── Full Evaluation ───────────────────────────────────────────
def evaluate_model(model, test_gen,
                   class_labels: list = None,
                   save_dir: str = "outputs/plots"):
    """
    Runs complete evaluation on test data.

    Args:
        model:        Trained Keras model
        test_gen:     Test data generator
        class_labels: List of class names
        save_dir:     Directory to save plots

    Returns:
        dict with all metrics
    """
    ensure_dirs(save_dir)
    cfg = load_config()
    class_labels = class_labels or cfg["dataset"]["classes"]

    logger.info("Running model evaluation on test set...")
    test_gen.reset()

    # ── Get Predictions ───────────────────────────────────
    y_pred_proba = model.predict(test_gen, verbose=1)

    # Handle binary vs multi-class
    if y_pred_proba.shape[1] == 1:
        # Binary sigmoid output
        y_pred_proba_full = np.hstack([1 - y_pred_proba, y_pred_proba])
        y_pred = (y_pred_proba.flatten() > 0.5).astype(int)
    else:
        # Multi-class softmax output
        y_pred_proba_full = y_pred_proba
        y_pred = np.argmax(y_pred_proba, axis=1)

    y_true = test_gen.classes

    # ── Compute Metrics ───────────────────────────────────
    acc       = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall    = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1        = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # Print metrics table
    print("\n" + "=" * 55)
    print("        EVALUATION METRICS (Test Set)")
    print("=" * 55)
    print(f"  {'Accuracy':<20}: {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  {'Precision':<20}: {precision:.4f}")
    print(f"  {'Recall':<20}: {recall:.4f}")
    print(f"  {'F1 Score':<20}: {f1:.4f}")
    print("=" * 55)

    print("\nPer-Class Report:")
    print(classification_report(y_true, y_pred, target_names=class_labels))

    # ── Save Plots ────────────────────────────────────────
    _plot_confusion_matrix(y_true, y_pred, class_labels, save_dir)
    _plot_roc_curve(y_true, y_pred_proba_full, class_labels, save_dir)
    _plot_precision_recall_curve(y_true, y_pred_proba_full, class_labels, save_dir)

    metrics = {
        "accuracy":  acc,
        "precision": precision,
        "recall":    recall,
        "f1_score":  f1,
    }
    return metrics


# ── Confusion Matrix ──────────────────────────────────────────
def _plot_confusion_matrix(y_true, y_pred, class_labels, save_dir):
    """
    Plots and saves a styled confusion matrix.

    Rows = Actual labels
    Cols = Predicted labels
    Diagonal = Correct predictions
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)  # normalize

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Confusion Matrix", fontsize=16, fontweight="bold")

    for idx, (matrix, title) in enumerate([
        (cm,      "Counts"),
        (cm_norm, "Normalized (Recall per Class)")
    ]):
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".2f" if idx == 1 else "d",
            cmap="Blues",
            xticklabels=class_labels,
            yticklabels=class_labels,
            linewidths=0.5,
            ax=axes[idx],
            cbar_kws={"shrink": 0.8}
        )
        axes[idx].set_title(title, fontsize=13)
        axes[idx].set_ylabel("Actual Label", fontsize=11)
        axes[idx].set_xlabel("Predicted Label", fontsize=11)

    plt.tight_layout()
    path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Confusion matrix saved → {path}")


# ── ROC Curve ─────────────────────────────────────────────────
def _plot_roc_curve(y_true, y_pred_proba, class_labels, save_dir):
    """
    Plots ROC curves (one per class, OvR strategy).

    AUC = 1.0 → perfect model
    AUC = 0.5 → no better than random
    """
    n_classes = len(class_labels)
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ["#0066CC", "#FF6600", "#00AA44", "#AA00FF"]

    for i, (label, color) in enumerate(zip(class_labels, colors)):
        y_bin = (y_true == i).astype(int)
        proba = y_pred_proba[:, i]
        fpr, tpr, _ = roc_curve(y_bin, proba)
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{label} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
    ax.set_title("ROC Curves — One vs Rest", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    path = os.path.join(save_dir, "roc_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"ROC curve saved → {path}")


# ── Precision–Recall Curve ────────────────────────────────────
def _plot_precision_recall_curve(y_true, y_pred_proba, class_labels, save_dir):
    """
    Plots Precision–Recall curves.
    Especially important for imbalanced medical datasets.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#0066CC", "#FF6600", "#00AA44", "#AA00FF"]

    for i, (label, color) in enumerate(zip(class_labels, colors)):
        y_bin = (y_true == i).astype(int)
        proba = y_pred_proba[:, i]
        precision_vals, recall_vals, _ = precision_recall_curve(y_bin, proba)
        pr_auc = auc(recall_vals, precision_vals)
        ax.plot(recall_vals, precision_vals, color=color, lw=2,
                label=f"{label} (AUC = {pr_auc:.3f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision–Recall Curves", fontsize=14, fontweight="bold")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)

    path = os.path.join(save_dir, "precision_recall_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"P-R curve saved → {path}")


# ── Quick Test on Single Sample ───────────────────────────────
def predict_single(model, img_array: np.ndarray,
                   class_labels: list) -> dict:
    """
    Runs inference on a single preprocessed image.

    Args:
        model:        Loaded Keras model
        img_array:    Preprocessed image, shape (1, H, W, 3)
        class_labels: e.g. ["NORMAL", "PNEUMONIA"]

    Returns:
        {
          "prediction": "PNEUMONIA",
          "confidence": 0.93,
          "probabilities": {"NORMAL": 0.07, "PNEUMONIA": 0.93}
        }
    """
    proba = model.predict(img_array, verbose=0)

    if proba.shape[1] == 1:
        # Binary sigmoid
        conf = float(proba[0][0])
        pred_idx = 1 if conf > 0.5 else 0
        probs = {class_labels[0]: round(1 - conf, 4),
                 class_labels[1]: round(conf, 4)}
    else:
        # Multi-class softmax
        pred_idx = int(np.argmax(proba[0]))
        conf = float(proba[0][pred_idx])
        probs = {label: round(float(p), 4)
                 for label, p in zip(class_labels, proba[0])}

    return {
        "prediction":    class_labels[pred_idx],
        "confidence":    round(conf, 4),
        "probabilities": probs,
        "raw_proba":     proba[0].tolist()
    }
