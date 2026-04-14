# ============================================================
# src/predict.py — Prediction Pipeline
# AI-Powered Medical Image Analysis System
# ============================================================

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from src.model import load_trained_model
from src.preprocess import preprocess_single_image, preprocess_pil_image
from src.evaluate import predict_single
from src.gradcam import visualize_gradcam, _find_last_conv_layer
from src.utils import get_logger, load_config, ensure_dirs

logger = get_logger(__name__)


def run_prediction(image_input,
                   model_path: str = "models/best_model.keras",
                   save_dir: str = "outputs/predictions") -> dict:
    """
    Main prediction function.
    Accepts a file path or PIL Image object.

    Args:
        image_input: str (file path) OR PIL.Image object
        model_path:  Path to the saved .keras model
        save_dir:    Directory to save Grad-CAM output

    Returns:
        dict: {prediction, confidence, probabilities, gradcam_path}
    """
    ensure_dirs(save_dir)
    cfg         = load_config()
    class_labels = cfg["dataset"]["classes"]
    img_size    = tuple(cfg["dataset"]["image_size"])

    # ── Load Model ────────────────────────────────────────
    model = load_trained_model(model_path)

    # ── Preprocess Image ──────────────────────────────────
    if isinstance(image_input, str):
        img_array = preprocess_single_image(image_input, img_size)
        img_name  = os.path.splitext(os.path.basename(image_input))[0]
    else:
        img_array = preprocess_pil_image(image_input, img_size)
        img_name  = "uploaded_image"

    # ── Predict ───────────────────────────────────────────
    result = predict_single(model, img_array, class_labels)
    logger.info(f"Prediction: {result['prediction']} ({result['confidence']*100:.1f}%)")

    # ── Grad-CAM ──────────────────────────────────────────
    gradcam_path = os.path.join(save_dir, f"{img_name}_gradcam.png")
    try:
        last_conv = _find_last_conv_layer(model)
        visualize_gradcam(
            model, img_array, result,
            class_labels=class_labels,
            save_path=gradcam_path,
            last_conv_layer=last_conv
        )
        result["gradcam_path"] = gradcam_path
    except Exception as e:
        logger.warning(f"Grad-CAM failed: {e}")
        result["gradcam_path"] = None

    # ── Print Summary ─────────────────────────────────────
    print("\n" + "=" * 50)
    print("         PREDICTION RESULT")
    print("=" * 50)
    print(f"  Diagnosis   : {result['prediction']}")
    print(f"  Confidence  : {result['confidence'] * 100:.2f}%")
    print(f"  Probabilities:")
    for cls, prob in result["probabilities"].items():
        bar = "█" * int(prob * 30)
        print(f"    {cls:<15}: {prob*100:5.1f}%  {bar}")
    print("=" * 50)

    return result


def predict_batch(image_dir: str,
                  model_path: str = "models/best_model.keras",
                  save_dir: str = "outputs/predictions") -> list:
    """
    Runs predictions on all images in a directory.

    Args:
        image_dir:  Directory with test images
        model_path: Path to trained model
        save_dir:   Where to save outputs

    Returns:
        list of dicts with prediction results
    """
    ensure_dirs(save_dir)
    cfg          = load_config()
    class_labels = cfg["dataset"]["classes"]
    img_size     = tuple(cfg["dataset"]["image_size"])

    model = load_trained_model(model_path)

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    image_files = [f for f in os.listdir(image_dir)
                   if os.path.splitext(f)[1].lower() in extensions]

    if not image_files:
        logger.warning(f"No images found in {image_dir}")
        return []

    results = []
    print(f"\n Processing {len(image_files)} images...\n")

    for i, fname in enumerate(image_files, 1):
        fpath = os.path.join(image_dir, fname)
        img_array = preprocess_single_image(fpath, img_size)
        result = predict_single(model, img_array, class_labels)
        result["filename"] = fname
        results.append(result)

        symbol = "✓" if result["prediction"] == "NORMAL" else "⚠"
        print(f"  [{i:>3}/{len(image_files)}] {symbol} {fname:<30} → "
              f"{result['prediction']} ({result['confidence']*100:.1f}%)")

    _save_batch_summary(results, class_labels,
                         os.path.join(save_dir, "batch_summary.png"))
    return results


def _save_batch_summary(results: list, class_labels: list, save_path: str):
    """Saves a bar chart summary of batch predictions."""
    from collections import Counter
    counts = Counter(r["prediction"] for r in results)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Batch Prediction Summary", fontsize=14, fontweight="bold")

    # Bar chart
    labels = list(counts.keys())
    values = [counts[l] for l in labels]
    colors = ["#0066CC" if l == "NORMAL" else "#CC0000" for l in labels]
    axes[0].bar(labels, values, color=colors, edgecolor="white", linewidth=1.5)
    axes[0].set_title("Prediction Distribution")
    axes[0].set_ylabel("Count")
    for j, v in enumerate(values):
        axes[0].text(j, v + 0.1, str(v), ha="center", fontweight="bold")

    # Confidence distribution
    confs = [r["confidence"] for r in results]
    pred_colors = ["#0066CC" if r["prediction"] == "NORMAL" else "#CC0000"
                   for r in results]
    axes[1].scatter(range(len(confs)), confs, c=pred_colors, alpha=0.7, s=60)
    axes[1].axhline(0.5, linestyle="--", color="gray", alpha=0.7)
    axes[1].set_title("Confidence Scores per Image")
    axes[1].set_xlabel("Image Index")
    axes[1].set_ylabel("Confidence")
    axes[1].set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Batch summary saved → {save_path}")
