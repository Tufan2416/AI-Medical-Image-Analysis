# ============================================================
# src/gradcam.py — Grad-CAM Explainability
# AI-Powered Medical Image Analysis System
# ============================================================
#
# WHAT IS GRAD-CAM?
#   Gradient-weighted Class Activation Mapping (Grad-CAM)
#   highlights WHICH REGIONS of the image the model looked at
#   when making its prediction.
#
# WHY IS THIS CRITICAL IN HEALTHCARE?
#   Doctors don't trust a black-box AI.
#   With Grad-CAM, the AI says: "I predicted PNEUMONIA because
#   this region of the lung showed abnormal density."
#   This builds clinical trust and enables human-AI collaboration.
#
# HOW IT WORKS:
#   1. Forward pass → get prediction
#   2. Backprop gradients to last conv layer
#   3. Weight each feature map by gradient magnitude
#   4. Average → heatmap
#   5. Overlay on original image
# ============================================================

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
from src.utils import get_logger, ensure_dirs
import os

logger = get_logger(__name__)


# ── Core Grad-CAM ─────────────────────────────────────────────
def compute_gradcam(model, img_array, pred_index=None):
    import tensorflow as tf

    # Step 1: Get backbone (MobileNetV2)
    backbone = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            backbone = layer
            break

    if backbone is None:
        raise ValueError("Backbone model not found")

    # Step 2: Find last Conv2D layer inside backbone
    last_conv_layer = None
    for layer in reversed(backbone.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
            break

    if last_conv_layer is None:
        raise ValueError("No Conv2D layer found in backbone")

    # Step 3: Create grad model
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [last_conv_layer.output, model.output]
    )

    # Step 4: Compute gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)

    return heatmap.numpy()
    # ── Auto-detect last conv layer ───────────────────────
    last_conv_layer = None

    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model):
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, tf.keras.layers.Conv2D):
                    last_conv_layer = sub_layer.name
                    break
            if last_conv_layer:
                break

    last_conv_layer_name = last_conv_layer

    # ── Build Grad-CAM Model ──────────────────────────────
    # This sub-model outputs both the conv activations AND final predictions
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    # ── Forward + Gradient Pass ───────────────────────────
    with tf.GradientTape() as tape:
        # Cast to float32
        img_tensor = tf.cast(img_array, tf.float32)
        conv_outputs, predictions = grad_model(img_tensor)

        if pred_index is None:
            pred_index = tf.argmax(predictions[0])

        # Score for the target class
        class_channel = predictions[:, pred_index]

    # Gradients of class score w.r.t. conv feature maps
    grads = tape.gradient(class_channel, conv_outputs)

    # ── Pool Gradients ────────────────────────────────────
    # Global average pooling over the spatial dimensions
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight each feature map by its importance
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ── Normalize to [0, 1] ───────────────────────────────
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


# ── Overlay Heatmap on Image ──────────────────────────────────
def overlay_gradcam(original_img: np.ndarray,
                    heatmap: np.ndarray,
                    alpha: float = 0.4,
                    colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """
    Overlays a Grad-CAM heatmap on the original image.

    Args:
        original_img: Original image, shape (H, W, 3), values [0, 1] or [0, 255]
        heatmap:      Grad-CAM heatmap, shape (h, w), values [0, 1]
        alpha:        Transparency for overlay (0 = invisible, 1 = opaque)
        colormap:     OpenCV colormap (JET = red-hot, VIRIDIS = green)

    Returns:
        Overlaid image as uint8 np.ndarray (H, W, 3)
    """
    # Ensure original_img is uint8
    if original_img.dtype != np.uint8:
        original_img = (original_img * 255).astype(np.uint8)

    H, W = original_img.shape[:2]

    # Resize heatmap to match original image
    heatmap_resized = cv2.resize(heatmap, (W, H))

    # Apply colormap
    heatmap_colored = cv2.applyColorMap(
        (heatmap_resized * 255).astype(np.uint8),
        colormap
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Superimpose
    overlaid = cv2.addWeighted(original_img, 1 - alpha,
                                heatmap_colored, alpha, 0)
    return overlaid


# ── Full Grad-CAM Visualization ───────────────────────────────
def visualize_gradcam(model: tf.keras.Model,
                      img_array: np.ndarray,
                      prediction_result: dict,
                      class_labels: list,
                      save_path: str = "outputs/predictions/gradcam.png",
                      last_conv_layer: str = None) -> np.ndarray:
    """
    Creates a 3-panel Grad-CAM visualization:
      Left:   Original X-ray
      Center: Grad-CAM heatmap
      Right:  Overlay (heatmap on X-ray)

    Args:
        model:             Trained Keras model
        img_array:         Preprocessed image (1, H, W, 3)
        prediction_result: Output of evaluate.predict_single()
        class_labels:      List of class names
        save_path:         Where to save the figure

    Returns:
        overlaid_img: np.ndarray
    """
    ensure_dirs(os.path.dirname(save_path))

    pred_class = prediction_result["prediction"]
    confidence = prediction_result["confidence"]
    pred_idx   = class_labels.index(pred_class)
    
 # ── Compute Grad-CAM ──────────────────────────────────
def compute_gradcam(model, img_array, pred_index=None):
    import tensorflow as tf
    import numpy as np
    batch_gradcam_grid(...)

    # 1. Find last Conv2D layer automatically
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
            break

        # Handle nested models (MobileNetV2)
        if isinstance(layer, tf.keras.Model):
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, tf.keras.layers.Conv2D):
                    last_conv_layer = sub_layer
                    break
        if last_conv_layer is not None:
            break

    if last_conv_layer is None:
        raise ValueError("No Conv2D layer found in model.")

    # 2. Create Grad-CAM model properly
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[last_conv_layer.output, model.output]
    )

    # 3. Compute gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)

        if pred_index is None:
            pred_index = tf.argmax(predictions[0])

        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)

    # 4. Compute weights
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # 5. Normalize
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)

    return heatmap.numpy()
    # Panel 1 — Original
    axes[0].imshow(original)
    axes[0].set_title("Original X-Ray", fontsize=13, fontweight="bold")
    axes[0].axis("off")

    # Panel 2 — Grad-CAM heatmap
    heatmap_display = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
    im = axes[1].imshow(heatmap_display, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM Heatmap\n(Red = High Attention)", fontsize=13, fontweight="bold")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3 — Overlay
    axes[2].imshow(overlaid)
    axes[2].set_title("Overlay: AI Focus Region", fontsize=13, fontweight="bold")
    axes[2].axis("off")

    # Add probability bar
    probs = prediction_result["probabilities"]
    prob_text = "  |  ".join([f"{k}: {v*100:.1f}%" for k, v in probs.items()])
    fig.text(0.5, 0.02, prob_text, ha="center", fontsize=11,
             color="#333333", style="italic")

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Grad-CAM saved → {save_path}")

    return overlaid


# ── Helper: Find Last Conv Layer ─────────────────────────────
def _find_last_conv_layer(model: tf.keras.Model) -> str:
    """
    Auto-detects the name of the last convolutional layer.
    Works with MobileNetV2, ResNet50, and most CNN architectures.
    """
    for layer in reversed(model.layers):
        # Check if it's a Conv2D or has 4D output (spatial feature maps)
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        # For nested models (e.g., MobileNetV2 inside Model)
        if hasattr(layer, 'layers'):
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, tf.keras.layers.Conv2D):
                    return sub_layer.name

    raise ValueError("Could not find a Conv2D layer in the model.")


# ── Batch Grad-CAM ────────────────────────────────────────────
def batch_gradcam_grid(model: tf.keras.Model,
                       test_gen,
                       class_labels: list,
                       num_images: int = 8,
                       save_path: str = "outputs/plots/gradcam_grid.png"):
    """
    Generates Grad-CAM overlays for multiple test images
    and arranges them in a grid.

    Great for your GitHub README screenshot!
    """
    ensure_dirs(os.path.dirname(save_path))
    last_conv = _find_last_conv_layer(model)

    test_gen.reset()
    images_batch, labels_batch = next(test_gen)
    n = min(num_images, len(images_batch))

    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    fig.suptitle("Grad-CAM Batch Visualization", fontsize=16, fontweight="bold")
    axes = axes.flatten()

    for i in range(n):
        img = images_batch[i:i+1]
        true_idx = int(np.argmax(labels_batch[i]))
        true_label = class_labels[true_idx]

        proba = model.predict(img, verbose=0)
        pred_idx = int(np.argmax(proba[0]))
        pred_label = class_labels[pred_idx]
        conf = float(proba[0][pred_idx])

        heatmap = compute_gradcam(model, img, pred_index=pred_idx)
        overlaid = overlay_gradcam(img[0], heatmap)

        axes[i].imshow(overlaid)
        color = "green" if pred_label == true_label else "red"
        axes[i].set_title(
            f"True: {true_label}\nPred: {pred_label} ({conf*100:.0f}%)",
            fontsize=9, color=color
        )
        axes[i].axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Grad-CAM grid saved → {save_path}")
