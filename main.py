# ============================================================
# main.py — Central Entry Point
# AI-Powered Medical Image Analysis System
# ============================================================
#
# USAGE:
#   python main.py --mode train
#   python main.py --mode evaluate
#   python main.py --mode predict --image path/to/xray.jpg
#   python main.py --mode predict_batch --image_dir data/sample/
#   python main.py --mode demo
#
# ============================================================

import argparse
import os
import sys

from src.utils import get_logger, load_config, ensure_dirs

logger = get_logger("main")


def parse_args():
    parser = argparse.ArgumentParser(
        description="AI-Powered Medical Image Analysis System",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  python main.py --mode train
  python main.py --mode evaluate
  python main.py --mode predict --image data/sample/test_xray.jpg
  python main.py --mode predict_batch --image_dir data/sample/
  python main.py --mode demo
        """
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "evaluate", "predict", "predict_batch", "demo"],
        help="Mode to run"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to a single image (used in 'predict' mode)"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="data/sample",
        help="Directory of images (used in 'predict_batch' mode)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/raw",
        help="Root dataset directory (used in 'train' / 'evaluate' mode)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/best_model.keras",
        help="Path to saved model"
    )
    return parser.parse_args()


def mode_train(args):
    """Trains the MobileNetV2 model on the dataset."""
    logger.info("Mode: TRAIN")
    from src.train import train
    model, history, test_gen = train(
        data_dir=args.data_dir,
        model_save_path=args.model_path
    )
    logger.info("Training complete. Now running evaluation on test set...")
    mode_evaluate_with(model, test_gen)


def mode_evaluate(args):
    """Evaluates a saved model on the test set."""
    logger.info("Mode: EVALUATE")
    from src.model import load_trained_model
    from src.preprocess import create_data_generators
    from src.evaluate import evaluate_model
    from src.gradcam import batch_gradcam_grid
    from src.utils import load_config

    cfg = load_config()
    model = load_trained_model(args.model_path)
    _, _, test_gen = create_data_generators(args.data_dir)
    metrics = evaluate_model(model, test_gen, cfg["dataset"]["classes"])

    # Grad-CAM batch grid
    batch_gradcam_grid(model, test_gen, cfg["dataset"]["classes"])
    logger.info(f"Final Metrics: {metrics}")


def mode_evaluate_with(model, test_gen):
    """Evaluates with an already-loaded model."""
    from src.evaluate import evaluate_model
    from src.gradcam import batch_gradcam_grid
    from src.utils import load_config

    cfg = load_config()
    metrics = evaluate_model(model, test_gen, cfg["dataset"]["classes"])
    batch_gradcam_grid(model, test_gen, cfg["dataset"]["classes"])
    return metrics


def mode_predict(args):
    """Runs prediction on a single image."""
    logger.info("Mode: PREDICT")
    if args.image is None:
        logger.error("Please provide --image path for predict mode.")
        sys.exit(1)
    from src.predict import run_prediction
    result = run_prediction(args.image, model_path=args.model_path)
    return result


def mode_predict_batch(args):
    """Runs prediction on all images in a directory."""
    logger.info("Mode: PREDICT BATCH")
    from src.predict import predict_batch
    results = predict_batch(args.image_dir, model_path=args.model_path)
    return results


def mode_demo(args):
    """
    Demo mode — generates a synthetic test image and runs full pipeline.
    Useful for testing the project setup without real X-ray data.
    """
    logger.info("Mode: DEMO")
    import numpy as np
    from PIL import Image
    from src.predict import run_prediction

    ensure_dirs("data/sample")
    demo_path = "data/sample/demo_xray.jpg"

    # Create a synthetic grayscale image that mimics an X-ray
    np.random.seed(42)
    img_data = np.random.randint(30, 200, (224, 224, 3), dtype=np.uint8)
    # Add a bright circular region (simulates a lesion)
    center = (112, 112)
    for y in range(224):
        for x in range(224):
            dist = ((x - center[0])**2 + (y - center[1])**2)**0.5
            if dist < 40:
                img_data[y, x] = np.clip(img_data[y, x] + 80, 0, 255)

    Image.fromarray(img_data).save(demo_path)
    logger.info(f"Demo image saved: {demo_path}")

    try:
        result = run_prediction(demo_path, model_path=args.model_path)
        logger.info(f"Demo complete! Result: {result['prediction']}")
    except FileNotFoundError:
        logger.warning(
            "Model not found. Train first:\n"
            "  python main.py --mode train\n\n"
            "Then run demo:\n"
            "  python main.py --mode demo"
        )


# ── Main Dispatcher ───────────────────────────────────────────
def main():
    print_banner()
    args = parse_args()

    mode_map = {
        "train":         mode_train,
        "evaluate":      mode_evaluate,
        "predict":       mode_predict,
        "predict_batch": mode_predict_batch,
        "demo":          mode_demo,
    }

    mode_map[args.mode](args)


def print_banner():
    print("""
╔══════════════════════════════════════════════════════╗
║     AI-Powered Medical Image Analysis System         ║
║     Version 2.0 | MobileNetV2 + Grad-CAM            ║
║     Transfer Learning | Explainable AI               ║
╚══════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    main()
