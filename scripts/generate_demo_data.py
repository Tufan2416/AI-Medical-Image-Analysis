# ============================================================
# scripts/generate_demo_data.py
# Generates synthetic X-ray-like images for testing
# WITHOUT needing real dataset download
# ============================================================
#
# HOW TO USE:
#   python scripts/generate_demo_data.py
#
# This creates a small demo dataset in data/raw/ so you can
# test the full pipeline (train, evaluate, predict) without
# downloading the real 1.1GB Kaggle dataset.
#
# NOTE: The model will have low accuracy on synthetic data.
# This is only for testing that your code RUNS correctly.
# Use real Kaggle data for actual results.
# ============================================================

import os
import numpy as np
from PIL import Image
import random

random.seed(42)
np.random.seed(42)


def make_normal_xray(size=(224, 224)):
    """Generates a synthetic 'normal' chest X-ray."""
    img = np.zeros((*size, 3), dtype=np.uint8)

    # Background (dark)
    img[:, :] = [30, 30, 30]

    # Lung areas (lighter)
    h, w = size
    # Left lung
    for y in range(int(h*0.2), int(h*0.85)):
        for x in range(int(w*0.1), int(w*0.42)):
            dist = ((x - w*0.26)**2 / (w*0.15)**2 + (y - h*0.5)**2 / (h*0.3)**2)
            if dist < 1:
                val = int(180 - 60*dist + np.random.normal(0, 8))
                img[y, x] = [val, val, val]

    # Right lung
    for y in range(int(h*0.2), int(h*0.85)):
        for x in range(int(w*0.58), int(w*0.90)):
            dist = ((x - w*0.74)**2 / (w*0.15)**2 + (y - h*0.5)**2 / (h*0.3)**2)
            if dist < 1:
                val = int(175 - 55*dist + np.random.normal(0, 8))
                img[y, x] = [val, val, val]

    # Spine (bright vertical line)
    for y in range(int(h*0.1), int(h*0.9)):
        for x in range(int(w*0.48), int(w*0.52)):
            img[y, x] = [220, 220, 220]

    return img


def make_pneumonia_xray(size=(224, 224)):
    """Generates a synthetic 'pneumonia' chest X-ray (with opacities)."""
    img = make_normal_xray(size)
    h, w = size

    # Add infiltrate/opacity patches (simulate pneumonia consolidation)
    n_patches = random.randint(2, 5)
    for _ in range(n_patches):
        # Random location in lung area
        cx = random.randint(int(w*0.1), int(w*0.9))
        cy = random.randint(int(h*0.25), int(h*0.75))
        r  = random.randint(15, 40)
        intensity = random.randint(60, 140)

        for y in range(max(0, cy-r), min(h, cy+r)):
            for x in range(max(0, cx-r), min(w, cx+r)):
                dist = ((x-cx)**2 + (y-cy)**2)**0.5
                if dist < r:
                    # Make brighter (more radio-opaque)
                    factor = 1 - dist/r
                    current = img[y, x].astype(float)
                    new_val = np.clip(current + intensity * factor, 0, 255).astype(np.uint8)
                    img[y, x] = new_val

    return img


def generate_demo_dataset(data_dir: str = "data/raw",
                            n_train: int = 200,
                            n_val:   int = 40,
                            n_test:  int = 60):
    """
    Creates a synthetic demo dataset.

    Args:
        data_dir: Root data directory
        n_train:  Number of training images per class
        n_val:    Number of validation images per class
        n_test:   Number of test images per class
    """
    splits = {
        "train": n_train,
        "val":   n_val,
        "test":  n_test
    }
    classes = {
        "NORMAL":    make_normal_xray,
        "PNEUMONIA": make_pneumonia_xray
    }

    total = 0
    print("=" * 55)
    print("  Generating Synthetic Demo Dataset")
    print("=" * 55)

    for split, n in splits.items():
        for cls, gen_fn in classes.items():
            dir_path = os.path.join(data_dir, split, cls)
            os.makedirs(dir_path, exist_ok=True)

            for i in range(n):
                img_data = gen_fn()
                # Add slight random variations
                noise = np.random.normal(0, 5, img_data.shape).astype(np.int16)
                img_data = np.clip(img_data.astype(np.int16) + noise, 0, 255).astype(np.uint8)

                img_path = os.path.join(dir_path, f"{cls}_{i:04d}.jpg")
                Image.fromarray(img_data).save(img_path, quality=90)
                total += 1

            print(f"  ✓ {split}/{cls}: {n} images")

    print("=" * 55)
    print(f"  Total images created: {total}")
    print(f"  Location: {os.path.abspath(data_dir)}")
    print("=" * 55)
    print("\n⚠️  REMINDER: This is SYNTHETIC data for testing only.")
    print("   For real results, use the Kaggle Chest X-Ray dataset:")
    print("   python scripts/download_dataset.py")
    print("\n🚀 Ready to test! Run:")
    print("   python main.py --mode train")


if __name__ == "__main__":
    generate_demo_dataset()
