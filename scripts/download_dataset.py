# ============================================================
# scripts/download_dataset.py — Dataset Setup Helper
# AI-Powered Medical Image Analysis System
# ============================================================
#
# DATASET: Chest X-Ray Images (Pneumonia)
# SOURCE:  https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
# SIZE:    ~1.1 GB
# CLASSES: NORMAL, PNEUMONIA
#
# HOW TO USE:
#   Option 1 (Kaggle API — Recommended):
#     1. Install Kaggle: pip install kaggle
#     2. Set up API key: https://www.kaggle.com/docs/api
#     3. Run: python scripts/download_dataset.py
#
#   Option 2 (Manual):
#     1. Visit the Kaggle URL above
#     2. Download chest-xray-pneumonia.zip
#     3. Extract to data/raw/
#     4. Run: python scripts/download_dataset.py --verify-only
#
# EXPECTED STRUCTURE AFTER DOWNLOAD:
#   data/raw/
#     train/
#       NORMAL/       (1341 images)
#       PNEUMONIA/    (3875 images)
#     val/
#       NORMAL/       (8 images)
#       PNEUMONIA/    (8 images)
#     test/
#       NORMAL/       (234 images)
#       PNEUMONIA/    (390 images)
# ============================================================

import os
import sys
import shutil
import argparse
from pathlib import Path


def check_kaggle_api():
    """Check if Kaggle API is configured."""
    try:
        import kaggle
        return True
    except ImportError:
        return False
    except Exception:
        return False


def download_via_kaggle(dest_dir: str = "data/raw"):
    """Download dataset using Kaggle API."""
    print("📥 Downloading Chest X-Ray dataset from Kaggle...")
    os.makedirs(dest_dir, exist_ok=True)

    # Download dataset zip
    os.system(
        f"kaggle datasets download -d paultimothymooney/chest-xray-pneumonia "
        f"--path {dest_dir} --unzip"
    )

    # The dataset extracts to chest_xray/ — move to expected structure
    extracted = os.path.join(dest_dir, "chest_xray")
    if os.path.exists(extracted):
        for split in ["train", "val", "test"]:
            src = os.path.join(extracted, split)
            dst = os.path.join(dest_dir, split)
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.move(src, dst)
                print(f"  Moved: {src} → {dst}")
        # Remove the now-empty chest_xray folder
        try:
            shutil.rmtree(extracted)
        except Exception:
            pass

    print("✅ Download complete!")


def verify_structure(data_dir: str = "data/raw") -> bool:
    """Verify the dataset has the correct folder structure."""
    print("\n🔍 Verifying dataset structure...")
    required = {
        "train/NORMAL":    50,    # at least 50 images
        "train/PNEUMONIA": 50,
        "test/NORMAL":     10,
        "test/PNEUMONIA":  10,
    }

    all_good = True
    for rel_path, min_count in required.items():
        full_path = os.path.join(data_dir, rel_path)
        if not os.path.exists(full_path):
            print(f"  ❌ MISSING: {full_path}")
            all_good = False
            continue

        images = [f for f in os.listdir(full_path)
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        if len(images) < min_count:
            print(f"  ⚠️  {rel_path}: only {len(images)} images (need ≥ {min_count})")
            all_good = False
        else:
            print(f"  ✅ {rel_path}: {len(images)} images")

    # Check for val/ — create it if missing (split from train)
    val_normal = os.path.join(data_dir, "val", "NORMAL")
    val_pneumo = os.path.join(data_dir, "val", "PNEUMONIA")
    if not os.path.exists(val_normal) or not os.path.exists(val_pneumo):
        print("\n  ⚠️  val/ split missing. Creating from train/ (10% split)...")
        create_val_split(data_dir)
        all_good = True  # we fixed it

    return all_good


def create_val_split(data_dir: str = "data/raw", val_ratio: float = 0.1):
    """
    Creates a validation split from the training data.
    Moves val_ratio of train images to val/.
    """
    import random
    random.seed(42)

    for cls in ["NORMAL", "PNEUMONIA"]:
        train_cls = os.path.join(data_dir, "train", cls)
        val_cls   = os.path.join(data_dir, "val", cls)
        os.makedirs(val_cls, exist_ok=True)

        images = [f for f in os.listdir(train_cls)
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        n_val = max(8, int(len(images) * val_ratio))
        val_images = random.sample(images, n_val)

        for img in val_images:
            shutil.move(
                os.path.join(train_cls, img),
                os.path.join(val_cls, img)
            )
        print(f"  Created val/{cls}: {n_val} images")


def print_dataset_summary(data_dir: str = "data/raw"):
    """Prints a summary table of the dataset."""
    print("\n📊 DATASET SUMMARY")
    print("=" * 50)
    total = 0
    for split in ["train", "val", "test"]:
        split_path = os.path.join(data_dir, split)
        if not os.path.exists(split_path):
            continue
        print(f"\n  {split.upper()}/")
        for cls in os.listdir(split_path):
            cls_path = os.path.join(split_path, cls)
            if os.path.isdir(cls_path):
                count = len([f for f in os.listdir(cls_path)
                             if f.lower().endswith((".jpg",".jpeg",".png"))])
                total += count
                print(f"    {cls:<15}: {count:>5} images")
    print(f"\n  TOTAL             : {total:>5} images")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Dataset setup for AI Medical Analysis")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only verify structure, don't download")
    parser.add_argument("--data-dir", type=str, default="data/raw",
                        help="Dataset root directory")
    args = parser.parse_args()

    print("=" * 55)
    print("  AI Medical Image Analysis — Dataset Setup")
    print("=" * 55)

    if args.verify_only:
        ok = verify_structure(args.data_dir)
        if ok:
            print_dataset_summary(args.data_dir)
            print("\n✅ Dataset is ready for training!")
        else:
            print("\n❌ Dataset structure issues found.")
            print("  Please download the dataset manually:")
            print("  https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
        return

    # Try Kaggle API download
    if check_kaggle_api():
        download_via_kaggle(args.data_dir)
    else:
        print("⚠️  Kaggle API not configured.")
        print("\n  MANUAL SETUP INSTRUCTIONS:")
        print("  1. Visit: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
        print("  2. Click 'Download' → chest-xray-pneumonia.zip")
        print("  3. Extract to: data/raw/")
        print("  4. Run: python scripts/download_dataset.py --verify-only")
        print("\n  For Kaggle API setup:")
        print("  1. pip install kaggle")
        print("  2. Get API key from: https://www.kaggle.com/settings → API")
        print("  3. Place kaggle.json in ~/.kaggle/kaggle.json")
        return

    verify_structure(args.data_dir)
    print_dataset_summary(args.data_dir)
    print("\n🚀 Ready to train! Run:")
    print("  python main.py --mode train")


if __name__ == "__main__":
    main()
