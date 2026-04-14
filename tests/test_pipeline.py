# ============================================================
# tests/test_pipeline.py — Unit Tests
# AI-Powered Medical Image Analysis System
# ============================================================
#
# HOW TO RUN:
#   pytest tests/ -v
#   pytest tests/test_pipeline.py -v --tb=short
# ============================================================

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ── Test Config ───────────────────────────────────────────────
class TestConfig:
    def test_config_loads(self):
        """Config YAML should load without errors."""
        from src.utils import load_config
        cfg = load_config()
        assert isinstance(cfg, dict)
        assert "dataset" in cfg
        assert "model" in cfg
        assert "training" in cfg

    def test_config_values(self):
        """Config should have expected values."""
        from src.utils import load_config
        cfg = load_config()
        assert cfg["dataset"]["num_classes"] >= 2
        assert len(cfg["dataset"]["image_size"]) == 2
        assert cfg["dataset"]["image_size"][0] == 224


# ── Test Preprocessing ────────────────────────────────────────
class TestPreprocessing:
    def test_preprocess_pil_image(self, tmp_path):
        """PIL image should be preprocessed to correct shape."""
        from PIL import Image
        from src.preprocess import preprocess_pil_image

        # Create a dummy image
        img = Image.fromarray(
            np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        )
        arr = preprocess_pil_image(img, img_size=(224, 224))

        assert arr.shape == (1, 224, 224, 3), f"Expected (1,224,224,3), got {arr.shape}"
        assert arr.max() <= 1.0, "Values should be normalized to [0,1]"
        assert arr.min() >= 0.0, "Values should be >= 0"

    def test_preprocess_single_image(self, tmp_path):
        """File-based preprocessing should work correctly."""
        from PIL import Image
        from src.preprocess import preprocess_single_image

        # Save a dummy image
        img_path = str(tmp_path / "test_xray.jpg")
        img = Image.fromarray(
            np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        )
        img.save(img_path)

        arr = preprocess_single_image(img_path, img_size=(224, 224))

        assert arr.shape == (1, 224, 224, 3)
        assert arr.max() <= 1.0
        assert arr.dtype == np.float32 or arr.dtype == np.float64


# ── Test Model Build ──────────────────────────────────────────
class TestModelBuild:
    def test_model_builds(self):
        """Model should build without errors."""
        from src.model import build_model
        model = build_model(num_classes=2, img_size=(224, 224))
        assert model is not None
        assert model.name.startswith("MedicalAI")

    def test_model_input_shape(self):
        """Model should accept correct input shape."""
        from src.model import build_model
        model = build_model(num_classes=2, img_size=(224, 224))
        assert model.input_shape == (None, 224, 224, 3)

    def test_model_output_shape_binary(self):
        """Binary model should have 1 output unit."""
        from src.model import build_model
        model = build_model(num_classes=2, img_size=(224, 224))
        # For binary: output is (None, 1)
        assert model.output_shape[-1] in [1, 2]

    def test_model_forward_pass(self):
        """Model should process a dummy image without errors."""
        from src.model import build_model
        model = build_model(num_classes=2, img_size=(224, 224))

        dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
        output = model.predict(dummy_input, verbose=0)

        assert output is not None
        assert output.shape[0] == 1

    def test_model_multiclass(self):
        """Multi-class model should build for 3+ classes."""
        from src.model import build_model
        model = build_model(num_classes=3, img_size=(224, 224))
        assert model.output_shape[-1] == 3


# ── Test Prediction ───────────────────────────────────────────
class TestPrediction:
    def test_predict_single_returns_dict(self):
        """predict_single should return a dict with required keys."""
        from src.model import build_model
        from src.evaluate import predict_single

        model = build_model(num_classes=2)
        dummy_img = np.random.rand(1, 224, 224, 3).astype(np.float32)
        class_labels = ["NORMAL", "PNEUMONIA"]

        result = predict_single(model, dummy_img, class_labels)

        assert isinstance(result, dict)
        assert "prediction" in result
        assert "confidence" in result
        assert "probabilities" in result
        assert result["prediction"] in class_labels
        assert 0.0 <= result["confidence"] <= 1.0

    def test_predict_confidence_valid(self):
        """Confidence score should always be in [0, 1]."""
        from src.model import build_model
        from src.evaluate import predict_single

        model = build_model(num_classes=2)
        class_labels = ["NORMAL", "PNEUMONIA"]

        for _ in range(5):
            dummy_img = np.random.rand(1, 224, 224, 3).astype(np.float32)
            result = predict_single(model, dummy_img, class_labels)
            assert 0.0 <= result["confidence"] <= 1.0

    def test_probabilities_sum_to_one(self):
        """Probabilities should sum to ~1.0."""
        from src.model import build_model
        from src.evaluate import predict_single

        model = build_model(num_classes=2)
        dummy_img = np.random.rand(1, 224, 224, 3).astype(np.float32)
        class_labels = ["NORMAL", "PNEUMONIA"]

        result = predict_single(model, dummy_img, class_labels)
        total = sum(result["probabilities"].values())
        assert abs(total - 1.0) < 0.05, f"Probabilities sum: {total}"


# ── Test Grad-CAM ─────────────────────────────────────────────
class TestGradCAM:
    def test_gradcam_output_shape(self):
        """Grad-CAM heatmap should have 2D shape."""
        from src.model import build_model
        from src.gradcam import compute_gradcam, _find_last_conv_layer

        model = build_model(num_classes=2)
        dummy_img = np.random.rand(1, 224, 224, 3).astype(np.float32)

        last_conv = _find_last_conv_layer(model)
        assert last_conv is not None

        heatmap = compute_gradcam(model, dummy_img, pred_index=0,
                                   last_conv_layer_name=last_conv)
        assert heatmap is not None
        assert len(heatmap.shape) == 2  # 2D spatial heatmap

    def test_gradcam_values_range(self):
        """Grad-CAM values should be in [0, 1]."""
        from src.model import build_model
        from src.gradcam import compute_gradcam, _find_last_conv_layer

        model = build_model(num_classes=2)
        dummy_img = np.random.rand(1, 224, 224, 3).astype(np.float32)

        last_conv = _find_last_conv_layer(model)
        heatmap = compute_gradcam(model, dummy_img, pred_index=0,
                                   last_conv_layer_name=last_conv)

        assert heatmap.min() >= 0.0
        assert heatmap.max() <= 1.0 + 1e-6  # small epsilon for float precision


# ── Test Utils ────────────────────────────────────────────────
class TestUtils:
    def test_ensure_dirs(self, tmp_path):
        """ensure_dirs should create directories."""
        from src.utils import ensure_dirs
        test_dir = str(tmp_path / "test" / "nested" / "dir")
        ensure_dirs(test_dir)
        assert os.path.exists(test_dir)

    def test_get_logger(self):
        """Logger should be created without errors."""
        from src.utils import get_logger
        logger = get_logger("test_logger")
        assert logger is not None
        logger.info("Test log message — OK")


# ── Run all tests ─────────────────────────────────────────────
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
