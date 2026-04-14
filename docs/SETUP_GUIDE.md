# Setup Guide — Step by Step
## AI-Powered Medical Image Analysis System

---

## Prerequisites

| Requirement | Version | Check Command |
|-------------|---------|---------------|
| Python | 3.10+ | `python --version` |
| pip | Latest | `pip --version` |
| Git | Any | `git --version` |
| VS Code | Any | _(open manually)_ |
| RAM | 8GB+ | — |
| GPU | Optional | Speeds training 10× |

---

## Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/AI-Medical-Image-Analysis.git
cd AI-Medical-Image-Analysis
```

---

## Step 2: Create Virtual Environment

### Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

### Mac/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

---

## Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs TensorFlow, Streamlit, FastAPI, OpenCV, and all other dependencies.

**If you have a GPU (NVIDIA):**
```bash
pip install tensorflow[and-cuda]
```

**Expected install time:** 5-10 minutes

---

## Step 4: Verify Installation

```bash
python -c "
import tensorflow as tf
import cv2
import streamlit
import fastapi
print(f'TensorFlow: {tf.__version__}')
print(f'OpenCV: {cv2.__version__}')
print(f'Streamlit: {streamlit.__version__}')
print(f'GPU: {len(tf.config.list_physical_devices(\"GPU\"))} device(s)')
print('All dependencies OK!')
"
```

---

## Step 5: Get the Dataset

### Option A: Quick Demo (Synthetic data — no download needed)
```bash
python scripts/generate_demo_data.py
```
Creates 600 synthetic X-ray images for testing.

### Option B: Real Dataset (Kaggle — Recommended for real results)

1. Install Kaggle CLI:
   ```bash
   pip install kaggle
   ```

2. Get your API key:
   - Go to https://www.kaggle.com → Settings → API → Create New Token
   - Download `kaggle.json`
   - Place it at: `~/.kaggle/kaggle.json` (Mac/Linux) or `C:\Users\YOU\.kaggle\kaggle.json` (Windows)

3. Download:
   ```bash
   python scripts/download_dataset.py
   ```

4. Verify:
   ```bash
   python scripts/download_dataset.py --verify-only
   ```

---

## Step 6: Train the Model

```bash
python main.py --mode train
```

**What happens:**
- Loads dataset generators
- Builds MobileNetV2
- Trains for up to 20 epochs
- Saves best model to `models/best_model.keras`
- Generates training plots in `outputs/plots/`

**Expected training time:**
- CPU: ~2-4 hours for 20 epochs
- GPU (RTX 3060+): ~15-30 minutes
- Google Colab (free GPU): ~20-40 minutes

**Tip for Google Colab:**
```python
# Mount Drive first
from google.colab import drive
drive.mount('/content/drive')

# Then clone and run
!git clone https://github.com/yourusername/AI-Medical-Image-Analysis.git
%cd AI-Medical-Image-Analysis
!pip install -r requirements.txt
!python main.py --mode train
```

---

## Step 7: Evaluate

```bash
python main.py --mode evaluate
```

Generates:
- Confusion matrix → `outputs/plots/confusion_matrix.png`
- ROC curve → `outputs/plots/roc_curve.png`
- P-R curve → `outputs/plots/precision_recall_curve.png`
- Grad-CAM grid → `outputs/plots/gradcam_grid.png`

---

## Step 8: Predict on a Single Image

```bash
python main.py --mode predict --image path/to/your_xray.jpg
```

Output:
```
==================================================
         PREDICTION RESULT
==================================================
  Diagnosis   : PNEUMONIA
  Confidence  : 92.50%
  Probabilities:
    NORMAL         :   7.5%  ██
    PNEUMONIA      :  92.5%  ████████████████████████████
==================================================
```

---

## Step 9: Run the Streamlit App

```bash
streamlit run app/streamlit_app.py
```

Opens in browser: http://localhost:8501

---

## Step 10: Run the FastAPI

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
```

- Swagger UI: http://localhost:8000/docs
- API docs:   http://localhost:8000/redoc
- Health:     http://localhost:8000/health

**Test with curl:**
```bash
curl -X POST http://localhost:8000/predict \
     -F "file=@data/sample/your_image.jpg"
```

**Response:**
```json
{
  "prediction": "PNEUMONIA",
  "confidence": 0.9250,
  "probabilities": {"NORMAL": 0.075, "PNEUMONIA": 0.925},
  "model": "MobileNetV2",
  "status": "success"
}
```

---

## Common Errors & Fixes

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: tensorflow` | Run `pip install -r requirements.txt` |
| `FileNotFoundError: best_model.keras` | Run `python main.py --mode train` first |
| `No images found` | Run `python scripts/generate_demo_data.py` |
| OOM (Out of Memory) | Reduce `batch_size` in `config.yaml` to 16 or 8 |
| Streamlit port busy | Run `streamlit run app/streamlit_app.py --server.port 8502` |
| API port busy | Change port: `uvicorn app.api:app --port 8001` |

---

## Project Folder Structure

```
AI-Medical-Image-Analysis/
│
├── 📁 data/
│   ├── raw/            ← Kaggle dataset (train/val/test)
│   ├── processed/      ← Preprocessed data cache
│   └── sample/         ← Small sample images for testing
│
├── 📁 notebooks/
│   ├── 01_EDA.ipynb
│   └── 02_model_training.ipynb
│
├── 📁 src/             ← All Python modules
│   ├── utils.py        ← Logger, config, plotting helpers
│   ├── preprocess.py   ← Image preprocessing & augmentation
│   ├── model.py        ← MobileNetV2 architecture
│   ├── train.py        ← Training pipeline
│   ├── evaluate.py     ← Metrics, plots
│   ├── gradcam.py      ← Grad-CAM explainability
│   └── predict.py      ← Inference pipeline
│
├── 📁 app/
│   ├── streamlit_app.py ← Web UI
│   └── api.py           ← FastAPI REST API
│
├── 📁 models/          ← Saved model weights
├── 📁 outputs/
│   ├── plots/          ← Training curves, confusion matrix
│   └── predictions/    ← Grad-CAM outputs
│
├── 📁 scripts/
│   ├── download_dataset.py
│   └── generate_demo_data.py
│
├── 📁 docs/            ← Documentation
├── 📄 main.py          ← CLI entry point
├── 📄 config.yaml      ← All settings
├── 📄 requirements.txt
└── 📄 README.md
```
