# System Architecture

## AI-Powered Medical Image Analysis System

---

## Block Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    INPUT LAYER                          │
│  Patient X-Ray → Upload (UI/API) → Image File           │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│               PREPROCESSING MODULE                      │
│  • Resize to 224×224 pixels                             │
│  • Convert to RGB (3 channels)                          │
│  • Normalize pixel values: [0,255] → [0.0, 1.0]        │
│  • Augmentation (train only): flip, rotate, zoom        │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│            FEATURE EXTRACTION (MobileNetV2)             │
│  • Pre-trained on ImageNet (1.2M images)               │
│  • Frozen layers: low-level features (edges, textures)  │
│  • Fine-tuned layers: high-level (lung patterns)        │
│  • Output: 1280-dimensional feature vector              │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│              CLASSIFICATION HEAD                        │
│  GlobalAveragePooling2D                                 │
│         → BatchNorm                                    │
│         → Dense(128, ReLU) + L2 regularization         │
│         → Dropout(0.3)                                  │
│         → Dense(N, Softmax)                             │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                PREDICTION OUTPUT                        │
│  • Class: NORMAL / PNEUMONIA                            │
│  • Confidence: 0.93 (93%)                               │
│  • Probabilities: {NORMAL: 0.07, PNEUMONIA: 0.93}      │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│            EXPLAINABILITY (Grad-CAM)                    │
│  • Backpropagate gradients to last conv layer           │
│  • Weight feature maps by gradient magnitude            │
│  • Generate heatmap → overlay on original image        │
│  • "AI looked HERE to decide PNEUMONIA"                 │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                OUTPUT PRESENTATION                      │
│  Streamlit UI     → Visual dashboard for patients       │
│  FastAPI /predict → JSON for other systems              │
│  Diagnostic Report → Simulated clinical report          │
└─────────────────────────────────────────────────────────┘
```

---

## Data Flow

```
data/raw/
  train/ ─────────────────────────────→ Training (80%)
  val/   ─────────────────────────────→ Validation (20%)
  test/  ─────────────────────────────→ Final evaluation

             ↓ ImageDataGenerator
             ↓ Augmentation (train only)
             ↓ Normalization
             ↓
         MobileNetV2 (imagenet)
             ↓
         Custom Head
             ↓
         Trained Model (.keras)
             ↓
         Evaluate → Metrics, Confusion Matrix, ROC
             ↓
         Grad-CAM → Heatmap
             ↓
         Streamlit UI / FastAPI
```

---

## Module Overview

| Module | File | Purpose |
|--------|------|---------|
| Preprocessing | `src/preprocess.py` | Image loading, augmentation, normalization |
| Model | `src/model.py` | MobileNetV2 architecture, callbacks |
| Training | `src/train.py` | Full training pipeline |
| Evaluation | `src/evaluate.py` | Metrics, confusion matrix, ROC curve |
| Grad-CAM | `src/gradcam.py` | Explainability heatmaps |
| Prediction | `src/predict.py` | Single image & batch inference |
| Streamlit UI | `app/streamlit_app.py` | Web dashboard |
| FastAPI | `app/api.py` | REST API |
| Entry Point | `main.py` | CLI dispatcher |
| Config | `config.yaml` | All settings |

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| Deep Learning | TensorFlow 2.13 / Keras |
| Backbone | MobileNetV2 (ImageNet pretrained) |
| Explainability | Grad-CAM (custom implementation) |
| Image Processing | OpenCV, Pillow, scikit-image |
| Metrics | scikit-learn |
| Web UI | Streamlit |
| REST API | FastAPI + Uvicorn |
| Visualization | Matplotlib, Seaborn, Plotly |
| Dataset | Kaggle Chest X-Ray (Pneumonia) |

---

## Hospital Workflow Simulation

```
Step 1: PATIENT ARRIVAL
  Patient arrives with respiratory symptoms

Step 2: X-RAY CAPTURE
  Radiographer captures chest PA X-ray

Step 3: IMAGE UPLOAD
  Technician uploads JPEG/PNG to MedAI system
  (Streamlit UI or API endpoint)

Step 4: AI PREPROCESSING
  System resizes, normalizes image (< 100ms)

Step 5: AI INFERENCE
  MobileNetV2 extracts features, predicts class

Step 6: EXPLAINABILITY
  Grad-CAM generates heatmap showing suspicious regions

Step 7: RESULT DISPLAY
  Dashboard shows: Prediction + Confidence + Heatmap

Step 8: DOCTOR REVIEW
  Radiologist reviews AI prediction + highlighted regions
  AI assists, doctor makes final decision

Step 9: REPORT GENERATION
  System generates structured diagnostic report

Step 10: PATIENT OUTCOME
  Treatment initiated based on combined AI + doctor decision
```
