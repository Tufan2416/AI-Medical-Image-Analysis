# Deployment Guide
## AI-Powered Medical Image Analysis System

---

## Option 1: Streamlit Cloud (FREE — Recommended for Students)

### Steps:

1. **Push your code to GitHub**
   ```bash
   git push origin main
   ```

2. **Go to** https://share.streamlit.io

3. **Click** "New app"

4. **Fill in:**
   - Repository: `yourusername/AI-Medical-Image-Analysis`
   - Branch: `main`
   - Main file path: `app/streamlit_app.py`

5. **Add Secrets** (Settings → Secrets):
   ```toml
   # No secrets needed for this project
   ```

6. **Click Deploy** → Your app is live!

**URL format:** `https://yourusername-ai-medical-image-analysis-app-streamlit-app-xyz.streamlit.app`

> **Note:** Upload your trained model (`models/best_model.keras`) to GitHub using Git LFS or host it on HuggingFace Hub.

---

## Option 2: HuggingFace Spaces (FREE — Great for AI projects)

1. **Create account** at https://huggingface.co

2. **Create a new Space:**
   - Name: `AI-Medical-Image-Analysis`
   - SDK: Streamlit
   - Visibility: Public

3. **Upload files via UI or Git:**
   ```bash
   git clone https://huggingface.co/spaces/yourusername/AI-Medical-Image-Analysis
   # Copy your files into the cloned repo
   git add .
   git commit -m "Initial deployment"
   git push
   ```

4. **HuggingFace auto-builds** your Streamlit app!

---

## Option 3: Render (FREE — For FastAPI)

1. **Go to** https://render.com → New → Web Service

2. **Connect GitHub repo**

3. **Build Command:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start Command:**
   ```bash
   uvicorn app.api:app --host 0.0.0.0 --port $PORT
   ```

5. **Click Deploy** → Get your API URL

---

## Option 4: Local Docker

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501 8000

# Run both Streamlit UI and FastAPI
CMD ["bash", "-c", "streamlit run app/streamlit_app.py --server.port 8501 & uvicorn app.api:app --port 8000"]
```

```bash
# Build and run
docker build -t ai-medical-analysis .
docker run -p 8501:8501 -p 8000:8000 ai-medical-analysis
```

---

## Model Hosting (for large model files)

Since `best_model.keras` (~10-50MB) may be too large for GitHub:

### HuggingFace Hub:
```python
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj="models/best_model.keras",
    path_in_repo="best_model.keras",
    repo_id="yourusername/ai-medical-models",
    repo_type="model"
)
```

### Git LFS (GitHub):
```bash
git lfs install
git lfs track "*.keras"
git lfs track "*.h5"
git add .gitattributes
git add models/best_model.keras
git commit -m "Add trained model via LFS"
git push
```

---

## Environment Variables

Create `.env` file (never commit this):
```env
MODEL_PATH=models/best_model.keras
DATA_DIR=data/raw
LOG_LEVEL=INFO
```

Load in Python:
```python
from dotenv import load_dotenv
load_dotenv()
```

---

## Performance Tips for Deployment

1. **Cache the model** in Streamlit:
   ```python
   @st.cache_resource
   def load_model(): ...
   ```

2. **Reduce model size** (optional):
   ```python
   # Convert to TensorFlow Lite for faster inference
   converter = tf.lite.TFLiteConverter.from_keras_model(model)
   tflite_model = converter.convert()
   with open('models/model.tflite', 'wb') as f:
       f.write(tflite_model)
   ```

3. **Use smaller input size** (160×160 instead of 224×224) for faster API response
