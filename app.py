import os, json
from pathlib import Path
import requests
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image

BASE_DIR = Path(__file__).parent
# Kandidat lokasi lokal (relatif repo) untuk model & label
MODEL_CANDIDATES = [
    BASE_DIR / "models" / "mobilenetv2_tbc.keras",
    BASE_DIR / "checkpoints" / "mobilenetv2_tbc_finetuned_best.keras",
    BASE_DIR / "exports" / "mobilenetv2_tbc.keras",
]
LABELS_CANDIDATES = [
    BASE_DIR / "models" / "labels.json",
    BASE_DIR / "exports" / "labels.json",
]
IMG_SIZE = (224, 224)

def _first_existing(paths):
    for p in paths:
        if Path(p).exists():
            return str(p)
    return None

@st.cache_resource
def ensure_artifacts():
    # 1) Coba file lokal di repo
    model_path = _first_existing(MODEL_CANDIDATES)
    labels_path = _first_existing(LABELS_CANDIDATES)

    # 2) Jika tidak ada, coba unduh dari Secrets (MODEL_URL, LABELS_URL)
    if model_path is None:
        model_url = st.secrets.get("MODEL_URL")
        if not model_url:
            raise FileNotFoundError(
                f"Model tidak ditemukan di {MODEL_CANDIDATES}. Set MODEL_URL di Secrets atau commit model ke folder models/."
            )
        target = BASE_DIR / "models" / "mobilenetv2_tbc.keras"
        target.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(model_url, stream=True) as r:
            r.raise_for_status()
            with open(target, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    if chunk:
                        f.write(chunk)
        model_path = str(target)

    if labels_path is None:
        labels_url = st.secrets.get("LABELS_URL")
        if not labels_url:
            raise FileNotFoundError(
                f"Label tidak ditemukan di {LABELS_CANDIDATES}. Set LABELS_URL di Secrets atau commit labels.json ke folder models/."
            )
        target = BASE_DIR / "models" / "labels.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(labels_url, stream=True) as r:
            r.raise_for_status()
            with open(target, "wb") as f:
                f.write(r.content)
        labels_path = str(target)

    return model_path, labels_path

@st.cache_resource
def load_artifacts():
    model_path, labels_path = ensure_artifacts()
    model = tf.keras.models.load_model(model_path)
    with open(labels_path, "r") as f:
        labels = json.load(f)  # e.g. {"0":"normal","1":"tuberculosis"}
    return model, labels

def predict_image(model, img_pil):
    img = img_pil.convert("RGB").resize(IMG_SIZE)
    x = np.array(img, dtype=np.float32)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)  # sama seperti training
    prob = float(model.predict(x, verbose=0)[0, 0])
    cls = 1 if prob >= 0.5 else 0
    return cls, prob

st.set_page_config(page_title="Deteksi TBC - MobileNetV2", page_icon="🫁")
st.title("🫁 Deteksi Tuberkulosis (X-ray Dada)")

# Kontrol sensitivitas (threshold) di sidebar
threshold = st.sidebar.slider("Threshold positif TBC", 0.0, 1.0, 0.25, 0.01)
st.info("Unggah gambar X-ray untuk mulai melakukan prediksi.")

uploaded = st.file_uploader("Unggah gambar X-ray (.png/.jpg)", type=["png", "jpg", "jpeg"])
if uploaded is not None:
    try:
        with st.spinner("Memuat model..."):
            model, LABEL_MAP = load_artifacts()
    except Exception as e:
        st.error(f"Gagal memuat model/label: {e}")
        st.stop()

    img = Image.open(uploaded)
    st.image(img, caption="Input", use_column_width=True)
    with st.spinner("Memprediksi..."):
        cls_tmp, prob = predict_image(model, img)
        cls = 1 if prob >= threshold else 0
    # Normalisasi akses label map
    try:
        cls_name = LABEL_MAP[str(cls)] if isinstance(LABEL_MAP, dict) else ("tuberculosis" if cls == 1 else "normal")
    except Exception:
        # Fallback jika label map berupa {0:"normal",1:"tuberculosis"} dengan key int
        cls_name = LABEL_MAP.get(cls, "tuberculosis" if cls == 1 else "normal") if isinstance(LABEL_MAP, dict) else ("tuberculosis" if cls == 1 else "normal")
    st.subheader(f"Hasil: {cls_name}")
    st.write(f"Probabilitas (positif): {prob:.4f}")
    st.write(f"Threshold: {threshold:.2f}")

st.caption("Model: MobileNetV2, preprocessing: mobilenet_v2.preprocess_input")