import os, json
from pathlib import Path
import requests
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import io

BASE_DIR = Path(__file__).parent
# Kandidat lokasi lokal (relatif repo) untuk model & label
MODEL_CANDIDATES = [
    BASE_DIR / "models" / "mobilenetv2_tbc_finetuned_best.keras",  # Prioritas fine-tuned
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

def is_likely_chest_xray(img_pil):
    """Heuristik ringan untuk memeriksa apakah gambar tampak seperti X-ray dada.
    Mengembalikan (bool, detail_str)."""
    try:
        rgb = np.array(img_pil.convert("RGB"))
        # Ukuran kecil untuk analisis cepat
        small = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)

        # 1) Saturasi rendah (X-ray dominan grayscale)
        hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
        sat_mean = float(np.mean(hsv[..., 1]) / 255.0)

        # 2) Perbedaan kanal kecil (grayscale): std antar channel rendah
        channel_std = float(np.mean(np.std(small.astype(np.float32), axis=2)))

        # 3) Ketajaman/struktur (variance of Laplacian)
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # 4) Kecerahan wajar
        brightness = float(np.mean(gray))  # 0..255

        reasons = []
        ok = True
        if sat_mean > 0.25:
            ok = False; reasons.append(f"Saturasi tinggi ({sat_mean:.2f})")
        if channel_std > 18.0:
            ok = False; reasons.append(f"Perbedaan kanal besar ({channel_std:.1f})")
        if lap_var < 20.0:
            ok = False; reasons.append(f"Detail rendah (Laplacian var {lap_var:.1f})")
        if not (10.0 <= brightness <= 245.0):
            ok = False; reasons.append(f"Kecerahan ekstrem ({brightness:.1f})")

        detail = (
            f"sat_mean={sat_mean:.2f}, channel_std={channel_std:.1f}, "
            f"lap_var={lap_var:.1f}, brightness={brightness:.1f}"
        )
        if not ok and not reasons:
            reasons.append("Pola tidak menyerupai X-ray")
        return ok, ("; ".join(reasons) + f" | {detail}") if not ok else detail
    except Exception as e:
        return False, f"Deteksi gagal: {e}"

def predict_image(model, img_pil, threshold=0.5):
    img = img_pil.convert("RGB").resize(IMG_SIZE)
    x = np.array(img, dtype=np.float32)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)  # sama seperti training
    prob = float(model.predict(x, verbose=0)[0, 0])
    cls = 1 if prob >= threshold else 0
    return cls, prob, x[0]  # Return original image array for Grad-CAM

def generate_gradcam(model, img_array, last_conv_layer_name='Conv_1'):
    """Generate Grad-CAM heatmap for the input image"""
    try:
        # Find the base model (MobileNetV2) within the model
        base_in_graph = None
        for lyr in model.layers:
            if hasattr(lyr, 'layers') and len(getattr(lyr, 'layers', [])) > 0:
                base_in_graph = lyr
                break
        
        if base_in_graph is None:
            return None
            
        # Get the last conv layer
        try:
            last_conv_layer = base_in_graph.get_layer(last_conv_layer_name)
        except ValueError:
            last_conv_layer_name = 'out_relu'
            last_conv_layer = base_in_graph.get_layer(last_conv_layer_name)
        
        # Build grad model
        base_input = base_in_graph.input
        conv_target = last_conv_layer.output
        x_head = base_in_graph.output
        
        # Reconstruct head layers
        start_idx = [i for i, lyr in enumerate(model.layers) if lyr is base_in_graph][0] + 1
        for lyr in model.layers[start_idx:]:
            x_head = lyr(x_head)
        
        grad_model = tf.keras.models.Model(inputs=base_input, outputs=[conv_target, x_head])
        
        # Generate Grad-CAM
        img_batch = np.expand_dims(img_array, axis=0)
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_batch, training=False)
            loss = predictions[:, 0]
        
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        # Normalize using TensorFlow ops to keep as Tensor, then convert to NumPy
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()
    except Exception as e:
        st.warning(f"Grad-CAM generation failed: {e}")
        return None

st.set_page_config(page_title="Deteksi TBC - MobileNetV2", page_icon="🫁", layout="wide")
st.title("🫁 Deteksi Tuberkulosis (X-ray Dada)")

# Sidebar controls
st.sidebar.header("⚙️ Pengaturan")
threshold = st.sidebar.slider("Threshold positif TBC", 0.0, 1.0, 0.25, 0.01)
show_gradcam = st.sidebar.checkbox("Tampilkan Grad-CAM", value=True)
model_choice = st.sidebar.selectbox("Model", ["Fine-tuned (Recommended)", "Baseline"])

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Upload Gambar")
    uploaded = st.file_uploader("Unggah gambar X-ray (.png/.jpg)", type=["png", "jpg", "jpeg"])
    
    if uploaded is not None:
        try:
            with st.spinner("Memuat model..."):
                model, LABEL_MAP = load_artifacts()
        except Exception as e:
            st.error(f"Gagal memuat model/label: {e}")
            st.stop()

        img = Image.open(uploaded)
        st.image(img, caption="Input Image", use_container_width=True)

        # Validasi: cek apakah gambar tampak seperti X-ray
        ok_xray, xray_detail = is_likely_chest_xray(img)
        if not ok_xray:
            st.warning(
                "Gambar tidak terdeteksi sebagai X-ray dada. "
                "Pastikan mengunggah foto X-ray dada."
            )
            with st.expander("Detail deteksi"):
                st.code(xray_detail)
            proceed = st.checkbox("Tetap lanjut prediksi (override)", value=False)
            if not proceed:
                st.stop()
        
        with st.spinner("Memprediksi..."):
            cls, prob, img_array = predict_image(model, img, threshold)
        
        # Normalize label map access
        try:
            cls_name = LABEL_MAP[str(cls)] if isinstance(LABEL_MAP, dict) else ("tuberculosis" if cls == 1 else "normal")
        except Exception:
            cls_name = LABEL_MAP.get(cls, "tuberculosis" if cls == 1 else "normal") if isinstance(LABEL_MAP, dict) else ("tuberculosis" if cls == 1 else "normal")

with col2:
    if uploaded is not None:
        st.header("📊 Hasil Prediksi")
        
        # Confidence visualization
        confidence = prob if cls == 1 else (1 - prob)
        st.metric("Kelas Prediksi", cls_name, f"{confidence:.1%} confidence")
        
        # Probability bar
        st.progress(prob)
        st.caption(f"Probabilitas TBC: {prob:.4f}")
        st.caption(f"Threshold: {threshold:.2f}")
        
        # Risk assessment
        if cls == 1:
            if prob > 0.8:
                st.error("🔴 Risiko Tinggi - Konsultasi segera diperlukan")
            elif prob > 0.5:
                st.warning("🟡 Risiko Sedang - Perlu evaluasi lebih lanjut")
            else:
                st.info("🟢 Risiko Rendah - Tetap pantau kondisi")
        else:
            st.success("✅ Normal - Tidak terdeteksi tanda TBC")
        
        # Grad-CAM visualization
        if show_gradcam:
            st.header("🔍 Grad-CAM Analysis")
            with st.spinner("Generating heatmap..."):
                heatmap = generate_gradcam(model, img_array)
            
            if heatmap is not None:
                # Create overlay
                img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                img_cv = cv2.resize(img_cv, IMG_SIZE)
                heatmap_resized = cv2.resize(heatmap, IMG_SIZE)
                heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
                heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
                overlay = np.uint8(0.4 * heatmap_color + 0.6 * img_cv)
                
                st.image(overlay, caption="Grad-CAM Heatmap (Red = High Attention)", use_container_width=True)
            else:
                st.warning("Grad-CAM tidak tersedia untuk model ini")

# Footer
st.markdown("---")
st.caption("Model: MobileNetV2 Fine-tuned | Preprocessing: mobilenet_v2.preprocess_input | Framework: TensorFlow + Streamlit")