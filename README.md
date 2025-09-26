# 🫁 Deteksi Tuberkulosis dengan MobileNetV2

Sistem deteksi tuberkulosis menggunakan deep learning dengan MobileNetV2 untuk analisis X-ray dada. Dilengkapi dengan Grad-CAM untuk interpretabilitas model.

## 🚀 Demo Online
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

## 📊 Dataset
- **Normal**: 3,093 gambar
- **Tuberculosis**: 700 gambar  
- **Total**: 3,793 gambar
- **Format**: PNG
- **Ukuran**: 224×224 pixels
- **Split**: 80% training, 20% validation

## 🤖 Model Architecture
- **Base Model**: MobileNetV2 (ImageNet pre-trained)
- **Transfer Learning**: Frozen base + custom head
- **Fine-tuning**: Unfreeze last layers for better performance
- **Input**: 224×224×3 RGB images
- **Output**: Binary classification (normal/tuberculosis)
- **Preprocessing**: MobileNetV2 standard preprocessing

## 📈 Performance
- **Accuracy**: ~79% (baseline), ~93% (fine-tuned)
- **Sensitivity**: Optimized with threshold tuning
- **Interpretability**: Grad-CAM heatmaps for model explanation

## 🛠️ Installation & Usage

### Local Development
```bash
# Clone repository
git clone https://github.com/Akamalll/DETEKSI-TBC.git
cd DETEKSI-TBC

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
```

### Docker Deployment
```bash
# Build and run with Docker
docker build -t tbc-detection .
docker run -p 8501:8501 tbc-detection

# Or use Docker Compose
docker-compose up -d
```

### Streamlit Cloud Deployment
1. Fork this repository
2. Connect to [Streamlit Cloud](https://streamlit.io/cloud)
3. Deploy from GitHub repository
4. Set secrets (if using external model URLs):
   ```
   MODEL_URL = https://your-model-url.keras
   LABELS_URL = https://your-labels-url.json
   ```

## 📁 Project Structure
```
DETEKSI-TBC/
├── app.py                          # Streamlit web application
├── predict.py                      # Command-line prediction script
├── TBC_MobileNetV2.ipynb          # Training notebook
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Docker configuration
├── docker-compose.yml             # Docker Compose setup
├── models/                         # Model files for deployment
│   ├── mobilenetv2_tbc_finetuned_best.keras
│   └── labels.json
├── checkpoints/                    # Training checkpoints
├── exports/                        # Exported models and results
└── DATASET/                        # Training data (not in repo)
```

## 🔧 Features

### Web Application
- **Interactive Upload**: Drag & drop X-ray images
- **Real-time Prediction**: Instant classification results
- **Threshold Tuning**: Adjustable sensitivity slider
- **Grad-CAM Visualization**: Model attention heatmaps
- **Risk Assessment**: Clinical interpretation of results
- **Responsive Design**: Works on desktop and mobile

### Model Capabilities
- **Transfer Learning**: Leverages ImageNet pre-trained weights
- **Fine-tuning**: Optimized for medical imaging
- **Interpretability**: Grad-CAM for model explanation
- **Threshold Optimization**: ROC-based threshold selection

## 📊 Training Process

### Baseline Training
```python
# 20 epochs with frozen base model
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

### Fine-tuning
```python
# Unfreeze last layers, lower learning rate
base_model.trainable = True
for layer in base_model.layers[:-120]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

## 🎯 Usage Examples

### Web Interface
1. Upload X-ray image (PNG/JPG)
2. Adjust threshold if needed
3. View prediction results
4. Analyze Grad-CAM heatmap
5. Review risk assessment

### Command Line
```bash
python predict.py --image path/to/xray.png --threshold 0.25
```

### Programmatic
```python
from predict import load_model, predict_image

model = load_model('models/mobilenetv2_tbc_finetuned_best.keras')
cls, prob = predict_image(model, 'xray.png', threshold=0.25)
print(f"Prediction: {cls}, Probability: {prob}")
```

## 🔬 Model Interpretation

### Grad-CAM Analysis
- **Red areas**: High model attention (likely TBC indicators)
- **Blue areas**: Low attention (normal tissue)
- **Overlay**: Combines original image with heatmap

### Threshold Guidelines
- **0.1-0.3**: High sensitivity (catch more cases, more false positives)
- **0.4-0.6**: Balanced sensitivity/specificity
- **0.7-0.9**: High specificity (fewer false positives, may miss cases)

## ⚠️ Medical Disclaimer
This tool is for research and educational purposes only. It should not be used as the sole basis for medical diagnosis. Always consult with qualified healthcare professionals for medical decisions.

## 🤝 Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Contact
- GitHub: [@Akamalll](https://github.com/Akamalll)
- Project: [DETEKSI-TBC](https://github.com/Akamalll/DETEKSI-TBC)

## 🙏 Acknowledgments
- Dataset: Medical imaging datasets
- Base model: MobileNetV2 (Google)
- Framework: TensorFlow, Streamlit
- Visualization: Grad-CAM implementation