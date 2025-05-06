# 🩺 Pneumonia Detection from Chest X-Rays

A deep learning-powered web application to detect pneumonia from chest X-ray images using fine-tuned ResNet50 and Grad-CAM for visual explainability. Built with PyTorch and deployed via Streamlit.

---

## 📌 Features

- 🔍 **Binary Classification**: Detects presence of pneumonia in X-ray images.
- 🧠 **Transfer Learning**: Uses ResNet50 pre-trained on ImageNet and fine-tuned on a medical dataset.
- 🌡️ **Grad-CAM Visualizations**: Highlights the lung regions influencing model predictions.
- ⚡ **Real-time Inference**: Lightweight model with ONNX export for fast CPU predictions.
- 🖼️ **Streamlit App**: Upload images and view predictions and explanations instantly.

---

## 📁 Project Structure

```text
pneumonia_detector/
├── app/
│   ├── streamlit_app.py         # Web interface using Streamlit
│   └── utils.py                 # Image preprocessing, prediction, Grad-CAM
├── data/
│   └── chest_xray/              # Dataset: train/val/test directories (from Kaggle)
├── models/
│   ├── pneumonia_resnet50.pt    # Trained PyTorch model
│   └── pneumonia_resnet50.onnx  # Exported ONNX model for optimized inference
├── visuals/                     # Saved plots: ROC curve, confusion matrix, samples
├── visual.py                    # Visualization script (plots and sample grids)
├── train.py                     # Model training script with validation and export
├── requirements.txt             # Python dependencies
└── README.md                    # Project overview and instructions
```


---

## 📦 Installation

```bash
# Clone the repo
git clone https://github.com/Rukki2705/Pneumonia-Detection-from-Chest-X-Rays.git
cd pneumonia-detector

# Create and activate environment
python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install grad-cam from GitHub (if pip install fails)
pip install git+https://github.com/jacobgil/pytorch-grad-cam.git
```

### 🧪 Dataset

This project uses the [Chest X-Ray Pneumonia dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) from Kaggle.

```
data/chest_xray/train/
data/chest_xray/val/
data/chest_xray/test/
```
### 📈 Results
```
| Metric         | Value     |
|----------------|-----------|
| AUC Score      | 94.2%     |
| F1 Score       | 91.0%     |
| Inference Time | < 200 ms  |
```
---

## 📄 License

This project is intended for **educational and research purposes only**.  
It is **not approved for clinical or diagnostic use**.

---

## 🤝 Acknowledgements

- [NIH Chest X-Ray Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)

---

## 👤 Author

**Hrushikesh Attarde**  
[LinkedIn](https://www.linkedin.com/in/hrushikesh-attarde) · [GitHub](https://github.com/Rukki2705)
