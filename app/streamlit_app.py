import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
from utils import load_model, preprocess_image, predict, generate_gradcam

# -------------------- Setup --------------------
st.set_page_config(page_title="Pneumonia Detector", layout="centered")
st.title("🩺 Pneumonia Detection from Chest X-Rays")
st.write("Upload a chest X-ray image to predict whether it indicates pneumonia.")

model_path = "models/pneumonia_resnet50.pt"
model = load_model(model_path)

# -------------------- Upload Section --------------------
uploaded_file = st.file_uploader("Choose a Chest X-ray image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        input_tensor = preprocess_image(image)
        pred, prob = predict(model, input_tensor)

        label = "PNEUMONIA 🚨" if pred == 1 else "NORMAL ✅"
        st.markdown(f"### Prediction: `{label}`")
        st.markdown(f"### Confidence: `{prob:.2f}`")

        # -------------------- Grad-CAM --------------------
        st.markdown("### 🔍 Model Explanation (Grad-CAM)")
        cam_overlay = generate_gradcam(model, input_tensor, image)
        st.image(cam_overlay, caption="Grad-CAM Heatmap", use_column_width=True)
