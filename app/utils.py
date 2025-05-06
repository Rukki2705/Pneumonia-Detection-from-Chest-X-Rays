import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from PIL import Image
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path):
    model = models.resnet50(pretrained=False)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(128, 1)
    )
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval().to(DEVICE)
    return model

def preprocess_image(image_pil):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485], [0.229])
    ])
    return transform(image_pil).unsqueeze(0).to(DEVICE)

def predict(model, input_tensor):
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()
        pred = 1 if prob >= 0.5 else 0
    return pred, prob

def generate_gradcam(model, input_tensor, original_pil):
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image

    target_layers = [model.layer4[-1]]
    
    cam = GradCAM(model=model, target_layers=target_layers) 

    grayscale_cam = cam(input_tensor=input_tensor, targets=[BinaryClassifierOutputTarget(1)])
    grayscale_cam = grayscale_cam[0, :]

    rgb_img = np.array(original_pil.resize((224, 224))) / 255.0
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    return visualization
