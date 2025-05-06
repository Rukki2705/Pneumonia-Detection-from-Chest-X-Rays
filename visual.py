import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay


os.makedirs("visuals", exist_ok=True)

# --------------------------------------
# 1. Sample Image Grid (Normal & Pneumonia)
# --------------------------------------
def plot_sample_images(dataset_path, label="NORMAL", num_images=6, save_path=None):
    path = os.path.join(dataset_path, label)
    images = os.listdir(path)[:num_images]

    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    for i, img in enumerate(images):
        img_path = os.path.join(path, img)
        image = Image.open(img_path)
        axes[i].imshow(image.convert("L"), cmap="gray")
        axes[i].set_title(label)
        axes[i].axis('off')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # ✅ ensure visuals/ exists
        plt.savefig(save_path)
        print(f"✅ Saved to {save_path}")
    plt.show()


# --------------------------------------
# 2. ROC Curve
# --------------------------------------
def plot_auc_roc(y_true, y_probs, save_path=None):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}', color='darkorange', lw=2)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        print(f"✅ Saved to {save_path}")
    plt.show()

# --------------------------------------
# 3. Confusion Matrix
# --------------------------------------
def plot_confusion_matrix(y_true, y_pred, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Pneumonia"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    if save_path:
        plt.savefig(save_path)
        print(f"✅ Saved to {save_path}")
    plt.show()

# --------------------------------------
# Run Examples (Uncomment to use)
# --------------------------------------
if __name__ == "__main__":
    # Visualize Normal and Pneumonia samples
    plot_sample_images("data/chest_xray/train", label="NORMAL", save_path="visuals/sample_normal.png")
    plot_sample_images("data/chest_xray/train", label="PNEUMONIA", save_path="visuals/sample_pneumonia.png")

    # Dummy data for example (replace with real values)
    y_true = [0, 0, 1, 1, 1, 0, 1, 0]
    y_probs = [0.1, 0.3, 0.8, 0.65, 0.95, 0.2, 0.9, 0.4]
    y_pred = [int(p >= 0.5) for p in y_probs]

    plot_auc_roc(y_true, y_probs, save_path="visuals/roc_curve.png")
    plot_confusion_matrix(y_true, y_pred, save_path="visuals/confusion_matrix.png")
