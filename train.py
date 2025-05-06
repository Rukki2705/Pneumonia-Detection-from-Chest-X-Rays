import os
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from sklearn.metrics import roc_auc_score
import argparse

# ----------------------------- Configuration -----------------------------
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "data/chest_xray"

# ----------------------------- Data Preparation -----------------------------
def get_dataloaders(data_dir):
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485], [0.229])
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485], [0.229])
    ])

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform_train)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader

# ----------------------------- Model Setup -----------------------------
def build_model():
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False  # freeze all layers
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(128, 1)
    )
    return model.to(DEVICE)

# ----------------------------- Training & Evaluation -----------------------------
def evaluate(model, dataloader):
    model.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE).float()
            outputs = model(images).squeeze()
            probs = torch.sigmoid(outputs)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    auc = roc_auc_score(all_labels, all_probs)
    return auc

def train():
    train_loader, val_loader = get_dataloaders(DATA_DIR)
    model = build_model()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

    best_auc = 0.0
    os.makedirs("models", exist_ok=True)

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE).float().unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_auc = evaluate(model, val_loader)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {avg_loss:.4f}, Val AUC: {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), "models/pneumonia_resnet50.pt")
            print("✅ Model saved.")

if __name__ == "__main__":
    train()
