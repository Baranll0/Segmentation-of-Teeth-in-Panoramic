import os
import json
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import ColorJitter
from src.preprocessing.dataset import load_datasets
from src.models.unet import UNet
from tqdm import tqdm
import numpy as np
import evaluate
import torchvision.transforms as T

# Veri yolları
IMG_DIR = "/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/data/processed/img"
MASK_DIR = "/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/data/processed/mask"
META_FILE = "/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/dataset/datasetkaggle/Teeth Segmentation JSON/meta.json"

# Meta.json'dan sınıf sayısını al
def get_num_classes(meta_file):
    with open(meta_file, "r") as f:
        meta_data = json.load(f)
    return len(meta_data["classes"])

# Sınıf sayısını meta.json'dan al
num_classes = get_num_classes(META_FILE)
print(f"Sınıf Sayısı: {num_classes}")

# Dataset yükleme
dataset = load_datasets(IMG_DIR, MASK_DIR)

# Verileri dönüştürme ve normalize etme
def train_transforms(example_batch):
    images = [T.ToTensor()(x).float() for x in example_batch["pixel_values"]]
    labels = [T.ToTensor()(x).float() for x in example_batch["label"]]
    return {"pixel_values": torch.stack(images), "label": torch.stack(labels)}

def val_transforms(example_batch):
    images = [T.ToTensor()(x).float() for x in example_batch["pixel_values"]]
    labels = [T.ToTensor()(x).float() for x in example_batch["label"]]
    return {"pixel_values": torch.stack(images), "label": torch.stack(labels)}

# Transform'ları uygulama
dataset["train"].set_transform(train_transforms)
dataset["validation"].set_transform(val_transforms)

# DataLoader oluşturma
train_loader = DataLoader(dataset["train"], batch_size=2, shuffle=True)
val_loader = DataLoader(dataset["validation"], batch_size=2)

# Model ve Loss
model = UNet(input_channels=3, num_classes=num_classes).cuda()
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Evaluation Metric
metric = evaluate.load("mean_iou")

def compute_metrics(predictions, labels):
    pred_labels = predictions.argmax(dim=1).detach().cpu().numpy()
    true_labels = labels.detach().cpu().numpy()
    return metric.compute(predictions=pred_labels, references=true_labels, num_labels=num_classes, ignore_index=0)

# Eğitim döngüsü
num_epochs = 20
best_val_loss = float('inf')
output_dir = "/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/outputs"

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}")

    for batch in train_bar:
        images = batch["pixel_values"].cuda()
        labels = batch["label"].cuda().long().squeeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_bar.set_postfix(loss=train_loss / len(train_loader))

    # Validation
    model.eval()
    val_loss = 0.0
    val_bar = tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{num_epochs}")
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in val_bar:
            images = batch["pixel_values"].cuda()
            labels = batch["label"].cuda().long().squeeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            all_preds.append(outputs)
            all_labels.append(labels)

    # Metric hesaplama
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    metrics = compute_metrics(all_preds, all_labels)

    print(
        f"Epoch {epoch + 1}/{num_epochs} -> Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Metrics: {metrics}")

    # En iyi modeli kaydet
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), os.path.join(output_dir, "best.pth"))
        print(f"Best model saved with Val Loss: {val_loss:.4f}")

    # Son modeli kaydet
    torch.save(model.state_dict(), os.path.join(output_dir, "last.pth"))
    print(f"Last model saved at the end of epoch {epoch + 1}")
