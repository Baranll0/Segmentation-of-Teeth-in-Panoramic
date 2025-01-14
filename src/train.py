import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.model import NestedUNet
from src.dataset import MultiClassTeethDataset
from src.losses import CombinedLoss
from tqdm import tqdm
from utils import calculate_metrics, log_metrics, save_metrics_plot, save_sample_visualizations

def get_num_classes(meta_file):
    with open(meta_file, 'r') as f:
        data = json.load(f)
    return len(data['classes'])

def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs, checkpoints_dir, num_classes, min_dice_threshold=0.7):
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_dice": [],
        "val_dice": [],
        "train_f1": [],
        "val_f1": []
    }

    best_loss = float('inf')
    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_loss = 0.0
        train_metrics = {"mean_dice": 0, "mean_f1": 0}
        loop = tqdm(train_loader, leave=True)
        for images, masks in loop:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            metrics = calculate_metrics(outputs, masks, num_classes)
            train_metrics["mean_dice"] += metrics["mean_dice"]
            train_metrics["mean_f1"] += metrics["mean_f1"]

            loop.set_description(f"Epoch [{epoch+1}/{epochs}] (Train)")
            loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)
        history["train_dice"].append(train_metrics["mean_dice"] / len(train_loader))
        history["train_f1"].append(train_metrics["mean_f1"] / len(train_loader))

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_metrics = {"mean_dice": 0, "mean_f1": 0}
        with torch.no_grad():
            loop = tqdm(val_loader, leave=True)
            for images, masks in loop:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                metrics = calculate_metrics(outputs, masks, num_classes)
                val_metrics["mean_dice"] += metrics["mean_dice"]
                val_metrics["mean_f1"] += metrics["mean_f1"]

                loop.set_description(f"Epoch [{epoch+1}/{epochs}] (Val)")
                loop.set_postfix(loss=loss.item())

        avg_val_loss = val_loss / len(val_loader)
        history["val_loss"].append(avg_val_loss)
        history["val_dice"].append(val_metrics["mean_dice"] / len(val_loader))
        history["val_f1"].append(val_metrics["mean_f1"] / len(val_loader))

        # Log Metrics
        log_metrics(epoch, {
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "train_dice": history["train_dice"][-1],
            "val_dice": history["val_dice"][-1],
            "train_f1": history["train_f1"][-1],
            "val_f1": history["val_f1"][-1]
        }, os.path.join(checkpoints_dir, "metrics.log"))

        # Save Model Checkpoints
        torch.save(model.state_dict(), os.path.join(checkpoints_dir, "last.pth"))
        if avg_val_loss < best_loss and val_metrics["mean_dice"] / len(val_loader) > min_dice_threshold:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(checkpoints_dir, "best3.pth"))
            print("Best model saved based on loss and mean_dice!")

    # Save Metrics and Visualizations
    save_metrics_plot(history, checkpoints_dir)
    save_sample_visualizations(model, val_loader, device, checkpoints_dir)


if __name__ == "__main__":
    # Yollar
    data_dir = "/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/data/processed"
    meta_file = "/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/dataset/datasetkaggle/Teeth Segmentation JSON/meta.json"
    train_image_dir = os.path.join(data_dir, "resized_images")
    train_mask_dir = os.path.join(data_dir, "resized_masks")

    checkpoints_dir = "/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/checkpoints"
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Hyperparameters
    batch_size = 4
    learning_rate = 0.001
    epochs = 200
    num_classes = get_num_classes(meta_file)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset ve DataLoader
    train_dataset = MultiClassTeethDataset(train_image_dir, train_mask_dir)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Model, Loss, Optimizer
    model = NestedUNet(input_channels=3, output_channels=num_classes).to(device)
    criterion = CombinedLoss(alpha=0.5)  # CrossEntropy ve Dice Loss karışımı
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training
    train_model(model, train_loader, criterion, optimizer, device, epochs, checkpoints_dir, num_classes)
