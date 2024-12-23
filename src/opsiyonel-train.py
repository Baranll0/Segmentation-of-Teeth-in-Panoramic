import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.DeepLabV3 import DeepLabV3Plus
from src.dataset import MultiClassTeethDataset
from src.losses import CombinedLoss
from tqdm import tqdm
from utils import calculate_metrics, log_metrics, save_metrics_plot, save_sample_visualizations


def get_num_classes(meta_file):
    with open(meta_file, 'r') as f:
        data = json.load(f)
    return len(data['classes'])


def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs, checkpoints_dir, num_classes):
    """
    Modeli eğitir, checkpoint kaydeder ve görselleştirme yapar.
    """
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_dice": [],
        "val_dice": []
    }

    best_loss = float('inf')
    best_dice = 0.0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # Training Phase
        model.train()
        train_loss = 0.0
        train_metrics = {"mean_dice": 0}
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

            loop.set_description(f"Train [{epoch + 1}/{epochs}]")
            loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)
        avg_train_dice = train_metrics["mean_dice"] / len(train_loader)
        history["train_loss"].append(avg_train_loss)
        history["train_dice"].append(avg_train_dice)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_metrics = {"mean_dice": 0}
        with torch.no_grad():
            loop = tqdm(val_loader, leave=True)
            for images, masks in loop:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                metrics = calculate_metrics(outputs, masks, num_classes)
                val_metrics["mean_dice"] += metrics["mean_dice"]

                loop.set_description(f"Val [{epoch + 1}/{epochs}]")
                loop.set_postfix(loss=loss.item())

        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_metrics["mean_dice"] / len(val_loader)
        history["val_loss"].append(avg_val_loss)
        history["val_dice"].append(avg_val_dice)

        # Log Metrics
        log_metrics(epoch, {
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "train_dice": avg_train_dice,
            "val_dice": avg_val_dice
        }, os.path.join(checkpoints_dir, "metrics.log"))

        # Save Checkpoints
        torch.save(model.state_dict(), os.path.join(checkpoints_dir, "last.pth"))
        if avg_val_loss < best_loss and avg_val_dice > best_dice:
            best_loss = avg_val_loss
            best_dice = avg_val_dice
            torch.save(model.state_dict(), os.path.join(checkpoints_dir, "best.pth"))
            print(f"Best model saved! Loss: {avg_val_loss:.4f}, Dice: {avg_val_dice:.4f}")
        else:
            print(
                f"Model not saved. Loss: {avg_val_loss:.4f} (Best: {best_loss:.4f}), Dice: {avg_val_dice:.4f} (Best: {best_dice:.4f})")

    # Save Metrics Plot
    save_metrics_plot(history, checkpoints_dir)

    # Save Visualizations
    save_sample_visualizations(model, val_loader, device, checkpoints_dir)


if __name__ == "__main__":
    # Paths
    data_dir = "/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/data/split_data"
    meta_file = "/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/dataset/datasetkaggle/Teeth Segmentation JSON/meta.json"

    train_image_dir = os.path.join(data_dir, "train/images")
    train_mask_dir = os.path.join(data_dir, "train/masks")
    val_image_dir = os.path.join(data_dir, "val/images")
    val_mask_dir = os.path.join(data_dir, "val/masks")

    checkpoints_dir = "/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/checkpoints"
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Hyperparameters
    batch_size = 2
    learning_rate = 0.0001
    epochs = 10
    num_classes = get_num_classes(meta_file)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset and DataLoader
    train_dataset = MultiClassTeethDataset(train_image_dir, train_mask_dir)
    val_dataset = MultiClassTeethDataset(val_image_dir, val_mask_dir)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model, Loss, Optimizer
    model = DeepLabV3Plus(input_channels=3, num_classes=num_classes).to(device)
    criterion = CombinedLoss(alpha=0.5)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training
    train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs, checkpoints_dir, num_classes)
