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
import matplotlib.pyplot as plt

def get_num_classes(meta_file):
    with open(meta_file, 'r') as f:
        data = json.load(f)
    return len(data['classes'])

def visualize_samples(model, data_loader, device):
    """
    Orijinal görüntü, gerçek maske ve tahmin maskesini gösterir.
    """
    model.eval()
    with torch.no_grad():
        for images, masks in data_loader:
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            predicted = torch.argmax(outputs, dim=1).cpu().numpy()

            # İlk batch'ten bir örnek al
            image = images[0].cpu().numpy().transpose(1, 2, 0)
            true_mask = masks[0].cpu().numpy()
            pred_mask = predicted[0]

            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.title("Input Image")
            plt.imshow((image - image.min()) / (image.max() - image.min()))
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.title("True Mask")
            plt.imshow(true_mask, cmap="nipy_spectral")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.title("Predicted Mask")
            plt.imshow(pred_mask, cmap="nipy_spectral")
            plt.axis("off")

            plt.show()
            break

def train_model(model, train_loader, criterion, optimizer, device, epochs, checkpoints_dir):
    """
    Modeli eğitir, checkpoint kaydeder ve görselleştirme yapar.
    """
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        loop = tqdm(train_loader, leave=True)
        for images, masks in loop:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
            loop.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

        # Save 'last.pth'
        torch.save(model.state_dict(), os.path.join(checkpoints_dir, "last.pth"))

        # Save 'best3.pth'
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(checkpoints_dir, "best3.pth"))
            print("Best model saved!")

        # Visualize sample predictions after each epoch
        visualize_samples(model, train_loader, device)

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
    train_model(model, train_loader, criterion, optimizer, device, epochs, checkpoints_dir)
