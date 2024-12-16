import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

# Import Models
from model_unetplusplus import NestedUNet
from model_resnet_unet import ResNetUNet

# Paths
output_dir = "/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/data/processed"
checkpoints_dir = "/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/checkpoints"
os.makedirs(checkpoints_dir, exist_ok=True)

# Training parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 100
batch_size = 4
learning_rate = 0.0001


# Loss and metrics
def dice_coefficient(pred, target, smooth=1):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum()
        return 1 - (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)


class TeethDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))

        # Default normalization
        self.transform = Compose([
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize RGB images
        ])
        self.mask_transform = ToTensor()  # No normalization for masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Mask as a single channel

        # Apply transformations
        image = self.transform(image)
        mask = self.mask_transform(mask)

        return image, mask


# Training function
# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    best_iou = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # Training phase
        model.train()
        train_loss, train_dice = 0.0, 0.0

        loop = tqdm(train_loader, leave=False)
        for images, masks in loop:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_dice += dice_coefficient(outputs, masks).item() * images.size(0)

            loop.set_postfix(loss=loss.item())

        train_loss /= len(train_loader.dataset)
        train_dice /= len(train_loader.dataset)

        print(f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")
        torch.save(model.state_dict(), os.path.join(checkpoints_dir, "last.pth"))

        # Save best model
        if train_dice > best_iou:
            best_iou = train_dice
            torch.save(model.state_dict(), os.path.join(checkpoints_dir, "best.pth"))

        # Visualize predictions after each epoch
        print(f"Epoch {epoch + 1}: Tahmin Grafikleri")
        visualize_sample(model, train_loader)

def visualize_sample(model, train_loader):
    model.eval()
    with torch.no_grad():
        for sample_image, sample_mask in train_loader:
            sample_image = sample_image[0].to(device)
            sample_mask = sample_mask[0].cpu().squeeze().numpy()

            pred_mask = model(sample_image.unsqueeze(0))
            pred_mask = torch.sigmoid(pred_mask).cpu().squeeze().numpy()

            # Input image normalization to [0, 1] for visualization
            sample_image = sample_image.cpu().numpy().transpose(1, 2, 0)
            sample_image = (sample_image - sample_image.min()) / (sample_image.max() - sample_image.min())

            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1); plt.title("Input Image"); plt.imshow(sample_image)
            plt.subplot(1, 3, 2); plt.title("True Mask"); plt.imshow(sample_mask, cmap="gray")
            plt.subplot(1, 3, 3); plt.title("Predicted Mask"); plt.imshow(pred_mask, cmap="gray")
            plt.show()
            break

if __name__ == "__main__":
    # Model Selection
    print("Model Seçimi:")
    print("1: UNet++")
    print("2: ResNet Encoder'lı UNet")
    choice = input("Kullanmak istediğiniz modeli seçin (1/2): ")

    # Dataset paths
    train_image_dir = os.path.join(output_dir, "resized_images")
    train_mask_dir = os.path.join(output_dir, "resized_masks")

    train_dataset = TeethDataset(train_image_dir, train_mask_dir)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Model Initialization
    if choice == "1":
        print("UNet++ modeli kullanılıyor...")
        model = NestedUNet(input_channels=3, output_channels=1).to(device)
    elif choice == "2":
        print("ResNet Encoder'lı UNet modeli kullanılıyor...")
        model = ResNetUNet(input_channels=3, output_channels=1).to(device)
    else:
        raise ValueError("Geçersiz seçim! Lütfen 1 veya 2 girin.")

    # Training Setup
    criterion = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_model(model, train_loader, train_loader, criterion, optimizer, epochs)

    # Visualize a sample
    visualize_sample(model, train_loader)
