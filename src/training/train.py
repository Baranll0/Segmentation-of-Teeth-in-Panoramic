import os
import torch
from torch.optim import Adam
from torchvision import transforms
from src.preprocessing.dataset import get_data_loader
from src.models.unet import UNet
from src.models.utils import dice_loss
from src.evaluation.visualize import plot_loss_curve, visualize_predictions
from src.evaluation.metrics import evaluate_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(train_loader, model, criterion, optimizer, num_epochs=10, save_dir="/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/outputs/models"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    best_loss = float("inf")
    train_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, (segmentation1, _) in train_loader:
            images, segmentation1 = images.to(device), segmentation1.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, segmentation1)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        print(f"Epoch {epoch + 1}: Loss = {epoch_loss}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/outputs/models/best_model.pth"))

        torch.save(model.state_dict(), os.path.join(save_dir, "/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/outputs/models/last_model.pth"))

    plot_loss_curve(train_losses, filename=os.path.join(save_dir, "/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/outputs/visualizations/loss_curve.png"))

def main():
    images_dir = "/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/dataset/DentalPanoramicXrays/images"
    segmentation1_dir = "/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/dataset/DentalPanoramicXrays/segmentation1"
    segmentation2_dir = "/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/dataset/DentalPanoramicXrays/segmentation2"

    image_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    train_loader = get_data_loader(images_dir, segmentation1_dir, segmentation2_dir, batch_size=16,
                                   image_transform=image_transform, mask_transform=mask_transform)

    model = UNet(in_channels=3, out_channels=1).to(device)
    criterion = dice_loss
    optimizer = Adam(model.parameters(), lr=0.001)

    train_model(train_loader, model, criterion, optimizer, num_epochs=10, save_dir="./outputs/models")

    print("Evaluating model...")
    metrics = evaluate_model(model, train_loader, device)
    print("Metrics:", metrics)

if __name__ == "__main__":
    main()
