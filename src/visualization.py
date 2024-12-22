import os
import torch
import matplotlib.pyplot as plt
from src.DeepLabV3 import  DeepLabV3Plus
from src.dataset import MultiClassTeethDataset
from torch.utils.data import DataLoader


def load_model(checkpoints_dir, model, device):
    """
    Kaydedilen modeli yükler.
    """
    best_model_path = os.path.join(checkpoints_dir, "best.pth")
    state_dict = torch.load(best_model_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Model loaded from {best_model_path}")
    return model


def visualize_predictions(model, data_loader, device, output_dir, num_samples=1):
    """
    Modelin tahminlerini görselleştirir ve kaydeder.
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for idx, (images, masks) in enumerate(data_loader):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            predicted = torch.argmax(outputs, dim=1).cpu().numpy()

            # İlk batch'ten num_samples kadar örnek kaydet
            for i in range(min(len(images), num_samples)):
                image = images[i].cpu().numpy().transpose(1, 2, 0)
                true_mask = masks[i].cpu().numpy()
                pred_mask = predicted[i]

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

                sample_path = os.path.join(output_dir, f"sample_{idx}_{i}.png")
                plt.savefig(sample_path)
                plt.close()

            if idx >= 1:  # Sadece ilk batch'i işleyelim
                break


if __name__ == "__main__":
    # Yollar
    data_dir = "/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/data/processed"
    train_image_dir = os.path.join(data_dir, "resized_images")
    train_mask_dir = os.path.join(data_dir, "resized_masks")
    checkpoints_dir = "/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/checkpoints"
    output_dir = "/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/visualizations"

    # Hyperparameters
    batch_size = 4
    num_classes = 33  # Eğitim sırasında kullanılan sınıf sayısı
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset ve DataLoader
    dataset = MultiClassTeethDataset(train_image_dir, train_mask_dir)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Model
    # Model
    model = DeepLabV3Plus(input_channels=3, num_classes=num_classes).to(device)


    # Load Model
    model = load_model(checkpoints_dir, model, device)

    # Visualize Predictions
    visualize_predictions(model, data_loader, device, output_dir)
