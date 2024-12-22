import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
from src.DeepLabV3 import DeepLabV3Plus
from skimage import morphology
import numpy as np
from scipy.ndimage import binary_fill_holes


def load_model(checkpoints_dir, model, device):
    """
    Kaydedilen modeli yükler.
    """
    best_model_path = os.path.join(checkpoints_dir, "best.pth")
    state_dict = torch.load(best_model_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Model loaded from {best_model_path}")
    return model


def preprocess_image(img_path, device):
    """
    Dataset dışından gelen bir görüntüyü yükler ve ön işler.
    """
    transform = Compose([ToTensor(), Normalize(mean=[0.5], std=[0.5])])  # Dataset ile aynı işlemler
    image = Image.open(img_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    return image, image_tensor


def postprocess_mask(predicted_mask):
    """
    Tahmin maskesini işleyerek küçük noktaları kaldırır ve dişleri daha belirgin hale getirir.
    """
    # Küçük objeleri kaldır
    processed_mask = morphology.remove_small_objects(predicted_mask.astype(bool), min_size=500)

    # Küçük delikleri doldur
    processed_mask = binary_fill_holes(processed_mask)

    # Morfolojik işlemler
    processed_mask = morphology.dilation(processed_mask, morphology.disk(5))  # Dilate işlemi
    processed_mask = morphology.erosion(processed_mask, morphology.disk(3))  # Erode işlemi

    return processed_mask


def visualize_single_prediction(model, img_path, device, output_dir):
    """
    Verilen bir görüntü için modelin tahminini görselleştirir ve kaydeder.
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    # Görüntü işleme
    original_image, image = preprocess_image(img_path, device)

    with torch.no_grad():
        output = model(image)
        predicted_mask = torch.argmax(output, dim=1).cpu().numpy()[0]

        # Post-process tahmin maskesi
        processed_mask = postprocess_mask(predicted_mask)

        # Görselleştirme
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.title("Input Image")
        plt.imshow(original_image)
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.title("Predicted Mask (Raw)")
        plt.imshow(predicted_mask, cmap="nipy_spectral")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.title("Post-Processed Mask")
        plt.imshow(processed_mask, cmap="nipy_spectral")
        plt.axis("off")

        sample_path = os.path.join(output_dir, f"processed_{os.path.basename(img_path)}.png")
        plt.savefig(sample_path)
        plt.close()
        print(f"Processed prediction saved to {sample_path}")


if __name__ == "__main__":
    # Yollar
    checkpoints_dir = "/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/checkpoints"
    output_dir = "/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/visualizations"
    img_path = "/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/dataset/DentalPanoramicXrays/images/3.png"  # Dışarıdan yüklenen görüntü yolu

    # Hyperparameters
    num_classes = 33  # Eğitim sırasında kullanılan sınıf sayısı
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model = DeepLabV3Plus(input_channels=3, num_classes=num_classes).to(device)

    # Load Model
    model = load_model(checkpoints_dir, model, device)

    # Visualize Prediction
    visualize_single_prediction(model, img_path, device, output_dir)
