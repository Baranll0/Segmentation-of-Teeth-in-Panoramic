import gradio as gr
import torch
import os
import matplotlib.pyplot as plt
from PIL import Image
from src.DeepLabV3 import DeepLabV3Plus
import numpy as np

# Model yükleme fonksiyonu
def load_model(checkpoints_dir, model, device):
    best_model_path = os.path.join(checkpoints_dir, "best.pth")
    state_dict = torch.load(best_model_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Model loaded from {best_model_path}")
    return model

# Görüntü işleme, tahmin yapma ve toplam diş sayısını hesaplama
def predict_and_visualize(image, model, device, num_classes=33):
    model.eval()
    with torch.no_grad():
        # Görüntüyü işleme
        image = image.convert("RGB")
        image_resized = image.resize((512, 512))  # Modelin giriş boyutuna göre ayarla
        image_tensor = torch.tensor(np.array(image_resized).transpose(2, 0, 1)).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(device)

        # Tahmin yapma
        output = model(image_tensor)
        predicted_mask = torch.argmax(output, dim=1).cpu().numpy()[0]

        # Her farklı sınıf için diş sayısını hesaplama
        unique_classes = np.unique(predicted_mask)
        tooth_count = len(unique_classes) - 1  # Arka planı (class 0) çıkartıyoruz

        # Görselleştirme
        plt.figure(figsize=(16, 8))  # Daha büyük bir görselleştirme
        plt.subplot(1, 2, 1)
        plt.title("Input Image")
        plt.imshow(image_resized)
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title(f"Predicted Mask (Teeth Count: {tooth_count})")
        plt.imshow(predicted_mask, cmap="nipy_spectral")
        plt.axis("off")

        # Görüntüyü kaydetme
        output_image_path = "output_visualization.png"
        plt.savefig(output_image_path)
        plt.close()

        return output_image_path, tooth_count

# Gradio arayüzü
def gradio_interface(image):
    result_image, tooth_count = predict_and_visualize(image, model, device)
    return result_image, f"Total Teeth Count: {tooth_count}"

if __name__ == "__main__":
    # Model ve ayarlar
    checkpoints_dir = "/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/checkpoints"
    num_classes = 33
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DeepLabV3Plus(input_channels=3, num_classes=num_classes).to(device)
    model = load_model(checkpoints_dir, model, device)

    # Gradio arayüzü oluşturma
    interface = gr.Interface(
        fn=gradio_interface,
        inputs=gr.Image(type="pil"),
        outputs=[gr.Image(type="filepath"), gr.Text()],
        title="Teeth Segmentation",
        description="Upload a dental image to segment teeth and count them."
    )
    interface.launch()
