import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from src.models.unet import UNet
from src.preprocessing.dataset import load_images_and_masks
import numpy as np
import cv2

def test_pipeline():
    # Paths
    image_dir = "/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/dataset/DentalPanoramicXrays/images"
    mask_dir = "/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/dataset/DentalPanoramicXrays/masks"
    model_path = "/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/outputs/models/unet_best.pth"

    # Load data
    print("Loading images and masks...")
    images, masks = load_images_and_masks(image_dir, mask_dir)

    # Test data loader
    print("Preparing test data...")
    test_dataset = TensorDataset(torch.tensor(images).float(), torch.tensor(masks).float())
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Load model
    print("Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Predict and visualize
    print("Running predictions...")
    for idx, (image, _) in enumerate(test_loader):
        image = image.to(device)  # Only pass the image

        with torch.no_grad():
            output = model(image).squeeze().cpu().numpy()

        # Threshold the prediction
        binary_mask = (output > 0.5).astype(np.uint8)

        # Convert to OpenCV format for overlay
        input_image = image.squeeze(0).squeeze().cpu().numpy()
        input_image = (input_image * 255).astype(np.uint8)  # Rescale to 0-255
        overlay = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)  # Convert to BGR for overlay

        # Create contours from the binary mask
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        overlay_with_contours = cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)  # Draw green contours

        # Visualize input and overlaid output
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.title("Input Image")
        plt.imshow(input_image, cmap="gray")

        plt.subplot(1, 2, 2)
        plt.title("Predicted Overlay")
        plt.imshow(cv2.cvtColor(overlay_with_contours, cv2.COLOR_BGR2RGB))

        plt.show()

        # Stop after visualizing 5 samples
        if idx == 4:
            break

if __name__ == "__main__":
    test_pipeline()
