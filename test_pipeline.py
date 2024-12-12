import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from src.models.unet import UNet
from src.preprocessing.dataset import load_images_and_masks
from src.inference.postprocess import CCA_Analysis
import numpy as np
import cv2

def test_pipeline():
    """
    Test pipeline for evaluating a trained UNet model on test images and visualizing predictions.

    Steps:
        1. Load the test dataset (images and masks).
        2. Prepare a DataLoader for test data.
        3. Load the trained UNet model from the specified path.
        4. Perform predictions for the test images.
        5. Apply postprocessing (CCA and morphological operations).
        6. Visualize the original input image, ground truth mask, predicted segmentation, and overlay.

    Outputs:
        - Visualizations of input images with predicted segmentation overlays and ground truth masks.

    Returns:
        None
    """
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
    for idx, (image, ground_truth_mask) in enumerate(test_loader):
        image = image.to(device)  # Only pass the image

        with torch.no_grad():
            output = model(image).squeeze().cpu().numpy()

        # Threshold the prediction
        binary_mask = (output > 0.5).astype(np.uint8)

        # Convert to OpenCV format for overlay
        input_image = image.squeeze(0).squeeze().cpu().numpy()
        input_image = (input_image * 255).astype(np.uint8)  # Rescale to 0-255

        # Apply postprocessing
        processed_image, teeth_count = CCA_Analysis(input_image, binary_mask)

        # Ground truth mask for visualization
        gt_mask = ground_truth_mask.squeeze().numpy()

        # Visualize input, ground truth, prediction, and overlay
        plt.figure(figsize=(20, 5))

        plt.subplot(1, 4, 1)
        plt.title("Input Image")
        plt.imshow(input_image, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 4, 2)
        plt.title("Ground Truth Mask")
        plt.imshow(gt_mask, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 4, 3)
        plt.title("Predicted Mask")
        plt.imshow(binary_mask, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 4, 4)
        plt.title(f"Processed Overlay (Teeth: {teeth_count})")
        plt.imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
        plt.axis("off")

        plt.tight_layout()
        plt.show()

        # Stop after visualizing 5 samples
        if idx == 4:
            break

if __name__ == "__main__":
    test_pipeline()
