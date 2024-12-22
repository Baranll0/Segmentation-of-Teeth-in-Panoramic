import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, ToTensor, Normalize
from scipy.spatial import distance as dist
from imutils import perspective
from PIL import Image
from src.model import NestedUNet
import matplotlib.colors as mcolors

def midpoint(ptA, ptB):
    """Calculate midpoint between two points."""
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def load_model(model_path, num_classes, device):
    """Load the trained model."""
    model = NestedUNet(input_channels=3, output_channels=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def preprocess_image(image_path):
    """Preprocess the input image."""
    transform = Compose([ToTensor(), Normalize(mean=[0.5], std=[0.5])])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # Add batch dimension


def postprocess_mask(pred_mask):
    """Ensure mask is clean and visualizable."""
    # Ensure mask is uint8 and non-zero
    pred_mask = pred_mask.astype(np.uint8)
    return pred_mask


def CCA_Analysis(orig_image, cleaned_mask):
    """
    Perform CCA on the cleaned mask and draw rotated bounding boxes.
    """
    teeth_count = 0
    image_with_boxes = orig_image.copy()
    unique_classes = np.unique(cleaned_mask)

    for class_id in unique_classes:
        if class_id == 0:  # Skip background
            continue

        # Isolate single class
        single_class_mask = (cleaned_mask == class_id).astype(np.uint8) * 255
        contours, _ = cv2.findContours(single_class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) < 50:  # Filter very small areas
                continue

            # Get rotated bounding box
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(perspective.order_points(box))
            cv2.drawContours(image_with_boxes, [box], -1, (0, 255, 0), 1)

            # Add label
            teeth_count += 1
            (x, y), _ = cv2.minEnclosingCircle(contour)
            cv2.putText(image_with_boxes, f"",
                        (int(x), int(y) - 10),  # Y koordinatını hafif yukarı kaydır
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    return image_with_boxes, teeth_count



def visualize_results(image_path, true_mask_path, pred_mask, image_with_boxes):
    """Visualize the original image, true mask, predicted mask, and image with bounding boxes."""
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    true_mask = np.array(Image.open(true_mask_path))

    plt.figure(figsize=(16, 6))

    plt.subplot(1, 4, 1)
    plt.title("Original Image")
    plt.imshow(image)
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.title("True Mask")
    plt.imshow(true_mask, cmap="nipy_spectral")  # True mask colorized
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.title("Predicted Mask")
    cmap = mcolors.ListedColormap(plt.cm.nipy_spectral(np.linspace(0, 1, 33)))
    plt.imshow(pred_mask, cmap=cmap, interpolation="nearest")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.title("Bounding Boxes")
    plt.imshow(image_with_boxes)
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def predict(image_path, true_mask_path, model_path, num_classes, device):
    """Predict the mask and draw refined bounding boxes."""
    model = load_model(model_path, num_classes, device)
    input_image = preprocess_image(image_path).to(device)

    # Predict mask
    with torch.no_grad():
        output = model(input_image)
        print("Model output shape:", output.shape)
        pred_mask = output.argmax(1).cpu().numpy()[0]

        # Debug: check if the mask is empty
        if np.all(pred_mask == 0):
            print("Warning: Predicted mask is entirely zero!")

    # Post-process mask and analyze
    cleaned_mask = postprocess_mask(pred_mask)
    orig_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    image_with_boxes, teeth_count = CCA_Analysis(orig_image, cleaned_mask)

    print(f"Total Detected Teeth: {teeth_count-1}")

    # Visualize results
    visualize_results(image_path, true_mask_path, cleaned_mask, image_with_boxes)



if __name__ == "__main__":
    # Paths
    image_path = "/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/data/processed/resized_images/65.jpg"
    true_mask_path = "/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/data/processed/resized_masks/65.png"
    model_path = "/nested-unet/best3.pth"
    num_classes = 33

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Run prediction
    predict(image_path, true_mask_path, model_path, num_classes, device)
