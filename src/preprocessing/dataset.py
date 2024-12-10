import os
import numpy as np
import cv2
import torch

def load_images_and_masks(image_dir, mask_dir, img_size=(512, 512)):
    """
    Load images and masks from the specified directories and convert them to PyTorch tensor format.

    Args:
        image_dir (str): Path to the directory containing the images.
        mask_dir (str): Path to the directory containing the masks.
        img_size (tuple): Desired size of the images and masks (height, width).

    Returns:
        tuple: PyTorch tensors of images and masks.
    """
    images, masks = [], []
    for img_name in sorted(os.listdir(image_dir)):
        img_path = os.path.join(image_dir, img_name)
        mask_path = os.path.join(mask_dir, img_name)

        # Load image and mask in grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Resize and normalize
        img = cv2.resize(img, img_size) / 255.0
        mask = cv2.resize(mask, img_size) / 255.0

        # Convert to PyTorch tensor and change format to (C, H, W)
        images.append(torch.tensor(img, dtype=torch.float32).unsqueeze(0))  # Add channel dimension
        masks.append(torch.tensor(mask, dtype=torch.float32).unsqueeze(0))  # Add channel dimension

    return torch.stack(images), torch.stack(masks)
