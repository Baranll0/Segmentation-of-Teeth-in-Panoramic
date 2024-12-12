import os
import numpy as np
from PIL import Image
from natsort import natsorted
import torch


def convert_one_channel(img):
    """
    Convert an image to a single channel if it has multiple channels.

    Args:
        img (numpy array): Input image.

    Returns:
        numpy array: Single-channel image.
    """
    if len(img.shape) > 2:
        return img[:, :, 0]
    return img


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

    # Sort files in natural order
    image_files = natsorted(os.listdir(image_dir))
    mask_files = natsorted(os.listdir(mask_dir))

    for img_name, mask_name in zip(image_files, mask_files):
        img_path = os.path.join(image_dir, img_name)
        mask_path = os.path.join(mask_dir, mask_name)

        # Load image and mask using PIL
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        mask = Image.open(mask_path).convert('L')

        # Resize and normalize
        img = img.resize(img_size, Image.Resampling.LANCZOS)
        mask = mask.resize(img_size, Image.Resampling.LANCZOS)

        img = np.asarray(img) / 255.0  # Normalize to range [0, 1]
        mask = np.asarray(mask) / 255.0

        img = convert_one_channel(img)
        mask = convert_one_channel(mask)

        # Convert to PyTorch tensors and add channel dimension
        images.append(torch.tensor(img, dtype=torch.float32).unsqueeze(0))
        masks.append(torch.tensor(mask, dtype=torch.float32).unsqueeze(0))

    return torch.stack(images), torch.stack(masks)