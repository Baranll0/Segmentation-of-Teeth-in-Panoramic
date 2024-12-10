import albumentations as A
import numpy as np
import torch

def create_augmentation_pipeline():
    """
    Create an Albumentations augmentation pipeline.

    Returns:
        albumentations.Compose: The augmentation pipeline.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.GaussNoise(p=0.2),
        A.ElasticTransform(p=0.2)
    ])

def augment_images(images, masks, augmentation_pipeline):
    """
    Apply augmentation to images and masks.

    Args:
        images (torch.Tensor): Tensor of input images with shape (N, C, H, W).
        masks (torch.Tensor): Tensor of corresponding masks with shape (N, C, H, W).
        augmentation_pipeline (albumentations.Compose): Augmentation pipeline to apply.

    Returns:
        tuple: Augmented images and masks as PyTorch tensors.
    """
    augmented_images, augmented_masks = [], []

    for img, mask in zip(images, masks):
        # Convert PyTorch tensors to NumPy arrays
        img_np = img.squeeze(0).numpy() * 255  # Convert to (H, W) and scale to [0, 255]
        mask_np = mask.squeeze(0).numpy() * 255

        # Apply augmentation
        augmented = augmentation_pipeline(image=img_np.astype(np.uint8), mask=mask_np.astype(np.uint8))

        # Convert augmented data back to PyTorch tensors
        augmented_img = torch.tensor(augmented['image'] / 255.0, dtype=torch.float32).unsqueeze(0)
        augmented_mask = torch.tensor(augmented['mask'] / 255.0, dtype=torch.float32).unsqueeze(0)

        augmented_images.append(augmented_img)
        augmented_masks.append(augmented_mask)

    # Stack all augmented data into tensors
    return torch.stack(augmented_images), torch.stack(augmented_masks)
