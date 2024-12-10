import albumentations as A
import numpy as np


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
        images (numpy.ndarray): Array of input images.
        masks (numpy.ndarray): Array of corresponding masks.
        augmentation_pipeline (albumentations.Compose): Augmentation pipeline to apply.

    Returns:
        tuple: Augmented images and masks as numpy arrays.
    """
    augmented_images, augmented_masks = [], []
    for img, mask in zip(images, masks):
        # Convert images and masks to uint8 for augmentation
        img = (img * 255).astype(np.uint8)
        mask = (mask * 255).astype(np.uint8)

        augmented = augmentation_pipeline(image=img, mask=mask)

        # Normalize and append augmented data
        augmented_images.append(augmented['image'] / 255.0)
        augmented_masks.append(augmented['mask'] / 255.0)

    return np.array(augmented_images), np.array(augmented_masks)