import os
import numpy as np
from sklearn.model_selection import train_test_split
import cv2


def load_images_and_masks(image_dir, mask_dir, img_size=(512, 512)):
    """
    Load images and corresponding masks from directories, resize them, and normalize their pixel values.

    Args:
        image_dir (str): Path to the directory containing the input images.
        mask_dir (str): Path to the directory containing the corresponding masks.
        img_size (tuple): Target size for resizing the images and masks (height, width).

    Returns:
        tuple: Numpy arrays of images and masks with shape (num_samples, height, width, 1).
    """
    images = []
    masks = []

    for img_name in sorted(os.listdir(image_dir)):
        img_path = os.path.join(image_dir, img_name)
        mask_path = os.path.join(mask_dir, img_name)

        # Load the image and mask in grayscale mode
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Skip if any file is missing or could not be loaded
        if img is None or mask is None:
            print(f"Warning: Could not load {img_path} or {mask_path}. Skipping...")
            continue

        # Resize and normalize pixel values to [0, 1]
        img = cv2.resize(img, img_size) / 255.0
        mask = cv2.resize(mask, img_size) / 255.0

        # Expand dimensions to add a channel dimension
        images.append(np.expand_dims(img, axis=-1))
        masks.append(np.expand_dims(mask, axis=-1))

    return np.array(images), np.array(masks)


def prepare_data(image_dir, mask_dir, img_size=(512, 512), test_size=0.2, val_size=0.1, random_state=42):
    """
    Load and prepare the dataset by splitting it into training, validation, and test sets.

    Args:
        image_dir (str): Path to the directory containing the input images.
        mask_dir (str): Path to the directory containing the corresponding masks.
        img_size (tuple): Target size for resizing the images and masks (height, width).
        test_size (float): Proportion of the dataset to be used as the test set.
        val_size (float): Proportion of the training set to be used as the validation set.
        random_state (int): Seed for reproducibility of splits.

    Returns:
        tuple: Numpy arrays for training (X_train, y_train), validation (X_val, y_val), and test (X_test, y_test).
    """
    # Load images and masks
    images, masks = load_images_and_masks(image_dir, mask_dir, img_size)

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size=test_size, random_state=random_state)

    # Further split training data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=random_state)

    return X_train, X_val, y_train, y_val, X_test, y_test