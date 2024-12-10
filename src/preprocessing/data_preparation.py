import os
import numpy as np
from sklearn.model_selection import train_test_split
import cv2


def load_images_and_masks(image_dir, mask_dir, img_size=(512, 512)):
    images = []
    masks = []
    for img_name in sorted(os.listdir(image_dir)):
        img_path = os.path.join(image_dir, img_name)
        mask_path = os.path.join(mask_dir, img_name)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            print(f"Warning: Could not load {img_path} or {mask_path}. Skipping...")
            continue

        img = cv2.resize(img, img_size) / 255.0
        mask = cv2.resize(mask, img_size) / 255.0

        images.append(np.expand_dims(img, axis=-1))
        masks.append(np.expand_dims(mask, axis=-1))

    return np.array(images), np.array(masks)


def prepare_data(image_dir, mask_dir, img_size=(512, 512), test_size=0.2, val_size=0.1, random_state=42):
    # Load images and masks
    images, masks = load_images_and_masks(image_dir, mask_dir, img_size)

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size=test_size, random_state=random_state)

    # Further split train into train and validation
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=random_state)

    return X_train, X_val, y_train, y_val, X_test, y_test