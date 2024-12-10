import os
import numpy as np
from sklearn.model_selection import train_test_split
from src.training.train import train_model
from src.evaluation.visualize import plot_history
from src.inference.predict import predict_and_visualize
from src.preprocessing.dataset import load_images_and_masks
from src.preprocessing.augment import create_augmentation_pipeline, augment_images


def main_pipeline():
    image_dir = '/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/dataset/DentalPanoramicXrays/images'
    mask_dir = '/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/dataset/DentalPanoramicXrays/masks'

    # Load dataset
    print("Loading images and masks...")
    images, masks = load_images_and_masks(image_dir, mask_dir)

    # Split into training and validation sets
    print("Splitting dataset into training and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)

    # Create augmentation pipeline and augment training data
    print("Applying data augmentation...")
    augmentation_pipeline = create_augmentation_pipeline()
    X_train_aug, y_train_aug = augment_images(X_train, y_train, augmentation_pipeline)

    # Combine original and augmented training data
    X_train_combined = np.concatenate((X_train, X_train_aug))
    y_train_combined = np.concatenate((y_train, y_train_aug))

    # Train the model
    print("Training the model...")
    model, history = train_model(X_train_combined, y_train_combined, X_val, y_val)

    # Plot training history
    print("Plotting training history...")
    plot_history(history)

    # Visualize predictions
    print("Visualizing predictions...")
    predict_and_visualize(model, X_val, y_val, idx=0)


if __name__ == "__main__":
    main_pipeline()