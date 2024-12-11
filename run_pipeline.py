import os
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from src.models.unet import UNet
from src.training.train import train_model
from src.evaluation.metrics import dice_coefficient
from src.preprocessing.dataset import load_images_and_masks
from src.preprocessing.augment import create_augmentation_pipeline, augment_images
from src.evaluation.visualize import plot_loss, plot_dice_coefficient, save_visualizations
from src.inference.predict import predict_and_visualize
import torch

def main_pipeline():
    """
    Main pipeline for training a UNet model for image segmentation, saving visualizations,
    and evaluating predictions.

    Steps:
        1. Load image and mask data.
        2. Split the data into training and validation sets.
        3. Apply data augmentation to expand the training set.
        4. Prepare DataLoaders for training and validation.
        5. Train the UNet model using the training data.
        6. Save training visualizations (e.g., loss and dice coefficient plots).
        7. Visualize predictions on validation data.

    Paths:
        image_dir (str): Path to the directory containing input images.
        mask_dir (str): Path to the directory containing segmentation masks.
        visualization_dir (str): Directory to save output visualizations.

    Outputs:
        - Trained model weights.
        - Loss and dice coefficient plots.
        - Visualization of predicted masks overlaid on input images.

    Returns:
        None
    """
    # Paths
    image_dir = "/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/dataset/DentalPanoramicXrays/images"
    mask_dir = "/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/dataset/DentalPanoramicXrays/masks"
    visualization_dir = "/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/outputs/visualizations"

    os.makedirs(visualization_dir, exist_ok=True)

    # Load data
    print("Loading images and masks...")
    images, masks = load_images_and_masks(image_dir, mask_dir)

    # Train-test split
    print("Splitting dataset...")
    X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)

    # Augmentation
    print("Applying data augmentation...")
    augmentation_pipeline = create_augmentation_pipeline()
    X_train_aug, y_train_aug = augment_images(X_train, y_train, augmentation_pipeline)

    # Combine augmented data
    X_train_combined = np.concatenate((X_train, X_train_aug))
    y_train_combined = np.concatenate((y_train, y_train_aug))

    # Dataloader
    train_dataset = TensorDataset(torch.tensor(X_train_combined).float(), torch.tensor(y_train_combined).float())
    val_dataset = TensorDataset(torch.tensor(X_val).float(), torch.tensor(y_val).float())

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

    # Model
    model = UNet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train model
    print("Training the model...")
    model, history = train_model(model, train_loader, val_loader, device)

    # Plot and save history
    print("Saving training visualizations...")
    train_loss, val_loss, train_dice, val_dice = history
    plot_loss(train_loss, val_loss, save_path=os.path.join(visualization_dir, "loss_plot.png"))
    plot_dice_coefficient(train_dice, val_dice, save_path=os.path.join(visualization_dir, "dice_coefficient_plot.png"))

    # Visualize predictions
    print("Visualizing predictions...")
    for i in range(3):  # Visualize 3 random predictions
        idx = np.random.randint(0, len(val_dataset))
        save_path = os.path.join(visualization_dir, f"prediction_{i+1}.png")
        predict_and_visualize(model, val_dataset, idx, device, save_path)

if __name__ == "__main__":
    main_pipeline()
