import torch
import torch.optim as optim
from tqdm import tqdm
from src.evaluation.metrics import dice_loss, dice_coefficient
import os


def train_model(model, train_loader, val_loader, device, num_epochs=100, learning_rate=1e-4, patience=10):
    """
    Train a UNet model for semantic segmentation with early stopping and model checkpointing.

    Args:
        model (torch.nn.Module): The UNet model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training set.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation set.
        device (torch.device): Device to train the model on (e.g., 'cuda' or 'cpu').
        num_epochs (int, optional): Maximum number of training epochs. Defaults to 100.
        learning_rate (float, optional): Learning rate for the optimizer. Defaults to 1e-4.
        patience (int, optional): Number of epochs to wait for validation loss improvement before early stopping. Defaults to 10.

    Returns:
        tuple:
            - model (torch.nn.Module): The trained model.
            - history (tuple): Training history containing lists of train/val losses and accuracies.

    Outputs:
        - Saves the best and last model weights to specified file paths.
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = dice_loss

    train_loss, val_loss, train_accuracy, val_accuracy = [], [], [], []
    best_val_loss = float('inf')
    best_model_path = "/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/outputs/models/unet_best.pth"
    last_model_path = "/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/outputs/models/unet_last.pth"

    # Early stopping variables
    patience_counter = 0

    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_accuracy = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)

        for images, masks in loop:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(masks, outputs)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            accuracy = dice_coefficient(masks, outputs).item()
            epoch_accuracy += accuracy

            # Update tqdm bar with loss and accuracy
            loop.set_postfix(loss=loss.item(), accuracy=accuracy)

        train_loss.append(epoch_loss / len(train_loader))
        train_accuracy.append(epoch_accuracy / len(train_loader))

        # Validation
        model.eval()
        val_epoch_loss = 0
        val_epoch_accuracy = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(masks, outputs)
                val_epoch_loss += loss.item()
                val_epoch_accuracy += dice_coefficient(masks, outputs).item()

        val_loss.append(val_epoch_loss / len(val_loader))
        val_accuracy.append(val_epoch_accuracy / len(val_loader))

        print(f"Epoch {epoch + 1}/{num_epochs}, "
              f"Train Loss: {train_loss[-1]:.4f}, Train Accuracy: {train_accuracy[-1]:.4f}, "
              f"Val Loss: {val_loss[-1]:.4f}, Val Accuracy: {val_accuracy[-1]:.4f}")

        # Save the best model
        if val_loss[-1] < best_val_loss:
            best_val_loss = val_loss[-1]
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved at epoch {epoch + 1} with Val Loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement in Val Loss for {patience_counter} epoch(s).")

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement.")
            break

    # Save the last model
    torch.save(model.state_dict(), last_model_path)
    print(f"Last model saved at epoch {epoch + 1}")

    return model, (train_loss, val_loss, train_accuracy, val_accuracy)
