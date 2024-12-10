import matplotlib.pyplot as plt
import torch
import os

def plot_loss(train_loss, val_loss, save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_dice_coefficient(train_dice, val_dice, save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(train_dice, label="Train Dice Coefficient")
    plt.plot(val_dice, label="Validation Dice Coefficient")
    plt.xlabel("Epochs")
    plt.ylabel("Dice Coefficient")
    plt.legend()
    plt.title("Training and Validation Dice Coefficient")
    if save_path:
        plt.savefig(save_path)
    plt.show()

def save_visualizations(fig, save_path):
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)
    plt.show()

def predict_and_visualize(model, dataset, idx, device, save_path=None):
    model.eval()
    image, mask = dataset[idx]
    image = image.unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        pred_mask = model(image).squeeze().cpu().numpy()

    fig = plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("Input Image")
    plt.imshow(image.squeeze().cpu().numpy(), cmap="gray")
    plt.subplot(1, 3, 2)
    plt.title("True Mask")
    plt.imshow(mask.squeeze().numpy(), cmap="gray")
    plt.subplot(1, 3, 3)
    plt.title("Predicted Mask")
    plt.imshow(pred_mask, cmap="gray")

    if save_path:
        save_visualizations(fig, save_path)
    else:
        plt.show()
