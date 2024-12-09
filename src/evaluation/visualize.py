import matplotlib.pyplot as plt
import numpy as np

def plot_loss_curve(train_losses, val_losses, filename="loss_curve.png"):
    """
    Eğitim ve doğrulama kayıp eğrilerini çizer ve kaydeder.
    """
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.savefig(filename)
    plt.close()

def visualize_metrics(metrics, filename="metrics.png"):
    """
    Değerlendirme metriklerini çizer ve kaydeder.
    """
    labels = list(metrics.keys())
    values = list(metrics.values())

    plt.figure(figsize=(8, 4))
    plt.bar(labels, values, color='skyblue')
    plt.title('Evaluation Metrics')
    plt.savefig(filename)
    plt.close()

def visualize_predictions(images, ground_truth, predictions, idx=0, save_path=None):
    """
    Model tahminlerini görselleştirir ve kaydeder.
    """
    image = images[idx].cpu().numpy().transpose(1, 2, 0)
    gt = ground_truth[idx].cpu().numpy().squeeze()
    pred = predictions[idx].cpu().numpy().squeeze()

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(image, cmap='gray')
    axs[0].set_title("Input Image")
    axs[1].imshow(gt, cmap='gray')
    axs[1].set_title("Ground Truth")
    axs[2].imshow(pred, cmap='gray')
    axs[2].set_title("Prediction")
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()