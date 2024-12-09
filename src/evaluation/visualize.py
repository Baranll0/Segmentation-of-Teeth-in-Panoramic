import matplotlib.pyplot as plt
import numpy as np

def plot_loss_curve(train_losses, val_losses, filename="loss_curve.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', marker='x')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def visualize_metrics(metrics, filename="metrics.png"):
    labels = list(metrics.keys())
    values = list(metrics.values())

    plt.figure(figsize=(8, 6))
    plt.bar(labels, values, color="skyblue")
    plt.ylabel("Score")
    plt.title("Evaluation Metrics")
    plt.ylim(0, 1)
    plt.savefig(filename)
    plt.close()

def visualize_predictions(images, ground_truth, predictions, idx=0, save_path=None):
    image = images[idx].cpu().numpy().transpose(1, 2, 0)
    gt = ground_truth[idx].cpu().numpy()
    pred = predictions[idx].cpu().numpy()

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    axs[0].imshow(np.clip(image, 0, 1))
    axs[0].set_title("Input Image")
    axs[0].axis('off')

    axs[1].imshow(gt, cmap='gray')
    axs[1].set_title("Ground Truth")
    axs[1].axis('off')

    axs[2].imshow(pred, cmap='gray')
    axs[2].set_title("Prediction")
    axs[2].axis('off')

    if save_path:
        plt.savefig(save_path)
    plt.tight_layout()
    plt.show()