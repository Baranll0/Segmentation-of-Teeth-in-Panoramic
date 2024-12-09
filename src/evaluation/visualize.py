import matplotlib.pyplot as plt
import numpy as np


def visualize_predictions(images, ground_truth, predictions, idx=0):
    """
    Görsellerin ve tahminlerin karşılaştırılması.
    Args:
        images (torch.Tensor): Girdi görüntüleri.
        ground_truth (torch.Tensor): Gerçek segmentasyon maskeleri.
        predictions (torch.Tensor): Model tahminleri.
        idx (int): Görüntülerin sırası.
    """
    image = images[idx].cpu().numpy().transpose(1, 2, 0)  # Convert to HWC
    gt = ground_truth[idx].cpu().numpy()
    pred = predictions[idx].cpu().numpy()

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(image)
    axs[0].set_title("Input Image")
    axs[0].axis('off')

    axs[1].imshow(gt, cmap='gray')
    axs[1].set_title("Ground Truth")
    axs[1].axis('off')

    axs[2].imshow(pred, cmap='gray')
    axs[2].set_title("Prediction")
    axs[2].axis('off')

    plt.show()


def plot_loss_curve(losses, filename="loss_curve.png"):
    """
    Eğitim kaybını (loss) görselleştirmek için eğri çizimi.
    Args:
        losses (list): Her epoch'ta kaybedilen değerler.
        filename (str): Kaydedilecek dosya adı.
    """
    plt.figure()
    plt.plot(range(1, len(losses) + 1), losses, label='Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.savefig(filename)
    plt.close()