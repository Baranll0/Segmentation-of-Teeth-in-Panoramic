import os
import matplotlib.pyplot as plt
import numpy as np
import torch


def dice_coefficient(pred, target, num_classes):
    """
    Dice skoru hesaplar.
    """
    dice_scores = []
    for class_idx in range(num_classes):
        pred_class = (pred == class_idx).float()
        target_class = (target == class_idx).float()

        intersection = (pred_class * target_class).sum()
        union = pred_class.sum() + target_class.sum()

        dice = (2.0 * intersection) / (union + 1e-5)
        dice_scores.append(dice.item())

    mean_dice = np.mean(dice_scores)
    return mean_dice, dice_scores


def precision_recall_f1(pred, target, num_classes):
    """
    Precision, Recall ve F1 skorlarını hesaplar.
    """
    precisions = []
    recalls = []
    f1_scores = []

    for class_idx in range(num_classes):
        pred_class = (pred == class_idx).float()
        target_class = (target == class_idx).float()

        tp = (pred_class * target_class).sum()
        fp = (pred_class * (1 - target_class)).sum()
        fn = ((1 - pred_class) * target_class).sum()

        precision = tp / (tp + fp + 1e-5)
        recall = tp / (tp + fn + 1e-5)
        f1 = (2 * precision * recall) / (precision + recall + 1e-5)

        precisions.append(precision.item())
        recalls.append(recall.item())
        f1_scores.append(f1.item())

    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)
    mean_f1 = np.mean(f1_scores)

    return mean_precision, mean_recall, mean_f1, precisions, recalls, f1_scores


def calculate_metrics(pred, target, num_classes):
    """
    Accuracy, Dice, Precision, Recall ve F1 skorlarını hesaplar.
    """
    pred = torch.argmax(pred, dim=1)

    accuracy = (pred == target).float().mean().item()
    mean_dice, dice_scores = dice_coefficient(pred, target, num_classes)
    mean_precision, mean_recall, mean_f1, precisions, recalls, f1_scores = precision_recall_f1(pred, target,
                                                                                               num_classes)

    return {
        "accuracy": accuracy,
        "mean_dice": mean_dice,
        "dice_scores": dice_scores,
        "mean_precision": mean_precision,
        "mean_recall": mean_recall,
        "mean_f1": mean_f1,
        "precisions": precisions,
        "recalls": recalls,
        "f1_scores": f1_scores
    }


def log_metrics(epoch, metrics, log_file):
    """
    Metrikleri bir log dosyasına kaydeder.
    """
    with open(log_file, "a") as f:
        f.write(f"Epoch {epoch + 1}: {metrics}\n")


def save_metrics_plot(history, output_dir):
    """
    Eğitim sürecindeki metrikleri bir grafik olarak kaydeder.
    """
    epochs = range(1, len(history['loss']) + 1)

    plt.figure(figsize=(12, 12))

    plt.subplot(2, 2, 1)
    plt.plot(epochs, history['loss'], label='Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(epochs, history['dice'], label='Dice')
    plt.plot(epochs, history['val_dice'], label='Val Dice')
    plt.title('Dice Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Dice Coefficient')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(epochs, history['precision'], label='Precision')
    plt.plot(epochs, history['recall'], label='Recall')
    plt.title('Precision and Recall Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(epochs, history['f1'], label='F1 Score')
    plt.title('F1 Score Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics_plot.png"))
    plt.close()


def save_sample_visualizations(model, data_loader, device, output_dir):
    """
    Eğitim sonrası örnek görselleri kaydeder.
    """
    model.eval()
    with torch.no_grad():
        for idx, (images, masks) in enumerate(data_loader):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            predicted = torch.argmax(outputs, dim=1).cpu().numpy()

            # İlk batch'ten bir örnek kaydet
            for i in range(min(len(images), 5)):
                image = images[i].cpu().numpy().transpose(1, 2, 0)
                true_mask = masks[i].cpu().numpy()
                pred_mask = predicted[i]

                plt.figure(figsize=(12, 4))
                plt.subplot(1, 3, 1)
                plt.title("Input Image")
                plt.imshow((image - image.min()) / (image.max() - image.min()))
                plt.axis("off")

                plt.subplot(1, 3, 2)
                plt.title("True Mask")
                plt.imshow(true_mask, cmap="nipy_spectral")
                plt.axis("off")

                plt.subplot(1, 3, 3)
                plt.title("Predicted Mask")
                plt.imshow(pred_mask, cmap="nipy_spectral")
                plt.axis("off")

                sample_path = os.path.join(output_dir, f"sample_{idx}_{i}.png")
                plt.savefig(sample_path)
                plt.close()

            if idx >= 1:
                break
