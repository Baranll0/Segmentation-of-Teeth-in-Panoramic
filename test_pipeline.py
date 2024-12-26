import os
import torch
import matplotlib.pyplot as plt
from src.DeepLabV3 import DeepLabV3Plus
from src.dataset import MultiClassTeethDataset
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np

def load_model(checkpoints_dir, model, device):
    """
    Kaydedilen modeli yükler.
    """
    best_model_path = os.path.join(checkpoints_dir, "best.pth")
    state_dict = torch.load(best_model_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Model loaded from {best_model_path}")
    return model

def calculate_dice(pred, target, num_classes):
    """
    Dice skorunu hesaplar.
    """
    dice_scores = []
    for class_idx in range(num_classes):
        pred_class = (pred == class_idx).float()
        target_class = (target == class_idx).float()

        intersection = (pred_class * target_class).sum()
        union = pred_class.sum() + target_class.sum()

        dice = (2.0 * intersection) / (union + 1e-5)
        dice_scores.append(dice.item())

    return np.mean(dice_scores)

def calculate_metrics(pred, target, num_classes):
    """
    Precision, Recall ve F1 skorlarını hesaplar.
    """
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    precision = precision_score(target_flat.cpu(), pred_flat.cpu(), average="weighted")
    recall = recall_score(target_flat.cpu(), pred_flat.cpu(), average="weighted")
    f1 = f1_score(target_flat.cpu(), pred_flat.cpu(), average="weighted")
    return precision, recall, f1

def evaluate_model(model, data_loader, device, num_classes, output_dir):
    """
    Modelin performansını değerlendirir ve Dice, Precision, Recall, F1 skorlarını hesaplar.
    Ayrıca sonuçları grafik olarak kaydeder.
    """
    model.eval()
    total_dice = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    sample_count = 0

    dice_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    with torch.no_grad():
        for images, masks in data_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)

            dice = calculate_dice(predictions, masks, num_classes)
            precision, recall, f1 = calculate_metrics(predictions, masks, num_classes)

            total_dice += dice
            total_precision += precision
            total_recall += recall
            total_f1 += f1
            sample_count += 1

            dice_scores.append(dice)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)

    avg_dice = total_dice / sample_count
    avg_precision = total_precision / sample_count
    avg_recall = total_recall / sample_count
    avg_f1 = total_f1 / sample_count

    print(f"Average Dice Score: {avg_dice:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average F1 Score: {avg_f1:.4f}")

    # Grafik oluştur ve kaydet
    epochs = range(1, sample_count + 1)
    plt.figure(figsize=(10, 5))

    plt.plot(epochs, dice_scores, label="Dice Score")
    plt.plot(epochs, precision_scores, label="Precision")
    plt.plot(epochs, recall_scores, label="Recall")
    plt.plot(epochs, f1_scores, label="F1 Score")
    plt.xlabel("Samples")
    plt.ylabel("Score")
    plt.title("Dice, Precision, Recall, and F1 Scores per Sample")
    plt.legend()

    graph_path = os.path.join(output_dir, "metrics_graph.png")
    plt.savefig(graph_path)
    plt.close()

    print(f"Metrics graph saved to {graph_path}")

    return avg_dice, avg_precision, avg_recall, avg_f1

def visualize_predictions(model, data_loader, device, output_dir, num_samples=1):
    """
    Modelin tahminlerini görselleştirir ve kaydeder.
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for idx, (images, masks) in enumerate(data_loader):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            predicted = torch.argmax(outputs, dim=1).cpu().numpy()

            # İlk batch'ten num_samples kadar örnek kaydet
            for i in range(min(len(images), num_samples)):
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

            if idx >= 1:  # Sadece ilk batch'i işleyelim
                break

if __name__ == "__main__":
    # Yollar
    data_dir = "/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/data/split_data/test"
    train_image_dir = os.path.join(data_dir, "images")
    train_mask_dir = os.path.join(data_dir, "masks")
    checkpoints_dir = "/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/colab"
    output_dir = "/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/visualizations"

    # Hyperparameters
    batch_size = 4
    num_classes = 33  # Eğitim sırasında kullanılan sınıf sayısı
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset ve DataLoader
    dataset = MultiClassTeethDataset(train_image_dir, train_mask_dir)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Model
    model = DeepLabV3Plus(input_channels=3, num_classes=num_classes).to(device)

    # Load Model
    model = load_model(checkpoints_dir, model, device)

    # Evaluate Model
    evaluate_model(model, data_loader, device, num_classes, output_dir)

    # Visualize Predictions
    visualize_predictions(model, data_loader, device, output_dir)
