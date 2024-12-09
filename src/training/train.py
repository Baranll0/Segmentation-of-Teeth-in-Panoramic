import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
from src.models.unet import UNet
from src.models.utils import dice_loss
from src.preprocessing.dataset import get_data_loader

# Cihaz seçimi
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(train_loader, val_loader, model, criterion, optimizer, num_epochs=2, patience=5, save_dir="./outputs"):
    """
    Modeli eğitir ve çıktıları kaydeder.
    """
    models_dir = os.path.join(save_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    best_loss = float("inf")
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # Eğitim döngüsü
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Training]")
        for images, masks in train_bar:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            preds = (outputs > 0.5).float()
            correct_train += (preds == masks).sum().item()
            total_train += masks.numel()

            train_bar.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        train_accuracy = correct_train / total_train

        # Doğrulama döngüsü
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        val_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Validation]")
        with torch.no_grad():
            for images, masks in val_bar:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)

                loss = criterion(outputs, masks)
                val_loss += loss.item()

                preds = (outputs > 0.5).float()
                correct_val += (preds == masks).sum().item()
                total_val += masks.numel()

                val_bar.set_postfix(loss=loss.item())

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        val_accuracy = correct_val / total_val

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        # En iyi modeli kaydet
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(models_dir, "best_model.pth"))
            print(f"Best model saved at epoch {epoch + 1}")
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience counter: {patience_counter}")

        # Erken durdurma kontrolü
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

        # Son modeli kaydet
        torch.save(model.state_dict(), os.path.join(models_dir, "last_model.pth"))


def evaluate_model(model, test_loader, device):
    """
    Modeli test seti üzerinde değerlendirir ve metrikleri döner.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = (outputs > 0.5).float()

            all_preds.append(preds.cpu().numpy().flatten())
            all_targets.append(masks.cpu().numpy().flatten())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    precision = precision_score(all_targets, all_preds, average='binary', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='binary', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='binary', zero_division=0)

    return {"precision": precision, "recall": recall, "f1": f1}


def inference(model, test_dir, save_dir):
    """
    Test seti üzerindeki tahminleri görselleştir ve kaydet.
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    test_images = sorted(os.listdir(test_dir))
    for idx, image_name in enumerate(test_images[:5]):  # İlk 5 görüntüyü işle
        image_path = os.path.join(test_dir, image_name)
        image = plt.imread(image_path)
        image_tensor = torch.tensor(image).unsqueeze(0).unsqueeze(0).float().to(device) / 255.0

        with torch.no_grad():
            prediction = model(image_tensor)
            prediction = (prediction > 0.5).float().squeeze().cpu().numpy()

        processed_path = os.path.join(save_dir, f"processed_{image_name}")
        plt.imsave(processed_path, prediction, cmap="gray")
        print(f"Processed prediction saved at {processed_path}")


def main():
    # Dataset paths
    dataset_dirs = {
        "train": "/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/dataset/train",
        "val": "/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/dataset/val",
        "test": "/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/dataset/test",
    }

    # Data loaders
    train_loader = get_data_loader(
        images_dir=os.path.join(dataset_dirs["train"], "images"),
        segmentation1_dir=os.path.join(dataset_dirs["train"], "segmentation1"),
        segmentation2_dir=os.path.join(dataset_dirs["train"], "segmentation2"),
        batch_size=16,
    )
    val_loader = get_data_loader(
        images_dir=os.path.join(dataset_dirs["val"], "images"),
        segmentation1_dir=os.path.join(dataset_dirs["val"], "segmentation1"),
        segmentation2_dir=os.path.join(dataset_dirs["val"], "segmentation2"),
        batch_size=16,
    )
    test_loader = get_data_loader(
        images_dir=os.path.join(dataset_dirs["test"], "images"),
        segmentation1_dir=os.path.join(dataset_dirs["test"], "segmentation1"),
        segmentation2_dir=os.path.join(dataset_dirs["test"], "segmentation2"),
        batch_size=16,
    )

    # Model, loss, optimizer
    model = UNet(input_channels=1, last_activation="sigmoid").to(device)
    criterion = dice_loss
    optimizer = Adam(model.parameters(), lr=0.001)

    # Training
    train_model(train_loader, val_loader, model, criterion, optimizer, num_epochs=50, patience=5, save_dir="./outputs")

    # Evaluation
    metrics = evaluate_model(model, test_loader, device)
    print("Test Metrics:", metrics)

    # Inference
    inference(model, os.path.join(dataset_dirs["test"], "images"), save_dir="./outputs/inference")


if __name__ == "__main__":
    main()