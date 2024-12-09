import os
import torch
from torch.optim import Adam
from torchvision import transforms
from tqdm import tqdm
from src.preprocessing.dataset import get_data_loader
from src.preprocessing.preprocess import preprocess_image
from src.models.unet import UNet
from src.models.utils import dice_loss
from src.evaluation.metrics import evaluate_model
from src.evaluation.visualize import plot_loss_curve, visualize_metrics, visualize_predictions
from src.inference.predict import predict_segmentation
from src.inference.postprocess import postprocess_segmentation
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(train_loader, val_loader, model, criterion, optimizer, num_epochs=50, patience=5, save_dir="/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/outputs"):
    """
    Modeli eğitir ve çıktıları kaydeder.
    """
    models_dir = os.path.join(save_dir, "models")
    visualizations_dir = os.path.join(save_dir, "visualizations")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(visualizations_dir, exist_ok=True)

    best_loss = float("inf")
    train_losses = []
    val_losses = []
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # Eğitim aşaması
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Training]")
        for images, (segmentation1, _) in train_bar:
            images, segmentation1 = images.to(device), segmentation1.to(device)
            optimizer.zero_grad()
            outputs = model(images)

            # Loss hesaplama
            loss = criterion(outputs, segmentation1)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Accuracy hesaplama
            preds = (outputs > 0.5).long()
            correct_train += (preds == segmentation1).sum().item()
            total_train += segmentation1.numel()

            # TQDM güncellemesi
            train_bar.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        train_accuracy = correct_train / total_train

        # Doğrulama aşaması
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        val_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Validation]")
        with torch.no_grad():
            for images, (segmentation1, _) in val_bar:
                images, segmentation1 = images.to(device), segmentation1.to(device)
                outputs = model(images)

                # Loss hesaplama
                loss = criterion(outputs, segmentation1)
                val_loss += loss.item()

                # Accuracy hesaplama
                preds = (outputs > 0.5).long()
                correct_val += (preds == segmentation1).sum().item()
                total_val += segmentation1.numel()

                # TQDM güncellemesi
                val_bar.set_postfix(loss=loss.item())

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        val_accuracy = correct_val / total_val

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        # En iyi modeli kaydet
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0  # Patience sıfırla
            torch.save(model.state_dict(), os.path.join(models_dir, "best_model.pth"))
            print(f"Best model saved at epoch {epoch + 1}")
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience counter: {patience_counter}")

        # Son modeli kaydet
        torch.save(model.state_dict(), os.path.join(models_dir, "last_model.pth"))

        # Erken durdurma kontrolü
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # Eğitim ve doğrulama kayıp eğrilerini çiz ve kaydet
    plot_loss_curve(
        train_losses,
        val_losses,
        os.path.join(visualizations_dir, "loss_curve.png")
    )


def evaluate_and_visualize(model, test_loader, save_dir):
    """
    Modeli değerlendirir ve sonuçları görselleştirir.
    """
    visualizations_dir = os.path.join(save_dir, "visualizations")
    os.makedirs(visualizations_dir, exist_ok=True)

    # Değerlendirme metriklerini hesapla
    print("Evaluating model...")
    metrics = evaluate_model(model, test_loader, device=device)
    print("Evaluation Metrics:", metrics)

    # Değerlendirme metriklerini görselleştir
    visualize_metrics(
        metrics,
        filename=os.path.join(visualizations_dir, "metrics.png")
    )

    # Test tahminlerini görselleştir
    for idx, (images, (segmentation1, _)) in enumerate(test_loader):
        images, segmentation1 = images.to(device), segmentation1.to(device)
        predictions = model(images)
        predictions = (predictions > 0.5).long()

        visualize_predictions(
            images,
            segmentation1,
            predictions,
            idx=0,
            save_path=os.path.join(visualizations_dir, f"predictions_{idx}.png")
        )
        if idx == 5:  # Sadece 5 örnek kaydet
            break


def inference(model, test_dir, save_dir):
    """
    Test setinden 5 görüntü üzerinde tahmin yapar ve sonuçları kaydeder.
    """
    output_dir = os.path.join(save_dir, "inference")
    os.makedirs(output_dir, exist_ok=True)

    test_images = sorted(os.listdir(test_dir))
    selected_images = test_images[:5]  # İlk 5 görüntüyü seç

    for idx, image_name in enumerate(selected_images):
        image_path = os.path.join(test_dir, image_name)

        # Predict
        prediction = predict_segmentation(model, image_path, device=device)

        # Postprocess
        processed_prediction = postprocess_segmentation(prediction)

        # Save the processed prediction
        output_path = os.path.join(output_dir, f"processed_{image_name}")
        plt.imsave(output_path, processed_prediction[0], cmap='gray')
        print(f"Processed prediction saved to {output_path}")

        # Visualize input and prediction
        image_tensor = preprocess_image(image_path).to(device)
        image = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

        visualize_predictions(
            torch.tensor(image).unsqueeze(0),
            None,
            torch.tensor(processed_prediction).unsqueeze(0),
            idx=0,
            save_path=os.path.join(output_dir, f"visualized_prediction_{idx}.png")
        )
        print(f"Visualized prediction saved to {output_path}")


def main():
    # Dataset directories
    dataset_dirs = {
        "train": "/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/dataset/train",
        "val": "/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/dataset/val",
        "test": "/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/dataset/test",
    }

    # Transformations
    image_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Data loaders
    train_loader = get_data_loader(
        os.path.join(dataset_dirs["train"], "images"),
        os.path.join(dataset_dirs["train"], "segmentation1"),
        os.path.join(dataset_dirs["train"], "segmentation2"),
        batch_size=16,
        image_transform=image_transform,
        mask_transform=mask_transform,
    )
    val_loader = get_data_loader(
        os.path.join(dataset_dirs["val"], "images"),
        os.path.join(dataset_dirs["val"], "segmentation1"),
        os.path.join(dataset_dirs["val"], "segmentation2"),
        batch_size=16,
        image_transform=image_transform,
        mask_transform=mask_transform,
    )
    test_loader = get_data_loader(
        os.path.join(dataset_dirs["test"], "images"),
        os.path.join(dataset_dirs["test"], "segmentation1"),
        os.path.join(dataset_dirs["test"], "segmentation2"),
        batch_size=16,
        image_transform=image_transform,
        mask_transform=mask_transform,
    )

    # Model, loss, optimizer
    model = UNet(in_channels=3, out_channels=1).to(device)
    criterion = dice_loss
    optimizer = Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(train_loader, val_loader, model, criterion, optimizer, num_epochs=50, patience=5, save_dir="/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/outputs")

    # Evaluate the model
    evaluate_and_visualize(model, test_loader, save_dir="/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/outputs")

    # Inference
    inference(model, os.path.join(dataset_dirs["test"], "images"), save_dir="/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/outputs")


if __name__ == "__main__":
    main()
