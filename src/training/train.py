import torch
from torch.optim import Adam
from torchvision import transforms
from src.models.unet import UNet  # UNet modelinizi içe aktarın
from src.models.utils import dice_loss  # Özel dice loss fonksiyonunuz
from src.preprocessing.dataset import get_data_loader  # Veri yükleyicisini içe aktarın
from src.preprocessing.augment import augment_image  # Augmentasyon fonksiyonunu içe aktarın
from src.evaluation.metrics import evaluate_model  # Modeli değerlendirme fonksiyonunuz

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(train_loader, model, criterion, optimizer, num_epochs=10, save_dir='./outputs/models'):
    model.train()
    best_loss = float('inf')  # En iyi modeli takip etmek için
    for epoch in range(num_epochs):
        running_loss = 0.0
        print(f"Epoch {epoch + 1} of {num_epochs}")
        for images, (segmentation1, segmentation2) in train_loader:
            images, segmentation1, segmentation2 = images.to(device), segmentation1.to(device), segmentation2.to(device)

            optimizer.zero_grad()

            # Veri artırma işlemi (augmentation)
            augmented_images = torch.stack([augment_image(img) for img in images])  # Her bir görüntüyü artır

            # Model tahmini
            outputs = model(augmented_images)

            # Kaybı hesaplama (örneğin dice loss veya başka bir kayıp fonksiyonu)
            loss = criterion(outputs, segmentation1)  # segmentation1 maskesi ile karşılaştır
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss}")

        # Eğer model en iyi kaybı verdi ise kaydet
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), f"{save_dir}/best_model.pth")
            print(f"Saved best model at epoch {epoch + 1}")

        # Her epoch'ta son model kaydedilir
        torch.save(model.state_dict(), f"{save_dir}/last_model.pth")
        print(f"Saved last model at epoch {epoch + 1}")

def main():
    # Resim ve segmentasyon veri yolları
    images_dir = "/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/dataset/DentalPanoramicXrays/images"
    segmentation1_dir = "/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/dataset/DentalPanoramicXrays/segmentation1"
    segmentation2_dir = "/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/dataset/DentalPanoramicXrays/segmentation2"

    # Veri yükleyici oluşturma ve augmentasyon işlemleri
    image_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Görüntüleri normalize et
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),  # Maskeleri tensor'a çevir (normalizasyon gereksiz)
    ])

    train_loader = get_data_loader(images_dir, segmentation1_dir, segmentation2_dir, batch_size=16,
                                   image_transform=image_transform, mask_transform=mask_transform)

    # Model ve optimizer ayarları
    model = UNet(in_channels=3, out_channels=1).to(device)  # RGB giriş (3 kanal), 1 kanal çıkış (örneğin ikili segmentasyon)
    criterion = dice_loss  # Özel kayıp fonksiyonunuz (Dice Loss)
    optimizer = Adam(model.parameters(), lr=0.001)

    # Eğitim işlemini başlat
    train_model(train_loader, model, criterion, optimizer, num_epochs=10, save_dir='/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/outputs/models/')

    # Eğitimden sonra modeli test veri setinde değerlendirin
    print("Training complete. Evaluating model...")
    model.eval()  # Modeli değerlendirme moduna al
    test_loader = get_data_loader(images_dir, segmentation1_dir, segmentation2_dir, batch_size=16,
                                  image_transform=image_transform, mask_transform=mask_transform)

    # Modeli değerlendirin
    evaluate_model(model, test_loader)  # Değerlendirme fonksiyonunuz

if __name__ == "__main__":
    main()
