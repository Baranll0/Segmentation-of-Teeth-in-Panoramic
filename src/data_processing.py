import os
import shutil
from sklearn.model_selection import train_test_split

def organize_dataset(image_dir, mask_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Veriyi train, val ve test olarak organize eder ve belirtilen dizine kopyalar.
    """
    # Oranların toplamını kontrol et
    assert train_ratio + val_ratio + test_ratio == 1.0, "Train, val ve test oranlarının toplamı 1.0 olmalı!"

    os.makedirs(output_dir, exist_ok=True)
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    test_dir = os.path.join(output_dir, "test")

    for subdir in ["images", "masks"]:
        os.makedirs(os.path.join(train_dir, subdir), exist_ok=True)
        os.makedirs(os.path.join(val_dir, subdir), exist_ok=True)
        os.makedirs(os.path.join(test_dir, subdir), exist_ok=True)

    images = sorted(os.listdir(image_dir))
    masks = sorted(os.listdir(mask_dir))

    # Verilerin bölünmesi
    train_images, temp_images, train_masks, temp_masks = train_test_split(images, masks, test_size=1 - train_ratio, random_state=42)
    val_images, test_images, val_masks, test_masks = train_test_split(temp_images, temp_masks, test_size=test_ratio / (test_ratio + val_ratio), random_state=42)

    # Verilerin kopyalanması
    for split, (split_images, split_masks) in zip(
        ["train", "val", "test"], [(train_images, train_masks), (val_images, val_masks), (test_images, test_masks)]
    ):
        for img, mask in zip(split_images, split_masks):
            shutil.copy(os.path.join(image_dir, img), os.path.join(output_dir, split, "images", img))
            shutil.copy(os.path.join(mask_dir, mask), os.path.join(output_dir, split, "masks", mask))

    print(f"Veri organize edildi ve {output_dir} dizinine kopyalandı.")


# Ana fonksiyon
if __name__ == "__main__":
    image_dir = "/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/data/processed/img"
    mask_dir = "/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/data/processed/mask"
    output_dir = "/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/data/split_data"

    organize_dataset(image_dir, mask_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
