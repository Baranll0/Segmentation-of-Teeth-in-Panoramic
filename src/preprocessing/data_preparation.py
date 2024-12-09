import os
import random
from PIL import Image
from torchvision import transforms
from augment import Augmentations

def augment_and_save(image_path, seg1_path, seg2_path, output_dir, augmentations, prefix="aug"):
    """
    Görüntü ve segmentasyon maskelerine augmentasyon uygular ve kaydeder.
    Args:
        image_path (str): Görüntü yolu.
        seg1_path (str): Birinci segmentasyon maskesi yolu.
        seg2_path (str): İkinci segmentasyon maskesi yolu.
        output_dir (str): Kaydedilecek dizin.
        augmentations (callable): Augmentasyon işlemleri.
        prefix (str): Dosya ismi ön eki.
    """
    image = Image.open(image_path).convert("RGB")
    seg1 = Image.open(seg1_path).convert("L")
    seg2 = Image.open(seg2_path).convert("L")

    augmented_image = augmentations(image)
    augmented_seg1 = augmentations(seg1)
    augmented_seg2 = augmentations(seg2)

    # Kaydet
    image_name = f"{prefix}_{os.path.basename(image_path)}"
    seg1_name = f"{prefix}_{os.path.basename(seg1_path)}"
    seg2_name = f"{prefix}_{os.path.basename(seg2_path)}"

    augmented_image.save(os.path.join(output_dir, "images", image_name))
    augmented_seg1.save(os.path.join(output_dir, "segmentation1", seg1_name))
    augmented_seg2.save(os.path.join(output_dir, "segmentation2", seg2_name))


def prepare_augmented_data(input_dirs, output_dir, num_augments=5):
    """
    Augmentasyonu uygular ve augmented veri seti oluşturur.
    Args:
        input_dirs (dict): Görüntü ve maskelerin yollarını içeren sözlük.
        output_dir (str): Augmented veri setinin kaydedileceği dizin.
        num_augments (int): Her görüntü için oluşturulacak augmentasyon sayısı.
    """
    augmentations = Augmentations()
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "segmentation1"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "segmentation2"), exist_ok=True)

    image_paths = sorted(os.listdir(input_dirs["images"]))
    seg1_paths = sorted(os.listdir(input_dirs["segmentation1"]))
    seg2_paths = sorted(os.listdir(input_dirs["segmentation2"]))

    for img_file, seg1_file, seg2_file in zip(image_paths, seg1_paths, seg2_paths):
        img_path = os.path.join(input_dirs["images"], img_file)
        seg1_path = os.path.join(input_dirs["segmentation1"], seg1_file)
        seg2_path = os.path.join(input_dirs["segmentation2"], seg2_file)

        for i in range(num_augments):
            augment_and_save(img_path, seg1_path, seg2_path, output_dir, augmentations, prefix=f"aug_{i}")


if __name__ == "__main__":
    input_dirs = {
        "images": "/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/dataset/DentalPanoramicXrays/images",
        "segmentation1": "/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/dataset/DentalPanoramicXrays/segmentation1",
        "segmentation2": "/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/dataset/DentalPanoramicXrays/segmentation2",
    }
    output_dir = "/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/dataset/augmented"
    prepare_augmented_data(input_dirs, output_dir, num_augments=5)