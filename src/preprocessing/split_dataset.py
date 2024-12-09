import os
import random
import shutil

def split_data(input_dir, output_dirs, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Veriyi train, val ve test olarak böler.
    Args:
        input_dir (str): Kaynak veri yolu.
        output_dirs (dict): Hedef yollar.
        train_ratio (float): Eğitim seti oranı.
        val_ratio (float): Validation seti oranı.
        test_ratio (float): Test seti oranı.
    """
    os.makedirs(output_dirs["train"], exist_ok=True)
    os.makedirs(output_dirs["val"], exist_ok=True)
    os.makedirs(output_dirs["test"], exist_ok=True)

    image_files = sorted(os.listdir(os.path.join(input_dir, "images")))

    random.shuffle(image_files)
    total = len(image_files)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    splits = {
        "train": image_files[:train_end],
        "val": image_files[train_end:val_end],
        "test": image_files[val_end:],
    }

    for split, files in splits.items():
        for file in files:
            shutil.copy(os.path.join(input_dir, "images", file), os.path.join(output_dirs[split], "images", file))
            shutil.copy(os.path.join(input_dir, "segmentation1", file), os.path.join(output_dirs[split], "segmentation1", file))
            shutil.copy(os.path.join(input_dir, "segmentation2", file), os.path.join(output_dirs[split], "segmentation2", file))


if __name__ == "__main__":
    input_dir = "/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/dataset/augmented"
    output_dirs = {
        "train": "/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/dataset/train",
        "val": "/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/dataset/val",
        "test": "/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/dataset/test",
    }
    split_data(input_dir, output_dirs)
