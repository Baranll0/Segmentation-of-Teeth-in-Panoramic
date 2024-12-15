import os
from PIL import Image
from tqdm.auto import tqdm

# Veri yolları
RAW_IMG_DIR = "/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/dataset/datasetkaggle/Teeth Segmentation JSON/d2/img"
RAW_MASK_DIR = "/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/dataset/datasetkaggle/Teeth Segmentation JSON/d2/masks_machine"
PROCESSED_IMG_DIR = "/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/data/processed/img"
PROCESSED_MASK_DIR = "/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/data/processed/mask"

# Klasörleri oluştur
os.makedirs(PROCESSED_IMG_DIR, exist_ok=True)
os.makedirs(PROCESSED_MASK_DIR, exist_ok=True)

# Görüntü ve maskeleri işleme
def process_images_and_masks(raw_img_dir, raw_mask_dir, processed_img_dir, processed_mask_dir, new_size=(512, 512)):
    img_files = sorted(os.listdir(raw_img_dir))
    mask_files = sorted(os.listdir(raw_mask_dir))

    for img_file, mask_file in tqdm(zip(img_files, mask_files), total=len(img_files), desc="Processing Data"):
        img_path = os.path.join(raw_img_dir, img_file)
        mask_path = os.path.join(raw_mask_dir, mask_file)

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        img = img.resize(new_size, Image.Resampling.LANCZOS)
        mask = mask.resize(new_size, Image.Resampling.NEAREST)

        img.save(os.path.join(processed_img_dir, img_file))
        mask.save(os.path.join(processed_mask_dir, mask_file))

process_images_and_masks(RAW_IMG_DIR, RAW_MASK_DIR, PROCESSED_IMG_DIR, PROCESSED_MASK_DIR)
print("Veri işleme tamamlandı!")
