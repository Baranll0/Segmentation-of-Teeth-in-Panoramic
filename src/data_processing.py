import os
import json
from PIL import Image
from datasets import Dataset, DatasetDict
from torchvision.transforms import Resize, ToTensor
from pathlib import Path
from typing import List, Tuple

# Paths to the data folders
data_root_path = "/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/dataset/datasetkaggle/Teeth Segmentation JSON/d2"
image_dir = os.path.join(data_root_path, "img")
mask_dir = os.path.join(data_root_path, "masks_machine")

# Resize dimensions
resize_dim = (512, 512)

# Function to resize and save images
def resize_and_save_images(image_paths: List[str], output_dir: str, size: Tuple[int, int]):
    os.makedirs(output_dir, exist_ok=True)
    for img_path in image_paths:
        img = Image.open(img_path).convert("RGB")
        img_resized = img.resize(size, Image.LANCZOS)
        output_path = os.path.join(output_dir, os.path.basename(img_path))
        img_resized.save(output_path)

# Function to resize and save masks
def resize_and_save_masks(mask_paths: List[str], output_dir: str, size: Tuple[int, int]):
    os.makedirs(output_dir, exist_ok=True)
    for mask_path in mask_paths:
        mask = Image.open(mask_path).convert("P")
        mask_resized = mask.resize(size, Image.NEAREST)
        output_path = os.path.join(output_dir, os.path.basename(mask_path))
        mask_resized.save(output_path)

# Function to create a Dataset object
def create_dataset(image_paths, mask_paths):
    def load_image(image_path):
        return ToTensor()(Image.open(image_path).convert("RGB"))

    def load_mask(mask_path):
        return ToTensor()(Image.open(mask_path).convert("P"))

    dataset = Dataset.from_dict({
        "image": [load_image(p) for p in image_paths],
        "mask": [load_mask(p) for p in mask_paths],
    })
    return dataset

# Main data processing function
def process_data(image_dir: str, mask_dir: str, output_dir: str, resize_dim: Tuple[int, int]):
    # List all images and masks
    image_paths = sorted([os.path.join(image_dir, img) for img in os.listdir(image_dir)])
    mask_paths = sorted([os.path.join(mask_dir, mask) for mask in os.listdir(mask_dir)])

    # Prepare output directories
    resized_image_dir = os.path.join(output_dir, "resized_images")
    resized_mask_dir = os.path.join(output_dir, "resized_masks")

    # Resize images and masks
    resize_and_save_images(image_paths, resized_image_dir, resize_dim)
    resize_and_save_masks(mask_paths, resized_mask_dir, resize_dim)

    # Create datasets
    train_idx = int(len(image_paths) * 0.8)
    train_dataset = create_dataset(image_paths[:train_idx], mask_paths[:train_idx])
    val_dataset = create_dataset(image_paths[train_idx:], mask_paths[train_idx:])

    dataset_dict = DatasetDict({"train": train_dataset, "validation": val_dataset})
    return dataset_dict

if __name__ == "__main__":
    # Output directory for processed data
    output_dir = "/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/data/processed"

    # Process data and get the dataset object
    dataset = process_data(image_dir, mask_dir, output_dir, resize_dim)
    print("Data processing complete. Dataset ready for training.")
