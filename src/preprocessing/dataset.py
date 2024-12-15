import os
from datasets import Dataset, DatasetDict, Image


def create_dataset(image_paths, label_paths):
    dataset = Dataset.from_dict({"pixel_values": sorted(image_paths),
                                 "label": sorted(label_paths)})
    dataset = dataset.cast_column("pixel_values", Image())
    dataset = dataset.cast_column("label", Image())
    return dataset


def load_datasets(img_dir, mask_dir, train_split=0.8):
    img_paths = sorted([os.path.join(img_dir, img) for img in os.listdir(img_dir)])
    mask_paths = sorted([os.path.join(mask_dir, mask) for mask in os.listdir(mask_dir)])

    train_idx = int(len(img_paths) * train_split)
    train_images, val_images = img_paths[:train_idx], img_paths[train_idx:]
    train_masks, val_masks = mask_paths[:train_idx], mask_paths[train_idx:]

    train_dataset = create_dataset(train_images, train_masks)
    val_dataset = create_dataset(val_images, val_masks)

    return DatasetDict({"train": train_dataset, "validation": val_dataset})
