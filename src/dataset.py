import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Normalize
from PIL import Image
import numpy as np

class MultiClassTeethDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))

        # Normalizasyon
        self.transform = Compose([ToTensor(), Normalize(mean=[0.5], std=[0.5])])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        # Görüntüyü ve maskeyi yükle
        image = Image.open(image_path).convert("RGB")
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)

        image = self.transform(image)
        mask = torch.from_numpy(mask).long()
        return image, mask
