from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

class SegmentationDataset(Dataset):
    def __init__(self, images_dir, segmentation1_dir, segmentation2_dir, transform=None):
        self.images_dir = images_dir
        self.segmentation1_dir = segmentation1_dir
        self.segmentation2_dir = segmentation2_dir
        self.image_paths = sorted(os.listdir(images_dir))
        self.segmentation1_paths = sorted(os.listdir(segmentation1_dir))
        self.segmentation2_paths = sorted(os.listdir(segmentation2_dir))
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load images
        image_path = os.path.join(self.images_dir, self.image_paths[idx])
        segmentation1_path = os.path.join(self.segmentation1_dir, self.segmentation1_paths[idx])
        segmentation2_path = os.path.join(self.segmentation2_dir, self.segmentation2_paths[idx])

        image = Image.open(image_path).convert("RGB")
        segmentation1 = Image.open(segmentation1_path).convert("L")
        segmentation2 = Image.open(segmentation2_path).convert("L")

        if self.transform:
            image = self.transform(image)
            segmentation1 = self.transform(segmentation1)
            segmentation2 = self.transform(segmentation2)

        return image, (segmentation1, segmentation2)


from torch.utils.data import DataLoader

def get_data_loader(images_dir, segmentation1_dir, segmentation2_dir, batch_size=16):
    dataset = SegmentationDataset(images_dir, segmentation1_dir, segmentation2_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
