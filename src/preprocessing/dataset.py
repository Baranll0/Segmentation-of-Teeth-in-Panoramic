import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, images_dir, segmentation1_dir, segmentation2_dir, image_transform=None, mask_transform=None):
        """
        Args:
            images_dir (str): Path to the directory containing the images.
            segmentation1_dir (str): Path to the directory containing the first segmentation masks.
            segmentation2_dir (str): Path to the directory containing the second segmentation masks.
            image_transform (callable, optional): Transformation to be applied on images.
            mask_transform (callable, optional): Transformation to be applied on segmentation masks.
        """
        self.images_dir = images_dir
        self.segmentation1_dir = segmentation1_dir
        self.segmentation2_dir = segmentation2_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform

        # Get the list of image and mask paths
        self.image_paths = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith('.jpg') or f.endswith('.png')]
        self.segmentation1_paths = [os.path.join(segmentation1_dir, f) for f in os.listdir(segmentation1_dir) if f.endswith('.jpg') or f.endswith('.png')]
        self.segmentation2_paths = [os.path.join(segmentation2_dir, f) for f in os.listdir(segmentation2_dir) if f.endswith('.jpg') or f.endswith('.png')]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load the image and the segmentation masks
        image = Image.open(self.image_paths[idx]).convert('RGB')
        segmentation1 = Image.open(self.segmentation1_paths[idx]).convert('L')  # Grayscale for segmentation mask
        segmentation2 = Image.open(self.segmentation2_paths[idx]).convert('L')  # Grayscale for segmentation mask

        # Apply transformations
        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            segmentation1 = self.mask_transform(segmentation1)
            segmentation2 = self.mask_transform(segmentation2)

        return image, (segmentation1, segmentation2)

# DataLoader helper function
def get_data_loader(images_dir, segmentation1_dir, segmentation2_dir, batch_size=16, image_transform=None, mask_transform=None):
    dataset = CustomDataset(images_dir, segmentation1_dir, segmentation2_dir, image_transform, mask_transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
