import torch
from torchvision import transforms
import random


class Augmentations:
    def __init__(self):
        self.augment = transforms.Compose([
            transforms.RandomHorizontalFlip(),  # Yatay çevirme
            transforms.RandomVerticalFlip(),  # Dikey çevirme
            transforms.RandomRotation(30),  # Rastgele döndürme
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Renk ayarları
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Çevirme ve kaydırma
        ])

    def __call__(self, image):
        """
        Görüntü üzerinde veri artırma işlemleri yapar.
        Args:
            image (PIL.Image): Giriş görüntüsü.
        Returns:
            PIL.Image: Veri artırımı yapılmış görüntü.
        """
        return self.augment(image)


def augment_image(image):
    """
    Görüntü üzerinde artırma yapmak için augmentasyon sınıfını kullanır.
    Args:
        image (PIL.Image): Giriş görüntüsü.
    Returns:
        PIL.Image: Artırılmış görüntü.
    """
    augmenter = Augmentations()
    return augmenter(image)