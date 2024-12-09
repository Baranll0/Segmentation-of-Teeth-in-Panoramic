from torchvision import transforms
from PIL import Image
import torch

def preprocess_image(image_path, input_size=(224, 224)):
    """
    Görüntüyü model için ön işleme tabi tutma.
    Args:
        image_path (str): Görüntü dosyasının yolu.
        input_size (tuple): Modelin girdi boyutları.
    Returns:
        torch.Tensor: Modelin alabileceği şekilde işlenmiş görüntü.
    """
    image = Image.open(image_path).convert('RGB')  # Görüntüyü aç
    preprocess = transforms.Compose([
        transforms.Resize(input_size),  # Görüntüyü yeniden boyutlandırma
        transforms.ToTensor(),  # Tensöre dönüştürme
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize etme
    ])
    image_tensor = preprocess(image).unsqueeze(0)  # Batch boyutunu ekle
    return image_tensor