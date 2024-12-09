import torch
import cv2
from torchvision import transforms
from PIL import Image
import numpy as np

def load_model(model_path, model_class):
    """
    Modeli yükleme.
    Args:
        model_path (str): Model dosyasının yolu.
        model_class (torch.nn.Module): Modelin sınıfı.
    Returns:
        torch.nn.Module: Yüklenmiş model.
    """
    model = model_class()
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Modeli değerlendirme moduna al
    return model

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
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ResNet için normalize et
    ])
    image_tensor = preprocess(image).unsqueeze(0)  # Batch boyutunu ekle
    return image_tensor

def predict_segmentation(model, image_tensor):
    """
    Segmentasyon tahmini yapmak.
    Args:
        model (torch.nn.Module): Yüklü model.
        image_tensor (torch.Tensor): İşlenmiş görüntü.
    Returns:
        torch.Tensor: Modelin tahmini.
    """
    with torch.no_grad():
        prediction = model(image_tensor)  # Model tahminini al
    return prediction

def predict_classification(model, image_tensor):
    """
    Sınıflandırma tahmini yapmak.
    Args:
        model (torch.nn.Module): Yüklü model.
        image_tensor (torch.Tensor): İşlenmiş görüntü.
    Returns:
        int: Modelin tahmin ettiği sınıf.
    """
    with torch.no_grad():
        logits = model(image_tensor)  # Model tahminini al
    predicted_class = torch.argmax(logits, dim=1).item()  # En yüksek olasılığa sahip sınıfı al
    return predicted_class