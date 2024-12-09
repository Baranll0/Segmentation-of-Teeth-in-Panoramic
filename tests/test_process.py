import pytest
import torch
from preprocessing.preprocess import preprocess_image


def test_preprocess_image():
    """
    preprocess_image fonksiyonunun doğru çalıştığını test eder.
    """
    image_path = "tests/test_image.jpg"

    # Örnek bir boyut (224x224)
    input_size = (224, 224)

    # Görüntü işleme
    image_tensor = preprocess_image(image_path, input_size)

    # Çıktı tensörünün doğru boyutta olup olmadığını kontrol et
    assert image_tensor.shape == (
    1, 3, 224, 224), f"Beklenen boyut: (1, 3, 224, 224), ancak gelen: {image_tensor.shape}"

    # Tensör değerlerinin normalize edilmiş olup olmadığını kontrol et (tensör içeriği rastgele olacağı için bu örnek kontrol)
    assert torch.all(image_tensor >= 0.0) and torch.all(image_tensor <= 1.0), "Görüntü normalize edilmemiş!"
