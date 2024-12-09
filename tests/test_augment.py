import pytest
from PIL import Image
from preprocessing.augment import augment_image


def test_augment_image():
    """
    augment_image fonksiyonunun doğru çalıştığını test eder.
    """
    # Test için örnek bir görüntü yükleyelim
    image_path = "tests/test_image.jpg"
    image = Image.open(image_path)

    augmented_image = augment_image(image)

    # Augmentasyon sonrası görüntü boyutunun değişip değişmediğini kontrol et
    assert augmented_image.size == image.size, "Görüntü boyutu değişti"
    