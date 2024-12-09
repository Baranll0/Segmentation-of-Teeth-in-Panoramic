import torch
import numpy as np
import cv2

def postprocess_segmentation(predictions, threshold=0.5):
    """
    Segmentasyon çıktısını işleme.
    Args:
        predictions (torch.Tensor): Model'in tahmin ettiği segmentasyon maskeleri.
        threshold (float): Maskenin 1'e dönüştürülmesi için eşik değeri.
    Returns:
        torch.Tensor: İşlenmiş segmentasyon maskesi.
    """
    predictions = torch.sigmoid(predictions)  # Sigmoid activation for probability maps
    predictions = (predictions > threshold).float()  # Threshold to get binary masks
    return predictions

def postprocess_classification(predictions):
    """
    Sınıflandırma tahminlerini işleme.
    Args:
        predictions (torch.Tensor): Model'in tahmin ettiği sınıflandırma sonuçları (logits).
    Returns:
        int: En yüksek olasılığa sahip sınıf.
    """
    _, predicted_class = torch.max(predictions, dim=1)
    return predicted_class.item()  # Return the predicted class index

def decode_segmentation_mask(mask, num_classes):
    """
    Segmentation maskesinin renkli hale getirilmesi.
    Args:
        mask (torch.Tensor): Segmentasyon maskesi.
        num_classes (int): Sınıf sayısı.
    Returns:
        numpy.ndarray: Renkli segmentasyon maskesi.
    """
    # Create a color palette
    palette = np.random.randint(0, 255, (num_classes, 3), dtype=np.uint8)
    color_mask = palette[mask.cpu().numpy()]
    return color_mask