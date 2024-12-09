import numpy as np

def postprocess_segmentation(prediction, threshold=0.5):
    """
    Model tahminini ikili maske haline dönüştürür.
    """
    prediction = np.squeeze(prediction)  # (1, H, W) -> (H, W)
    binary_mask = (prediction > threshold).astype(np.uint8)
    return binary_mask