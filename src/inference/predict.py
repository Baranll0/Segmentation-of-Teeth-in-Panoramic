import torch
from src.preprocessing.preprocess import preprocess_image

def predict_segmentation(model, image_path, device='cuda'):
    """
    Model kullanılarak bir görüntü üzerinde segmentasyon tahmini yapar.
    Args:
        model (torch.nn.Module): Eğitimli model.
        image_path (str): Tahmin yapılacak görüntünün yolu.
        device (str): Kullanılacak cihaz ('cuda' veya 'cpu').
    Returns:
        torch.Tensor: Modelin segmentasyon tahmini.
    """
    model.eval()
    image_tensor = preprocess_image(image_path).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(image_tensor)
        return prediction