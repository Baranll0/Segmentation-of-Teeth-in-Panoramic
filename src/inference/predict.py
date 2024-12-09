import torch
from src.preprocessing.preprocess import preprocess_image

def predict_segmentation(model, image_path, device='cuda'):
    model.eval()
    image_tensor = preprocess_image(image_path).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(image_tensor)
    return torch.argmax(prediction, dim=1)