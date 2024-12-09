import torch
from torchvision import transforms
from PIL import Image

def preprocess_image(image_path, resize_shape=(256, 256)):
    """
    Bir görüntüyü model için ön işler.
    """
    transform = transforms.Compose([
        transforms.Resize(resize_shape),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image)

def predict_segmentation(model, image_path, device='cuda'):
    """
    Tek bir görüntü için model tahmini yapar.
    """
    model.eval()
    image_tensor = preprocess_image(image_path).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(image_tensor)
    return prediction.cpu().numpy()