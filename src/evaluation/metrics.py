import torch
from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate_model(model, dataloader, device='cuda'):
    """
    Modeli değerlendiren fonksiyon.
    Args:
        model (torch.nn.Module): Eğitimli model.
        dataloader (torch.utils.data.DataLoader): Test veya validation dataloader.
        device (str): Kullanılacak cihaz ('cuda' veya 'cpu').

    Returns:
        dict: Precision, Recall ve F1 skorlarını içeren metrikler.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(device), targets[0].to(device)  # İlk maskeyi kullanıyoruz
            outputs = model(images)

            # Tahminleri 0 ve 1'e dönüştür (binary thresholding)
            preds = (outputs > 0.5).long()

            # Hedef değerleri binary (0 ve 1) hale getir
            targets = (targets > 0.5).long()

            all_preds.append(preds.flatten())
            all_targets.append(targets.flatten())

    # Liste içindeki tensörleri birleştir
    all_preds = torch.cat(all_preds).cpu().numpy()
    all_targets = torch.cat(all_targets).cpu().numpy()

    # Precision, Recall ve F1-Skorunu hesapla
    precision = precision_score(all_targets, all_preds, average='binary')
    recall = recall_score(all_targets, all_preds, average='binary')
    f1 = f1_score(all_targets, all_preds, average='binary')

    return {'precision': precision, 'recall': recall, 'f1': f1}