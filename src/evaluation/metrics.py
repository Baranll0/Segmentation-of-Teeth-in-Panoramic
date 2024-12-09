import torch
from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate_model(model, dataloader, device='cuda'):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(device), targets[0].to(device)  # İlk maskeyi kullan
            outputs = model(images)

            # Tahminleri ikili değerlere dönüştür (0 veya 1)
            preds = (outputs > 0.5).long().flatten()  # 0.5 eşik değeriyle ikiliye çevir
            targets = targets.flatten().long()  # Hedef değerleri de long tipe çevir

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # Precision, Recall ve F1-Skorunu hesapla
    precision = precision_score(all_targets, all_preds, average='binary', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='binary', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='binary', zero_division=0)

    return {'precision': precision, 'recall': recall, 'f1': f1}
