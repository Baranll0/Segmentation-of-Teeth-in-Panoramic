import torch
from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate_model(model, dataloader, device='cuda'):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(device), targets[0].to(device)  # İlk maskeyi kullanıyoruz
            outputs = model(images)

            # Tahminleri ikili değerlere dönüştür
            preds = (outputs > 0.5).long()  # 0.5 eşik değeriyle ikili maskeye çevir
            all_preds.append(preds.flatten())
            all_targets.append(targets.flatten())

    # Liste içindeki tensörleri birleştir
    all_preds = torch.cat(all_preds).cpu().numpy()
    all_targets = torch.cat(all_targets).cpu().numpy()

    # Precision, Recall ve F1-Skorunu hesapla
    precision = precision_score(all_targets, all_preds)
    recall = recall_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds)

    return {'precision': precision, 'recall': recall, 'f1': f1}
