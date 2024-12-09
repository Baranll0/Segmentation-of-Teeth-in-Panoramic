import torch
from sklearn.metrics import precision_score, recall_score, f1_score
from torchmetrics import IoU


# Mevcut fonksiyonlar
def iou_score(pred, target, num_classes=2):
    iou_list = []
    for i in range(num_classes):
        intersection = torch.sum((pred == i) & (target == i))
        union = torch.sum((pred == i) | (target == i))
        iou = intersection.float() / (union.float() + 1e-6)
        iou_list.append(iou)
    return torch.mean(torch.tensor(iou_list))  # Return mean IoU over all classes


def dice_coefficient(pred, target):
    intersection = torch.sum(pred * target)
    return (2. * intersection + 1e-6) / (torch.sum(pred) + torch.sum(target) + 1e-6)


def precision_recall_f1(pred, target):
    pred = pred.flatten().cpu().numpy()
    target = target.flatten().cpu().numpy()
    precision = precision_score(target, pred)
    recall = recall_score(target, pred)
    f1 = f1_score(target, pred)
    return {'precision': precision, 'recall': recall, 'f1': f1}


# Yeni fonksiyon: Modeli değerlendir
def evaluate_model(model, dataloader, device='cuda'):
    """
    Modelin performansını değerlendiren fonksiyon.
    Args:
        model (torch.nn.Module): Eğitilmiş model.
        dataloader (torch.utils.data.DataLoader): Test veri kümesi.
        device (str): Cihaz ('cpu' veya 'cuda').

    Returns:
        dict: Değerlendirme metrikleri (IoU, Dice, Precision, Recall, F1).
    """
    model.eval()  # Değerlendirme moduna geç
    iou_metric = IoU(num_classes=2).to(device)  # IoU metrik hesaplama
    all_preds = []
    all_targets = []

    with torch.no_grad():  # Modeli tahmin yaparken gradyan hesaplamasına gerek yok
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)

            # Prediktif sınıfın seçimi ve metrik hesaplaması
            preds = torch.argmax(outputs, dim=1)

            # IoU hesaplama
            iou_metric.update(preds, targets)

            # Diğer metrikler için verileri sakla
            all_preds.append(preds.flatten())
            all_targets.append(targets.flatten())

    # IoU sonucu
    iou_score = iou_metric.compute()

    # Flatten edilmiş pred ve target tensorları üzerinde precision, recall ve f1 hesapla
    all_preds = torch.cat(all_preds).cpu().numpy()
    all_targets = torch.cat(all_targets).cpu().numpy()

    metrics = precision_recall_f1(all_preds, all_targets)
    metrics['iou'] = iou_score.item()
    metrics['dice'] = dice_coefficient(torch.tensor(all_preds), torch.tensor(all_targets)).item()

    return metrics
