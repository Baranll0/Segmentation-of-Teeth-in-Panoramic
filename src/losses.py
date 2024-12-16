import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """
    Dice Loss: Küçük ve detaylı segmentasyonlar için idealdir.
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        # Softmax uygula
        preds = torch.softmax(preds, dim=1)
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=preds.shape[1]).permute(0, 3, 1, 2).float()

        intersection = torch.sum(preds * targets_one_hot, dim=(0, 2, 3))
        union = torch.sum(preds, dim=(0, 2, 3)) + torch.sum(targets_one_hot, dim=(0, 2, 3))

        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice_score.mean()

class CombinedLoss(nn.Module):
    """
    CrossEntropyLoss ve DiceLoss'u birleştirir.
    """
    def __init__(self, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
        self.alpha = alpha  # CrossEntropy ve Dice Loss arasındaki denge

    def forward(self, preds, targets):
        ce_loss = self.cross_entropy(preds, targets)
        dice_loss = self.dice_loss(preds, targets)
        return self.alpha * ce_loss + (1 - self.alpha) * dice_loss
