import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50

class DeepLabV3Plus(nn.Module):
    def __init__(self, input_channels=3, num_classes=33, pretrained=True):
        super(DeepLabV3Plus, self).__init__()
        self.model = deeplabv3_resnet50(pretrained=pretrained)
        self.model.backbone.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        return self.model(x)['out']
