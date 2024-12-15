import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Çift katmanlı konvolüsyon bloğu: Conv2d -> BatchNorm -> ReLU -> Conv2d -> BatchNorm -> ReLU
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    """
    UNet modeli.
    Encoder-Decoder yapısında bir modeldir.
    """
    def __init__(self, input_channels=3, num_classes=1):
        super(UNet, self).__init__()

        # Encoder
        self.enc1 = DoubleConv(input_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.enc5 = DoubleConv(512, 1024)

        # Max Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        # Final output
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)  # [B, 64, H, W]
        enc2 = self.enc2(self.pool(enc1))  # [B, 128, H/2, W/2]
        enc3 = self.enc3(self.pool(enc2))  # [B, 256, H/4, W/4]
        enc4 = self.enc4(self.pool(enc3))  # [B, 512, H/8, W/8]
        enc5 = self.enc5(self.pool(enc4))  # [B, 1024, H/16, W/16]

        # Decoder
        dec4 = self.upconv4(enc5)  # [B, 512, H/8, W/8]
        dec4 = self.dec4(torch.cat((dec4, enc4), dim=1))  # [B, 512, H/8, W/8]

        dec3 = self.upconv3(dec4)  # [B, 256, H/4, W/4]
        dec3 = self.dec3(torch.cat((dec3, enc3), dim=1))  # [B, 256, H/4, W/4]

        dec2 = self.upconv2(dec3)  # [B, 128, H/2, W/2]
        dec2 = self.dec2(torch.cat((dec2, enc2), dim=1))  # [B, 128, H/2, W/2]

        dec1 = self.upconv1(dec2)  # [B, 64, H, W]
        dec1 = self.dec1(torch.cat((dec1, enc1), dim=1))  # [B, 64, H, W]

        # Final output
        output = self.final_conv(dec1)  # [B, num_classes, H, W]

        return output
