import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        x = self.batchnorm(x)
        return x

class UNet(nn.Module):
    def __init__(self, input_channels=1, last_activation='sigmoid'):
        super(UNet, self).__init__()

        # Encoder
        self.enc1 = ConvBlock(input_channels, 32, dropout=0.1)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(32, 64, dropout=0.2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(64, 128, dropout=0.3)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = ConvBlock(128, 256, dropout=0.4)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(256, 512, dropout=0.5)

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.dec4 = ConvBlock(512, 256, dropout=0.4)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.dec3 = ConvBlock(256, 128, dropout=0.3)

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.dec2 = ConvBlock(128, 64, dropout=0.2)

        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.dec1 = ConvBlock(64, 32, dropout=0.1)

        # Final Convolution
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)
        self.last_activation = nn.Sigmoid() if last_activation == 'sigmoid' else nn.Identity()

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))

        # Decoder
        dec4 = self.dec4(torch.cat([self.upconv4(bottleneck), enc4], dim=1))
        dec3 = self.dec3(torch.cat([self.upconv3(dec4), enc3], dim=1))
        dec2 = self.dec2(torch.cat([self.upconv2(dec3), enc2], dim=1))
        dec1 = self.dec1(torch.cat([self.upconv1(dec2), enc1], dim=1))

        # Final Output
        output = self.final_conv(dec1)
        return self.last_activation(output)