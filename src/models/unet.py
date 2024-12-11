import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        """
        Initialize the UNet model.

        Args:
            in_channels (int): Number of input channels (e.g., 1 for grayscale images).
            out_channels (int): Number of output channels (e.g., 1 for binary segmentation).
        """
        super(UNet, self).__init__()

        # Encoder layers
        self.encoder1 = self.conv_block(in_channels, 32)
        self.encoder2 = self.conv_block(32, 64)
        self.encoder3 = self.conv_block(64, 128)
        self.encoder4 = self.conv_block(128, 256)

        # Pooling layer
        self.pool = nn.MaxPool2d(2)

        # Decoder layers
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)

        self.decoder3 = self.conv_block(256, 128)
        self.decoder2 = self.conv_block(128, 64)
        self.decoder1 = self.conv_block(64, 32)

        # Final output layer
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        """
        Creates a convolutional block with two Conv2D layers, ReLU activations, and Batch Normalization.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.

        Returns:
            nn.Sequential: A convolutional block.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        """
        Forward pass of the UNet model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        # Encoder path
        enc1 = self.encoder1(x)  # First encoder block
        enc2 = self.encoder2(self.pool(enc1))  # Second encoder block
        enc3 = self.encoder3(self.pool(enc2))  # Third encoder block
        enc4 = self.encoder4(self.pool(enc3))  # Fourth encoder block

        # Decoder path
        dec3 = self.upconv3(enc4)  # Upsample from bottleneck
        dec3 = torch.cat((dec3, enc3), dim=1)  # Skip connection
        dec3 = self.decoder3(dec3)  # Third decoder block

        dec2 = self.upconv2(dec3)  # Upsample
        dec2 = torch.cat((dec2, enc2), dim=1)  # Skip connection
        dec2 = self.decoder2(dec2)  # Second decoder block

        dec1 = self.upconv1(dec2)  # Upsample
        dec1 = torch.cat((dec1, enc1), dim=1)  # Skip connection
        dec1 = self.decoder1(dec1)  # First decoder block

        # Output layer
        return torch.sigmoid(self.final_conv(dec1))
