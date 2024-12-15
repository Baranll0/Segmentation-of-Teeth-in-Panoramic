import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, input_channels=1, output_channels=1):
        super(UNet, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(input_channels, 32, dropout=0.1)
        self.enc2 = self.conv_block(32, 64, dropout=0.2)
        self.enc3 = self.conv_block(64, 128, dropout=0.3)
        self.enc4 = self.conv_block(128, 256, dropout=0.4)
        self.enc5 = self.conv_block(256, 512, dropout=0.5)

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.dec4 = self.conv_block(512, 256, dropout=0.4)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.dec3 = self.conv_block(256, 128, dropout=0.3)

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.dec2 = self.conv_block(128, 64, dropout=0.2)

        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.dec1 = self.conv_block(64, 32, dropout=0.1)

        # Output
        self.final_conv = nn.Conv2d(32, output_channels, kernel_size=1, padding=0)

    def conv_block(self, in_channels, out_channels, dropout):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, kernel_size=2))
        enc3 = self.enc3(F.max_pool2d(enc2, kernel_size=2))
        enc4 = self.enc4(F.max_pool2d(enc3, kernel_size=2))
        enc5 = self.enc5(F.max_pool2d(enc4, kernel_size=2))

        # Decoder
        dec4 = self.upconv4(enc5)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)

        # Output
        output = self.final_conv(dec1)
        return output


# Example usage
if __name__ == "__main__":
    model = UNet(input_channels=1, output_channels=1)
    x = torch.randn(1, 1, 512, 512)
    output = model(x)
    print(output.shape)
