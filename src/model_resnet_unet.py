import torch
import torch.nn as nn
import torchvision.models as models

class ResNetUNet(nn.Module):
    def __init__(self, input_channels=3, output_channels=1, last_activation="sigmoid"):
        super(ResNetUNet, self).__init__()

        self.last_activation = last_activation

        # Encoder (ResNet50 pretrained model)
        base_model = models.resnet50(pretrained=True)
        self.encoder = nn.Sequential(*list(base_model.children())[:-2])  # Remove fully connected layer

        # Decoder
        self.upconv4 = self.upconv_block(2048, 1024)
        self.dec4 = self.conv_block(2048, 1024)

        self.upconv3 = self.upconv_block(1024, 512)
        self.dec3 = self.conv_block(1024, 512)

        self.upconv2 = self.upconv_block(512, 256)
        self.dec2 = self.conv_block(512, 256)

        self.upconv1 = self.upconv_block(256, 64)
        self.dec1 = self.conv_block(128, 64)

        # Output layer
        self.final_conv = nn.Conv2d(64, output_channels, kernel_size=1)

    def upconv_block(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        # Encoder
        x1 = self.encoder[0:4](x)
        x2 = self.encoder[4](x1)
        x3 = self.encoder[5](x2)
        x4 = self.encoder[6](x3)
        x5 = self.encoder[7](x4)

        # Decoder
        d4 = self.upconv4(x5)
        d4 = torch.cat([d4, x4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, x3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, x2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, x1], dim=1)
        d1 = self.dec1(d1)

        # Final Output
        output = self.final_conv(d1)
        if self.last_activation == "sigmoid":
            output = torch.sigmoid(output)
        return output


# Model Test
if __name__ == "__main__":
    model = ResNetUNet(input_channels=3, output_channels=1).cuda()
    x = torch.randn((1, 3, 512, 512)).cuda()
    output = model(x)
    print("Output Shape:", output.shape)  # Should be [1, 1, 512, 512]
