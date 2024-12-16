import torch
import torch.nn as nn
import torch.nn.functional as F

class NestedUNet(nn.Module):
    def __init__(self, input_channels=3, output_channels=33):  # 33 sınıf için
        super(NestedUNet, self).__init__()
        nb_filter = [64, 128, 256, 512, 1024]

        # Encoder
        self.conv0_0 = self.conv_block(input_channels, nb_filter[0])
        self.conv1_0 = self.conv_block(nb_filter[0], nb_filter[1])
        self.conv2_0 = self.conv_block(nb_filter[1], nb_filter[2])
        self.conv3_0 = self.conv_block(nb_filter[2], nb_filter[3])
        self.conv4_0 = self.conv_block(nb_filter[3], nb_filter[4])

        # Decoder
        self.up1_0 = self.up_block(nb_filter[1], nb_filter[0])
        self.up2_0 = self.up_block(nb_filter[2], nb_filter[1])
        self.up3_0 = self.up_block(nb_filter[3], nb_filter[2])
        self.up4_0 = self.up_block(nb_filter[4], nb_filter[3])

        # Final output layer
        self.final = nn.Conv2d(nb_filter[0], output_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            self.conv_block(out_channels, out_channels)
        )

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(F.max_pool2d(x0_0, 2))
        x2_0 = self.conv2_0(F.max_pool2d(x1_0, 2))
        x3_0 = self.conv3_0(F.max_pool2d(x2_0, 2))
        x4_0 = self.conv4_0(F.max_pool2d(x3_0, 2))

        x1_0 = self.up1_0(x1_0)
        x2_0 = self.up2_0(x2_0)
        x3_0 = self.up3_0(x3_0)
        x4_0 = self.up4_0(x4_0)

        out = self.final(x1_0)
        return out
