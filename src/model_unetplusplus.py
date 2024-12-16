import torch
import torch.nn as nn


class NestedUNet(nn.Module):
    def __init__(self, input_channels=3, output_channels=1, deep_supervision=False):
        super(NestedUNet, self).__init__()
        self.deep_supervision = deep_supervision
        nb_filter = [32, 64, 128, 256, 512]

        # Encoder layers
        self.conv0_0 = self.conv_block(input_channels, nb_filter[0])
        self.conv1_0 = self.conv_block(nb_filter[0], nb_filter[1])
        self.conv2_0 = self.conv_block(nb_filter[1], nb_filter[2])
        self.conv3_0 = self.conv_block(nb_filter[2], nb_filter[3])
        self.conv4_0 = self.conv_block(nb_filter[3], nb_filter[4])

        # Decoder layers
        self.conv0_1 = self.conv_block(nb_filter[0] + nb_filter[1], nb_filter[0])
        self.conv1_1 = self.conv_block(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv2_1 = self.conv_block(nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv3_1 = self.conv_block(nb_filter[3] + nb_filter[4], nb_filter[3])

        self.conv0_2 = self.conv_block(nb_filter[0] * 2 + nb_filter[1], nb_filter[0])
        self.conv1_2 = self.conv_block(nb_filter[1] * 2 + nb_filter[2], nb_filter[1])
        self.conv2_2 = self.conv_block(nb_filter[2] * 2 + nb_filter[3], nb_filter[2])

        self.conv0_3 = self.conv_block(nb_filter[0] * 3 + nb_filter[1], nb_filter[0])
        self.conv1_3 = self.conv_block(nb_filter[1] * 3 + nb_filter[2], nb_filter[1])

        self.conv0_4 = self.conv_block(nb_filter[0] * 4 + nb_filter[1], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], output_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.downsample(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.upsample(x1_0)], dim=1))

        x2_0 = self.conv2_0(self.downsample(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.upsample(x2_0)], dim=1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.upsample(x1_1)], dim=1))

        x3_0 = self.conv3_0(self.downsample(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.upsample(x3_0)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.upsample(x2_1)], dim=1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.upsample(x1_2)], dim=1))

        x4_0 = self.conv4_0(self.downsample(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.upsample(x4_0)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.upsample(x3_1)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.upsample(x2_2)], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.upsample(x1_3)], dim=1))

        output = self.final(x0_4)
        return torch.sigmoid(output)

    def downsample(self, x):
        return nn.functional.max_pool2d(x, kernel_size=2)

    def upsample(self, x):
        return nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)


if __name__ == "__main__":
    model = NestedUNet(input_channels=3, output_channels=1).cuda()
    x = torch.randn((1, 3, 512, 512)).cuda()
    output = model(x)
    print("Output shape:", output.shape)
