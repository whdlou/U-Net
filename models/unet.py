import torch.nn as nn
import torch


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.down_sample = nn.Sequential(
            nn.MaxPool2d(2, 2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down_sample(x)


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()
        self.up_sample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=(1, 1), padding=0),
            nn.ReLU()
        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up_sample(x1)
        x2 = torch.cat([x2, x1], dim=1)
        return self.conv(x2)


class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.feature1 = DoubleConv(3, 64)
        self.feature2 = DownSample(64, 128)
        self.feature3 = DownSample(128, 256)
        self.feature4 = DownSample(256, 512)
        self.feature5 = DownSample(512, 1024)
        self.up_sample1 = Upsample(1024, 512)
        self.up_sample2 = Upsample(512, 256)
        self.up_sample3 = Upsample(256, 128)
        self.up_sample4 = Upsample(128, 64)
        self.classification = nn.Conv2d(64, num_classes, kernel_size=(1, 1))

    def forward(self, x):
        feature1 = self.feature1(x)
        feature2 = self.feature2(feature1)
        feature3 = self.feature3(feature2)
        feature4 = self.feature4(feature3)
        feature5 = self.feature5(feature4)
        x = self.up_sample1(feature5, feature4)
        x = self.up_sample2(x, feature3)
        x = self.up_sample3(x, feature2)
        x = self.up_sample4(x, feature1)
        x = self.classification(x)
        return x


if __name__ == '__main__':
    unet = UNet(32)
    x = torch.randn((1, 3, 128, 128))
    print(x.shape)
    output = unet(x)
    print(output.shape)
