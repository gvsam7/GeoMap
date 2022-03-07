from torch import nn
import torch


class CNN5(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CNN5, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=32, kernel_size=(3, 3)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, (3, 3)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, (3, 3)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, (3, 3)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, (3, 3)),
            nn.ReLU(inplace=True)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x


"""
class CNN3(nn.Module):
    def __init(self, in_channels=3, num_classes=2):
        super(CNN3, self).__init__()
        self.features = nn.Sequential(
            ConvLayer(in_channels, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ConvLayer(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Conv2d(64, 128),
            nn.ReLU(inplace=True)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvLayer).__init__()
        self.conv = nn.Sequence(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x1):
        x1 = self.conv(x1)
        return x1

"""

