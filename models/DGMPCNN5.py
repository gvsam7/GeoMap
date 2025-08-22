from torch import nn
from models.GaborLayer import GaborConv2d
from models.MixPool import MixPool
from models.Block import DACBlock


class DGMPCNN5(nn.Module):
    def __init__(self, in_channels, num_classes=2):
        super(DGMPCNN5, self).__init__()
        self.features = nn.Sequential(
            GaborConv2d(in_channels, out_channels=32, kernel_size=(3, 3)),
            # nn.Conv2d(in_channels, out_channels=32, kernel_size=(3, 3)),
            nn.ReLU(inplace=True),
            # MixPool(2, 2, 0, 1),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, (3, 3)),
            nn.ReLU(inplace=True),
            # MixPool(2, 2, 0, 0.6),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, (3, 3)),
            nn.ReLU(inplace=True),
            # MixPool(2, 2, 0, 0.2),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, (3, 3)),
            nn.ReLU(inplace=True),
            # MixPool(2, 2, 0, 0.2),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, (3, 3)),
            nn.ReLU(inplace=True),
            # DACBlock(512, 512)
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


