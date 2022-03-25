from torch import nn

CNN_arch = {
    'CNN4': [32, 64, 128, 256],
    'CNN5': [32, 64, 128, 256, 512],
    'CNN6': [32, 64, 128, 256, 512, 1024]
}


class CNN(nn.Module):
    def __init__(self, CNN_arch, in_linear, in_channels=3, num_classes=2):
        super(CNN, self).__init__()
        self.in_channels = in_channels
        self.features = self.conv_layers(CNN_arch)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(in_linear, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

    def conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            out_channels = x

            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                       nn.ReLU(inplace=True),
                       nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
                       nn.BatchNorm2d(x)]
            in_channels = x

        return nn.Sequential(*layers)


def CNN_4(in_channels=3, num_classes=2):
    in_linear = CNN_arch['CNN4'][-1]
    return CNN(CNN_arch['CNN4'], in_linear, in_channels, num_classes)


def CNN_5(in_channels=3, num_classes=2):
    in_linear = CNN_arch['CNN5'][-1]
    return CNN(CNN_arch['CNN5'], in_linear, in_channels, num_classes)


def CNN_6(in_channels=3, num_classes=2):
    in_linear = CNN_arch['CNN6'][-1]
    return CNN(CNN_arch['CNN6'], in_linear, in_channels, num_classes)


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
