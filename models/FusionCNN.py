from torch import nn

CNN_arch = {
    'CNN4': [32, 'M', 64, 'M', 128, 'M', 256],
    'CNN5': [32, 'M', 64, 'M', 128, 'M', 256, 'M', 512],
    'CNN6': [32, 'M', 64, 'M', 128, 'M', 256, 'M', 512, 'M', 1024],
    'CNN7': [32, 'M', 64, 'M', 128, 'M', 256, 256, 'M', 512, 512]
}


class FusionCNN(nn.Module):
    def __init__(self, CNN_arch, in_linear, in_channels=3, num_classes=2):
        super(FusionCNN, self).__init__()
        self.in_channels = in_channels
        self.features = self.conv_layers(CNN_arch)

    def forward(self, x):
        x = self.features(x)
        return x

    def conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int:
                out_channels = x

                layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                           nn.ReLU(inplace=True),
                           nn.BatchNorm2d(x)]
                in_channels = x
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

        return nn.Sequential(*layers)


def FusionCNN4(in_channels=3, num_classes=2):
    in_linear = CNN_arch['CNN4'][-1]
    return FusionCNN(CNN_arch['CNN4'], in_linear, in_channels, num_classes)


def FusionCNN5(in_channels=3, num_classes=2):
    in_linear = CNN_arch['CNN5'][-1]
    return FusionCNN(CNN_arch['CNN5'], in_linear, in_channels, num_classes)


def FusionCNN6(in_channels=3, num_classes=2):
    in_linear = CNN_arch['CNN6'][-1]
    return FusionCNN(CNN_arch['CNN6'], in_linear, in_channels, num_classes)


def FusionCNN7(in_channels=3, num_classes=2):
    in_linear = CNN_arch['CNN7'][-1]
    return FusionCNN(CNN_arch['CNN7'], in_linear, in_channels, num_classes)

