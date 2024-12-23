import torch.nn as nn

VGG_arch = {
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
}


class VGGFusion(nn.Module):
    def __init__(self, VGG_arch, in_channels=3, num_classes=2):
        super(VGGFusion, self).__init__()
        self.in_channels = in_channels
        self.conv = self.conv_layers(VGG_arch)

    def forward(self, x):
        x = self.conv(x)
        return x

    def conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int:
                out_channels = x

                layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                           nn.BatchNorm2d(x),
                           nn.ReLU()]
                in_channels = x
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

        return nn.Sequential(*layers)  # unpacks all that is stored on the layers list


def VGG13Fusion(in_channels=3, num_classes=2):
    return VGGFusion(VGG_arch['VGG13'], in_channels, num_classes)