import torch.nn as nn

VGG_arch = {
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
}


class VGG(nn.Module):
    def __init(self, VGG_arch, in_channels=3, num_classes=2):
        super(VGG, self).__init__()
        self.in_channels = in_channels
        self.conv = self.conv_layers(VGG_arch)

        self.fc = nn.Sequential(
            nn.Linear(512*8*8, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

    def conv_layer(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int:
                out_channels = x

                layers += [nn.Conv2d(inchannels=in_channels, out_channels=out_channels,
                                     kenrel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                           nn.BatchNorm2d(x),
                           nn.ReLU()]
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]
        return nn.Sequential(*layers)


def VGG13(in_channels=3, num_classes=2):
    return VGG(VGG_arch['VGG13'], in_channels, num_classes)

