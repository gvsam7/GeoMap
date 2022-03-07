import torchvision
from torch import nn
from models.ResNet import ResNet18, ResNet50
from models.CNN import CNN5


def networks(architecture, in_channels, num_classes):
    if architecture == 'cnn5':
        model = CNN5(in_channels, num_classes)
    elif architecture == 'resnet18':
        model = ResNet18(in_channels, num_classes)
    else:
        model = ResNet50(in_channels, num_classes)
    return model