import torchvision
from torch import nn
from models.ResNet import ResNet18, ResNet50
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import vgg13, VGG13_Weights
from torchvision.models import densenet161, DenseNet161_Weights
from torchvision.models import vit_s_16, vit_b_16
from models.DilGabMPResNet18 import DilGabMPResNet18, DilGabMPResNet50
from models.FusionNet import FusionNet
from models.ResFusionNet import ResFusionNet
from models.CNN import CNN4, CNN5, CNN6, CNN7
from models.VGG import VGG13
from models.ViT import ViT
from models.Identity import Identity
from models.EfficientNet import EfficientNet
from models.DGMPCNN5 import DGMPCNN5


def networks(architecture, in_channels, num_classes, pretrained, requires_grad, global_pooling, version):
    if architecture == 'cnn4':
        model = CNN4(in_channels, num_classes)
    elif architecture == 'cnn5':
        model = CNN5(in_channels, num_classes)
    elif architecture == 'cnn6':
        model = CNN6(in_channels, num_classes)
    elif architecture == 'cnn7':
        model = CNN7(in_channels, num_classes)
    elif architecture == 'dgmpcnn5':
        model = DGMPCNN5(in_channels, num_classes)
    elif architecture == 'efficientnet':
        print(f"version: {version}")
        model = EfficientNet(version, num_classes)
    elif architecture == 'resnet18':
        model = ResNet18(in_channels, num_classes)
    elif architecture == 'dilgabmpresnet18':
        model = DilGabMPResNet18(in_channels, num_classes)
    elif architecture == 'dilgabmpresnet50':
        model = DilGabMPResNet50(in_channels, num_classes)
    elif architecture == 'vit':
        model = ViT(in_channels, num_classes)
    elif architecture == 'vit_s16':
        model = vit_s_16(weights=None)
        # replace classifier head
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    elif architecture == 'vit_b16':
        model = vit_b_16(weights=None)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    elif architecture == 'fusionnet':
        model = FusionNet(in_channels, num_classes)
    elif architecture == 'resfusionnet':
        model = ResFusionNet(in_channels, num_classes)
    elif architecture == 'tlresnet18':
        if pretrained == 'True':
            print(f"Transfer Learning, Pretrained = {pretrained}")
            weights = ResNet18_Weights.DEFAULT
        else:
            weights = None

        model = resnet18(weights=weights)

        if pretrained == 'True':
            for param in model.parameters():
                param.requires_grad = requires_grad
            print(f"requires_grad = {requires_grad}")

        model.fc = nn.Linear(512, num_classes)
        """elif architecture == 'tlresnet18':
        model = torchvision.models.resnet18(pretrained)
        if pretrained == 'True':
            print(f"Transfer Learning, Pretrained = {pretrained}")
            for param in model.parameters():
                param.requires_grad = requires_grad
            print(f"requires_grad = {requires_grad}")
            model.fc = nn.Linear(512, num_classes)"""
    elif architecture == 'resnet50':
        model = ResNet50(in_channels, num_classes)
    elif architecture == 'tlresnet50':
        if pretrained == 'True':
            print(f"Transfer Learning, Pretrained = {pretrained}")
            weights = ResNet50_Weights.DEFAULT
        else:
            weights = None

        model = resnet50(weights=weights)

        if pretrained == 'True':
            for param in model.parameters():
                param.requires_grad = requires_grad
            print(f"requires_grad = {requires_grad}")

        model.fc = nn.Linear(2048, num_classes)
    elif architecture == 'tlvgg13':
        use_pretrained = (pretrained == 'True')

        weights = VGG13_Weights.DEFAULT if use_pretrained else None
        model = vgg13(weights=weights)

        if use_pretrained:
            print(f"Transfer Learning, Pretrained = {pretrained}")
            for param in model.parameters():
                param.requires_grad = requires_grad
            print(f"requires_grad = {requires_grad}")

        # Replace classifier head
        model.classifier[6] = nn.Linear(4096, num_classes)
        """elif architecture == 'tlresnet50':
        model = torchvision.models.resnet50(pretrained)
        if pretrained == 'True':
            print(f"Transfer Learning, Pretrained={pretrained}")
            for param in model.parameters():
                param.requires_grad = requires_grad
            print(f"requires_grad = {requires_grad}")
            model.fc = nn.Linear(512*4, num_classes)
    elif architecture == 'vgg13':
        model = VGG13(in_channels, num_classes)
    # Load a pretrained model and modify it
    elif architecture == 'tlvgg13':
        model = torchvision.models.vgg13(pretrained)
        if pretrained == 'True':
            print(f"Transfer Learning, Pretrained={pretrained}")
            # Does not change the layers up to this point
            # Freeze the parameters so that the gradients are not computed in backward()
            for param in model.parameters():
                param.requires_grad = requires_grad
            print(f"requires_grad = {requires_grad}")
            if global_pooling == 'GP':
                print(f"Global Pooling: {global_pooling}")
                model.classifier = nn.Sequential(nn.Linear(25088, 4096),
                                                 nn.ReLU(),
                                                 nn.Dropout(),
                                                 nn.Linear(4096, 4096),
                                                 nn.ReLU(),
                                                 nn.Dropout(),
                                                 nn.Linear(4096, 2)
                                                 )
            else:
                print(f"Global Pooling: {global_pooling}")
                model.avgpool = Identity()
                # This will only train the last layers
                model.classifier = nn.Sequential(nn.Linear(32768, 4096),
                                                 nn.ReLU(),
                                                 nn.Dropout(),
                                                 nn.Linear(4096, 4096),
                                                 nn.ReLU(),
                                                 nn.Dropout(),
                                                 nn.Linear(4096, 2))
        else:
            print(f"Fully trained from Sat data, Pretrained={pretrained}")
    else:
        model = torchvision.models.densenet161(pretrained)
        if pretrained == 'True':
            print(f"Transfer Learning, Pretrained = {pretrained}")
            for param in model.parameters():
                param.requires_grad = requires_grad
            print(f"requires_grad = {requires_grad}")
            model.classifier = nn.Linear(2208, num_classes)"""
    else:
        # architecture == 'tldensenet161':
        use_pretrained = (pretrained == 'True')

        weights = DenseNet161_Weights.DEFAULT if use_pretrained else None
        model = densenet161(weights=weights)

        if use_pretrained:
            print(f"Transfer Learning, Pretrained = {pretrained}")
            for param in model.parameters():
                param.requires_grad = requires_grad
            print(f"requires_grad = {requires_grad}")

        # DenseNet161 final classifier input = 2208
        model.classifier = nn.Linear(2208, num_classes)
    return model
