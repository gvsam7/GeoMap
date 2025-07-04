import argparse


# Hyperparameters
def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=102)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=21)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--save-model", default=False)
    parser.add_argument("--load-model", default=False)
    parser.add_argument("--augmentation", default="cutout", help="cutout, mixup, cutmix")
    parser.add_argument("--Augmentation", default="none", help="none, position, cutout")
    parser.add_argument("--pretrained", default=True)
    parser.add_argument("--requires-grad", default=False)
    parser.add_argument("--global-pooling", default=None)
    parser.add_argument("--class-weighting", default=False)
    parser.add_argument("--dataset", default="b10", help="b6, b7, b10, b11, b76, TreeCrown512, TreeCrown256, "
                                                         "TreeCrown128")
    parser.add_argument("--version", default="b0", help="b1:b7")
    parser.add_argument("--architecture", default="cnn_5", help="cnn4=CNN4, cnn5=CNN5, cnn6=CNN6, cnn7=CNN7,"
                                                                "resnet18=ResNet18, dilgabmpresnet18=DilGabMPResNet18,"
                                                                "dilgabmpresnet50=DilGabMPResNet50,"
                                                                "tlresnet18=pretrain ResNet18,"
                                                                "resnet50=ResNet50, tlresnet50=pretrain ResNet50,"
                                                                "vgg13=VGG13, tlvgg13=pretrain VGG13,"
                                                                "effivientnet=EfficientNet,"
                                                                "tldensenet161=pretrain DenseNet161,"
                                                                "fusionnet=FusionNet, resfusionnet=ResFusionNet, "
                                                                "vit=ViT")

    return parser.parse_args()
