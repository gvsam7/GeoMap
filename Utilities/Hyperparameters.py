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
    parser.add_argument("--augmentation", default="cutout", help="cutout, cutmix")
    parser.add_argument("--Augmentation", default="none", help="none, position, cutout")
    parser.add_argument("--dataset", help="b10")
    parser.add_argument("--architecture", default="cnn5", help="cnn5=CNN5, cnn_5=CNN_5, resnet18=ResNet18, "
                                                               "resnet50=ResNet50")

    return parser.parse_args()
