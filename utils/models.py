import torch
from models.ResNet import *
from models.DenseNet import *


def get_model(args):
    model = None

    if args.model == "resnet":
        model = resnet18(num_classes=args.num_classes)

    elif args.model == "DenseNet":
        model = DenseNet(growthRate=12, depth=100, reduction=0.5, bottleneck=True, nClasses=args.num_classes)

    return model

