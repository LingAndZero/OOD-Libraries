import torch
from models.ResNet import *
from models.DenseNet import *
from models.WideResNet import *


def get_model(args):
    model = None

    if args.model == "ResNet":
        model = resnet18(num_classes=args.num_classes)

    elif args.model == "DenseNet":
        model = DenseNet(growthRate=12, depth=100, reduction=0.5, bottleneck=True, nClasses=args.num_classes)

    elif args.model == "WideResNet":
        model = WideResNet(depth=28, num_classes=args.num_classes, widen_factor=1, droprate=0.0)

    return model

