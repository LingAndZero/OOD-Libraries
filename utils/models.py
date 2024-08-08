import torch
from models.resnet import *


def get_model(args):
    model = None

    if args.model == "resnet":
        model = resnet18(num_classes=args.num_classes)

    return model

