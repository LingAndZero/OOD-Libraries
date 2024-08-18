import torch
from models.ResNet import *
from models.DenseNet import *
from models.WideResNet import *


def get_model(args, pretrain=False):
    model = None

    if args.model == "ResNet":
        if args.ind_dataset in ['cifar10', 'cifar100']:
            model = resnet18(num_classes=args.num_classes)
        else:
            model = resnet50(num_classes=args.num_classes, pretrained=True)

    elif args.model == "DenseNet":
        model = DenseNet(growthRate=12, depth=100, reduction=0.5, bottleneck=True, nClasses=args.num_classes)

    elif args.model == "WideResNet":
        model = WideResNet(depth=28, num_classes=args.num_classes, widen_factor=1, droprate=0.0)

    if pretrain:
        filename = '/home/zwling/code/OOD-Libraries/checkpoints/' + args.ind_dataset + '-' + args.model + '-0-100.pt'
        model.load_state_dict(torch.load(filename))

    return model

