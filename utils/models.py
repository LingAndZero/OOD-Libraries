import torch
from torchvision import models
from models.ResNet_CIFAR import *
from models.ResNet import resnet50
from models.VGG_CIFAR import *
from models.VGG import vgg16
from models.MobileNetv2 import *
from models.DenseNet import *
from models.WideResNet import *


def get_model(args):
    model = None
    save_path = './checkpoints/' + args.ind_dataset + '-' + args.model + '-0'

    if args.model == "ResNet":
        if args.ind_dataset in ['cifar10', 'cifar100']:
            model = ResNet18()
            model.load_state_dict((torch.load(save_path + '/last.pth.tar')['state_dict']))
        else:
            model = resnet50(num_classes=args.num_classes, pretrained=True)

    elif args.model == "DenseNet":
        model = densenet_cifar()
        model.load_state_dict((torch.load(save_path + '/last.pth.tar')['state_dict']))

    elif args.model == "WideResNet":
        model = WideResNet(depth=28, num_classes=args.num_classes, widen_factor=10)
        model.load_state_dict((torch.load(save_path + '/last.pth.tar')['state_dict']))

    elif args.model == "VGG":
        if args.ind_dataset in ['cifar10', 'cifar100']:
            model = VGG_CIFAR('VGG11')
            model.load_state_dict((torch.load(save_path + '/last.pth.tar')['state_dict']))
        else:
            model = vgg16(pretrained=True)

    elif args.model == "MobileNetv2":
        model = MobileNetV2()
        model.load_state_dict((torch.load(save_path + '/last.pth.tar')['state_dict']))

    elif args.model == "VIT":
        model = models.vit_b_16(weights='IMAGENET1K_V1')

    elif args.model == "SWIN":
        model = models.swin_b(weights='IMAGENET1K_V1')

    elif args.model == "ConvNext":
        model = models.convnext_base(weights='IMAGENET1K_V1')

    return model

