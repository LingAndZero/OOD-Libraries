from torchvision import transforms


def get_dataset(dataset):

    train_dataset = None
    test_dataset = None

    if dataset == "cifar10":
        from torchvision.datasets import CIFAR10
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        ])
        train_dataset = CIFAR10("./data/cifar10", train=True, transform=train_transform, download=True)
        test_dataset = CIFAR10("./data/cifar10", train=False, transform=test_transform, download=True)

    elif dataset == "cifar100":
        from torchvision.datasets import CIFAR100
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        ])
        train_dataset = CIFAR100("./data/cifar100", train=True, transform=train_transform, download=True)
        test_dataset = CIFAR100("./data/cifar100", train=False, transform=test_transform, download=True)
    
    return train_dataset, test_dataset