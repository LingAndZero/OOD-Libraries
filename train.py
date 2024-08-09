import os
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm

from utils.utils import fix_random_seed
from utils.dataset import get_dataset
from utils.models import get_model


def get_train_options():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--model", type=str, default="DenseNet")
    parser.add_argument("--gpu", type=int, default=0)

    parser.add_argument("--random_seed", type=int, default=2)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--bs", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument('--num_classes', type=int, default=10)

    args = parser.parse_args()
    return args


def test(model, test_loader, device):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
        
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Accuracy: {accuracy * 100:.2f}%')


if __name__ == '__main__':
    args = get_train_options()
    fix_random_seed(args.random_seed)

    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    train_dataset, test_dataset = get_dataset(args.dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.bs, shuffle=False, num_workers=2)

    model = get_model(args).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    for epoch in range(args.epoch):
        model.train()
        print("epoch: {}".format(epoch))
        for batch_idx, (data, labels) in tqdm(enumerate(train_loader)):
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

        test(model, test_loader, device)
        scheduler.step()

    filename = 'checkpoints/{}-{}-{}.pt'.format(args.dataset, args.model, args.random_seed)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(model.state_dict(), filename)