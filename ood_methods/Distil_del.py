from torch import nn
from torch.autograd import Variable
import numpy as np
import torch
from torchvision.transforms import transforms
from tqdm import tqdm
import torch.nn.functional as F
from matplotlib import pyplot as plt


def distil_eval(model, data_loader, args, device):

    result = None

    for (images, labels) in tqdm(data_loader):
        images = images.to(device)
        distil_score = DISTIL(images, model, args, device)

        if result is None:
            result = distil_score
        else:
            result = np.append(result, distil_score)

    return result


def DISTIL(origin_data, model, args, device):
    score = [0 for i in range(origin_data.size()[0])]

    # outputs = model(origin_data)
    criterion = nn.CrossEntropyLoss()

    for c in range(args.num_classes):
        noise = torch.zeros(origin_data.size()[0], 3, 32, 32).to(device)
        noise = nn.Parameter(noise, requires_grad=True)
        optimizer = torch.optim.Adam([noise], lr=0.01)

        targets = torch.ones(origin_data.shape[0]) * c
        targets = targets.type(torch.long).to(device)

        for iter in range(50):
            optimizer.zero_grad()
            output = model.forward_noise(origin_data, noise)
            loss = criterion(output, targets)
            loss.backward()
            print(loss)
            optimizer.step()

        norm = torch.norm(noise, dim=[1,2,3], p=1).detach().cpu().numpy()
        score = [-max(x, y) for x, y in zip(score, norm)]
        # print(torch.norm(noise, dim=[1,2,3], p=1))

    print(score)
    return np.array(score)


def kl_divergence(p, q):
    epsilon = 1e-8
    p = p + epsilon
    q = q + epsilon

    kl_div_batch = torch.sum(p * torch.log(p / q), dim=1)
    kl_loss = torch.sum(kl_div_batch, dim=0)

    return kl_div_batch, kl_loss


class Normalize:
    def __init__(self, opt, expected_values, variance):
        self.n_channels = 3
        self.expected_values = expected_values
        self.variance = variance
        assert self.n_channels == len(self.expected_values)

    def __call__(self, x):
        x_clone = x.clone()
        for channel in range(self.n_channels):
            x_clone[:, channel] = (x[:, channel] - self.expected_values[channel]) / self.variance[channel]
        return x_clone
    

class Denormalize:
    def __init__(self, expected_values, variance):
        self.n_channels = 3
        self.expected_values = expected_values
        self.variance = variance
        assert self.n_channels == len(self.expected_values)

    def __call__(self, x):
        x_clone = x.clone()
        for channel in range(self.n_channels):
            x_clone[:, channel] = x[:, channel] * self.variance[channel] + self.expected_values[channel]
        return x_clone
