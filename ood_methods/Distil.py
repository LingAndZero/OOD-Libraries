from torch import nn
from torch.autograd import Variable
import numpy as np
import torch
from torchvision.transforms import transforms
from tqdm import tqdm
import torch.nn.functional as F


def distil_eval(model, data_loader, logits, args):
    result = None

    for (images, labels) in tqdm(data_loader):
        images = images.cuda()
        distil_score = DISTIL(images, model, logits, args)
        if result is None:
            result = distil_score
        else:
            result = np.append(result, distil_score)

    return result


def DISTIL(origin_data, model, logits, args):
    score = []

    # criterion = nn.MSELoss()
    criterion = nn.KLDivLoss(reduction='sum')
    noise = nn.Parameter(torch.zeros(origin_data.size()[0], 3, 224, 224).cuda().requires_grad_())
    optimizer = torch.optim.Adam([noise], lr=0.01)
    outputs = model(origin_data)
    p_labels = outputs.argmax(1)
    p_labels = p_labels.data.cpu()

    base_logits = None
    for i in range(origin_data.shape[0]):

        if base_logits is None:
            base_logits = logits[p_labels[i]].unsqueeze(0)
        else:
            base_logits = torch.cat((base_logits, logits[p_labels[i]].unsqueeze(0)), dim=0)
    
    base_logits = base_logits.float().cuda()
    for iters in range(10):
        optimizer.zero_grad()
        tmp_pred = model.forward_noise(origin_data, noise)

        total_loss = criterion(F.log_softmax(tmp_pred / 100, dim=1), F.softmax(base_logits / 100, dim=1))
        # print(total_loss)
        # total_loss = criterion(tmp_pred, base_logits)
        total_loss.backward()
        optimizer.step()

    noise = noise.detach()
    noise = pruning_p(noise)
    
    for idx in range(origin_data.shape[0]):
        print(torch.norm(noise[idx], p=2).cpu())
        score.append(-torch.norm(noise[idx], p=2).cpu())

    return np.array(score)


def get_logits(model, data_loader, args, mode="train"):
    model.eval()
    result = [[] for i in range(args.num_classes)]

    with torch.no_grad():
        for (images, labels) in tqdm(data_loader):
            images, labels = images.cuda(), labels.cuda()
            output = model(images)

            if mode == "train":
                output = output.data.cpu().numpy()
                for i in range(labels.size(0)):
                    result[labels[i]].append(output[i])
            else:
                p_labels = output.argmax(1)
                p_labels = p_labels.data.cpu().numpy()
                output = output.data.cpu().numpy()
                for i in range(labels.size(0)):
                    result[p_labels[i]].append(output[i])

    logits = []
    for i in range(args.num_classes):
        tmp = np.mean(result[i], axis=0)
        logits.append(tmp)
    
    return np.array(logits)


def pruning_p(x, percentile=50):
    b, c, h, w = x.shape

    s1 = x.sum(dim=[1, 2, 3])

    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile / 100.0))
    t = x.view((b, c * h * w))
    v, i = torch.topk(t, k, dim=1)
    t.zero_().scatter_(dim=1, index=i, src=v)
    return x


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
