import torch
import numpy as np
from tqdm import tqdm


class ASH:

    def __init__(self, model, device):
        self.model = model
        self.device = device

        '''
        Speical Parameters:
            T--Temperature
            p--Pruning Percentage
        '''
        self.T = 1
        self.p = 65

    def eval(self, data_loader):
        self.model.eval()
        result = []

        for (images, _) in tqdm(data_loader):
            images = images.to(self.device)
            feature = self.model.feature(images)

            # Three pruning method: ash_p, ash_b, ash_s
            output = ash_p(feature, self.p)
            # output = ash_b(feature, self.p)
            # output = ash_s(feature, self.p)

            output = output.view(output.size(0), -1)
            output = self.model.linear(output)

            output = self.T * torch.logsumexp(output / self.T, dim=1).data.cpu().numpy()

            result.append(output)

        return np.concatenate(result)



def ash_b(x, percentile):
    assert x.dim() == 4
    assert 0 <= percentile <= 100
    b, c, h, w = x.shape

    s1 = x.sum(dim=[1, 2, 3])

    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile / 100.0))
    t = x.view((b, c * h * w))
    v, i = torch.topk(t, k, dim=1)
    fill = s1 / k
    fill = fill.unsqueeze(dim=1).expand(v.shape)
    t.zero_().scatter_(dim=1, index=i, src=fill)
    return x


def ash_p(x, percentile):
    assert x.dim() == 4
    assert 0 <= percentile <= 100

    b, c, h, w = x.shape

    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile / 100.0))
    t = x.view((b, c * h * w))
    v, i = torch.topk(t, k, dim=1)
    t.zero_().scatter_(dim=1, index=i, src=v)

    return x


def ash_s(x, percentile):
    assert x.dim() == 4
    assert 0 <= percentile <= 100
    b, c, h, w = x.shape

    s1 = x.sum(dim=[1, 2, 3])
    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile / 100.0))
    t = x.view((b, c * h * w))
    v, i = torch.topk(t, k, dim=1)
    t.zero_().scatter_(dim=1, index=i, src=v)

    s2 = x.sum(dim=[1, 2, 3])

    scale = s1 / s2
    x = x * torch.exp(scale[:, None, None, None])

    return x