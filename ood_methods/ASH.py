import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm


def ash_eval(model, data_loader, device):
    model.eval()
    result = []

    for (images, _) in tqdm(data_loader):
        images = images.to(device)
        output = model.ash_forward(images)

        output = 1 * torch.logsumexp(output / 1, dim=1).data.cpu().numpy()

        result.append(output)

    return np.concatenate(result)


class ASH:

    def __init__(self, model, device):
        self.model = model
        self.device = device

        # Speical Parameters
        # temperature
        self.T = 1
        self.p = 0.65

    def eval(self, data_loader):
        self.model.eval()
        result = []

        for (images, _) in tqdm(data_loader):
            images = images.to(self.device)
            output = self.model.ash_forward(images)

            output = self.T * torch.logsumexp(output / self.T, dim=1).data.cpu().numpy()

            result.append(output)

        return np.concatenate(result)
