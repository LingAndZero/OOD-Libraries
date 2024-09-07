import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm


class Energy:

    def __init__(self, model, device):
        self.model = model
        self.device = device

        # Speical Parameters
        # temperature
        self.T = 1

    def eval(self, data_loader):
        self.model.eval()
        result = []

        for (images, _) in tqdm(data_loader):
            images = images.to(self.device)
            output = self.model(images)

            output = self.T * torch.logsumexp(output / self.T, dim=1).data.cpu().numpy()

            result.append(output)

        return np.concatenate(result)
