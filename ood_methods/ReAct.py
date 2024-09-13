import torch
import numpy as np
from tqdm import tqdm


class ReAct:

    def __init__(self, model, device):
        self.model = model
        self.device = device

        '''
        Special Parameters:
            T--Temperature
            p--Truncation Percentage
        '''
        self.T = 1
        self.p = 90

    def get_threshold(self, train_loader):
        self.model.eval()
        result = []

        with torch.no_grad():
            for (images, _) in tqdm(train_loader):
                images = images.cuda()
                output = self.model.feature(images).data.cpu().numpy()

                result.append(output)

        threshold = np.percentile(np.concatenate(result), self.p)

        return threshold

    def eval(self, data_loader, threshold):
        self.model.eval()
        result = []

        for (images, _) in tqdm(data_loader):
            images = images.to(self.device)
            output = self.model.feature(images)
            output = output.view(output.size(0), -1)
            output = output.clip(max=threshold)
            output = self.model.linear(output)

            output = self.T * torch.logsumexp(output / self.T, dim=1).data.cpu().numpy()

            result.append(output)

        return np.concatenate(result)
