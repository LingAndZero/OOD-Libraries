import torch
import numpy as np
from tqdm import tqdm


class DICE:

    def __init__(self, model, device):
        self.model = model
        self.device = device

        '''
        Special Parameters:
            T--Temperature
            p--Sparsity Parameter
        '''
        self.T = 1
        self.p = 70

    def get_mask(self, train_loader, num_classes):
        self.model.eval()
        linear_weights = self.model.linear.weight.data
        result = [[] for i in range(num_classes)]

        with torch.no_grad():
            for (images, _) in tqdm(train_loader):
                images = images.cuda()
                output = self.model.feature(images)
                output = output.view(output.size(0), -1)
                
                for c in range(num_classes):
                    result[c].append((output * linear_weights[c]).cpu())
        
        for c in range(num_classes):
            result[c] = np.mean(np.concatenate(result[c]), axis=0)

        threshold = np.percentile(np.array(result).flatten(), self.p)
        mask = result > threshold
        mask = torch.tensor(mask).to(self.device)
        self.model.linear.weight.data *= mask

    def eval(self, data_loader):
        self.model.eval()
        
        result = []

        for (images, _) in tqdm(data_loader):
            images = images.to(self.device)
            output = self.model(images)
        
            output = self.T * torch.logsumexp(output / self.T, dim=1).data.cpu().numpy()

            result.append(output)

        return np.concatenate(result)
