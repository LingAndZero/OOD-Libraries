import numpy as np
import torch.nn.functional as F
from tqdm import tqdm


class MSP:

    def __init__(self, model, device):
        self.model = model
        self.device = device

    def eval(self, data_loader):
        self.model.eval()
        result = []

        for (images, _) in tqdm(data_loader):
            images = images.to(self.device)
            output = self.model(images)

            smax = (F.softmax(output, dim=1)).data.cpu().numpy()
            output = np.max(smax, axis=1)

            result.append(output)

        return np.concatenate(result)
