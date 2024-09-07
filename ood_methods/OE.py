import tqdm
import numpy as np
import torch.nn.functional as F


class OE:
    def __init__(self, ind_dataset, ood_dataset):
        self.model = None
        self.ind_dataset = ind_dataset
        self.ood_dataset = ood_dataset

    def add_arguments(parser):
        pass

    def retrain():
        pass

    def eval(self):
        self.model.eval()
        result = None

        for (images, _) in tqdm(self.data_loader):
            images = images.to(self.device)
            output = self.model(images)

            smax = (F.softmax(output, dim=1)).data.cpu().numpy()
            output = np.max(smax, axis=1)

            if result is None:
                result = output
            else:
                result = np.append(result, output)

        return result
