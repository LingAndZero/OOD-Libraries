import tqdm
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR


class OE:
    def __init__(self, model, device):
        self.model = model
        self.device = device

        self.epochs = 5
        self.samples = 512
        self.lr = 0.0001
        self.weight_decay = 5e-4

    def fineTuning(self, train_loader):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs)

        train_loader_out.dataset.offset = np.random.randint(len(train_loader_out.dataset))
        
        self.model.train()
        for epoch in tqdm(range(self.epochs)):
            for batch_idx, (data, labels) in enumerate(train_loader):
                data, labels = data.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)

                loss = criterion(output, labels)
                loss += 0.5 * -(output[len(in_set[0]):].mean(1) - torch.logsumexp(output[len(in_set[0]):], dim=1)).mean()
                loss.backward()

                optimizer.step()
            scheduler.step()
            

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