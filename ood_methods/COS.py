import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn


class COS:

    def __init__(self, model, device):
        self.model = model
        self.device = device

        '''
        Special Parameters:
            T--Temperature
        '''
        self.T = 1
    

    def get_features(self, data_loader, num_classes):
        self.model.eval()
        result = [[] for _ in range(num_classes)]

        with torch.no_grad():
            for (images, labels) in tqdm(data_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                output = self.model.feature(images)
                output = output.view(output.size(0), -1)
                output = output.data.cpu().numpy()
                for i in range(labels.size(0)):
                    result[labels[i]].append(output[i])
    
        features = []
        for i in range(num_classes):
            tmp = np.mean(result[i], axis=0)
            features.append(tmp)
        
        return np.array(features)
    
    def get_noise(self, images, p_labels, features):
        cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)

        mean_features = features.mean(0, keepdim=True).to(self.device)
        base_features = None

        base_features = features[p_labels].float().to(self.device)
        
        noise = torch.zeros_like(images).to(self.device)
        noise = nn.Parameter(noise, requires_grad=True)
        optimizer = torch.optim.Adam([noise], lr=0.01)

        for iters in range(50):
            optimizer.zero_grad()

            tmp_pred = self.model.feature_noise(images, noise)

            loss1 = torch.sum(cos_sim(base_features, tmp_pred))
            loss2 = torch.sum(cos_sim(mean_features, tmp_pred))

            total_loss = loss1 - 0.01 * loss2
            print(total_loss)
            total_loss.backward()
            optimizer.step()

        return noise.detach().cpu()
    

    def eval(self, data_loader, features):
        self.model.eval()
        result = [] 
        cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)

        with torch.no_grad():
            for (images, _) in tqdm(data_loader):
                images = images.to(self.device)
                feature = self.model.feature(images)
                feature = feature.view(feature.size(0), -1)
                output = self.model.fc(feature)
                p_labels = output.argmax(1).data.cpu()
                
                base_feature = features[p_labels].to(self.device)
                h = cos_sim(feature, base_feature)

                result.append(h.cpu())
        print(result)
        return np.concatenate(result)


def kl_divergence(p, q):
    epsilon = 1e-8
    p = p + epsilon
    q = q + epsilon

    kl_div_batch = torch.sum(p * torch.log(p / q), dim=1)
    kl_loss = torch.sum(kl_div_batch, dim=0)

    return kl_div_batch, kl_loss