import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn


class DIST:

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.width = []
        self.left_boundary = []
    
    def get_features(self, data_loader, num_classes):
        self.model.eval()
        feature_train = [[] for _ in range(num_classes)]

        with torch.no_grad():
            for (images, labels) in tqdm(data_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                output, feature = self.model.feature(images)
                p_labels = output.argmax(1)

                for i in range(labels.size(0)):
                    feature_train[p_labels[i]].append(feature[i].cpu().numpy())

        for i in range(num_classes):
            feature_train[i] = np.concatenate(feature_train[i], axis=0)

        return feature_train

    @torch.no_grad()
    def get_optimal_shaping(self, features):
        w = self.model.linear.weight
        w = w.detach().cpu().numpy()

        for i in range(10):
            left_b = np.quantile(features[i], 1e-3)
            right_b = np.quantile(features[i], 1-1e-3)
        
            width = (right_b - left_b) / 100.0
            left_boundary = np.arange(left_b, right_b, self.width)
            self.width.append(width)
            self.left_boundary.append(left_boundary)

            lc = w[i] * features
            lc_fv_list = []

            for b in tqdm(left_boundary):
                mask = (features >= b) & (features < b + width)
                feat_masked = mask * lc
                res = np.mean(np.sum(feat_masked, axis=1))
                lc_fv_list.append(res)
            lc_fv_list = np.array(lc_fv_list)
            theta = lc_fv_list / np.linalg.norm(lc_fv_list, 2) * 1000

        return torch.from_numpy(theta[np.newaxis, :])
    
    def get_stats(features):
        mu = []
        sigma = []
        mean_feature = []
        
        for i in range(len(features)):
            mean_feature.append(torch.mean(features[i], dim=0))
            mu.append(features[i].mean())
            sigma.append(features[i].std())

        cv = [a / b for a, b in zip(sigma, mu)]
        
        return torch.tensor(cv), torch.stack(mean_feature)

    def eval(self, data_loader, mean_feature):
        self.model.eval()
        mean_feature = mean_feature.to(self.device)
        result = [] 

        with torch.no_grad():
            for (images, _) in tqdm(data_loader):
                images = images.to(self.device)
                output, feature = self.model.feature(images)
                p_labels = output.argmax(1)

                base_features = mean_feature[p_labels]
                error_rate = (base_features - feature).norm(dim=1, p=2) / (feature.norm(dim=1, p=1) * base_features.norm(dim=1, p=1))

                score = - error_rate
                print(score)
                result.append(score.cpu())

        return np.concatenate(result)