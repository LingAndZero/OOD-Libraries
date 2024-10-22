import torch
import numpy as np
from tqdm import tqdm


class OptFS:

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.left_boundary = None
        self.width = None
        '''
        Special Parameters:
            T--Temperature
            p--Pruning Percentage
        '''
        self.T = 1
    
    def get_features(self, data_loader, num_classes):
        self.model.eval()
        feature_train = []
        pred_train = []

        with torch.no_grad():
            for (images, labels) in tqdm(data_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                output, feature = self.model.feature(images)
                score = torch.softmax(output, dim=1)
                _, pred = torch.max(score, dim=1)

                feature_train.append(feature.cpu().numpy())
                pred_train.append(pred.cpu().numpy())
    
        feature_train = np.concatenate(feature_train, axis=0)
        pred_train = np.concatenate(pred_train, axis=0)

        return feature_train, pred_train
    
    @torch.no_grad()
    def get_optimal_shaping(self, features, preds):

        w = self.model.classifier.weight
        w = w.detach().cpu().numpy()

        left_b = np.quantile(features, 1e-3)
        right_b = np.quantile(features, 1-1e-3)
        
        self.width = (right_b - left_b) / 100.0
        self.left_boundary = np.arange(left_b, right_b, self.width)
        
        lc = w[preds] * features
        lc_fv_list = []
        for b in tqdm(self.left_boundary):
            mask = (features >= b) & (features < b + self.width)
            feat_masked = mask * lc
            res = np.mean(np.sum(feat_masked, axis=1))
            lc_fv_list.append(res)
        lc_fv_list = np.array(lc_fv_list)
        theta = lc_fv_list / np.linalg.norm(lc_fv_list, 2) * 1000

        return torch.from_numpy(theta[np.newaxis, :])

    def eval(self, data_loader, theta):
        self.model.eval()
        result = []

        with torch.no_grad():
            for (images, _) in tqdm(data_loader):
                images = images.to(self.device)
                _, feature = self.model.feature(images)
                feature = feature.view(feature.size(0), -1)

                feat_p = torch.zeros_like(feature).to(self.device)
                for i, x in enumerate(self.left_boundary):
                    mask = (feature >= x) & (feature < x + self.width)
                    feat_p += mask * feature * theta[0][i]

                output = self.model.classifier(feat_p)
                output = self.T * torch.logsumexp(output / self.T, dim=1).data.cpu().numpy()

                result.append(output)

        return np.concatenate(result)
