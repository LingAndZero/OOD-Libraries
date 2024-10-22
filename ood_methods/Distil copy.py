from torch import nn
from torch.autograd import Variable
import numpy as np
import torch
from torchvision.transforms import transforms
from tqdm import tqdm
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM, EigenCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


class Distil:

    def __init__(self, model, device):
        self.model = model
        self.device = device

        '''
        Special Parameters:
            T--Temperature
        '''
        self.T = 100


    def get_logits(self, data_loader, num_classes):
        self.model.eval()
        result = [[] for _ in range(num_classes)]

        with torch.no_grad():
            for (images, labels) in tqdm(data_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                output = self.model(images)

                output = output.data.cpu().numpy()
                for i in range(labels.size(0)):
                    result[labels[i]].append(output[i])
    
        logits = []
        for i in range(num_classes):
            tmp = np.mean(result[i], axis=0)
            logits.append(tmp)
        
        return np.array(logits)


    def get_noise(self, images, p_labels, logits):

        mean_logits = logits.mean(0, keepdim=True).to(self.device)

        base_logits = logits[p_labels]
        base_logits = base_logits.float().to(self.device)
        
        noise = torch.zeros_like(images).to(self.device)
        noise = nn.Parameter(noise, requires_grad=True)
        optimizer = torch.optim.Adam([noise], lr=0.01)

        for iters in range(50):
            optimizer.zero_grad()

            tmp_pred = self.model.forward_noise(images, noise)

            _, loss1 = kl_divergence(F.softmax(base_logits / self.T, dim=1), F.softmax(tmp_pred / self.T, dim=1))
            # _, loss2 = kl_divergence(F.softmax(mean_logits / self.T, dim=1), F.softmax(tmp_pred / self.T, dim=1))

            flip = transforms.RandomHorizontalFlip(p=1)
            tmp_pred = self.model.forward_noise(flip(images), flip(noise))

            _, loss3 = kl_divergence(F.softmax(base_logits / self.T, dim=1), F.softmax(tmp_pred / self.T, dim=1))
            # _, loss4 = kl_divergence(F.softmax(mean_logits / self.T, dim=1), F.softmax(tmp_pred / self.T, dim=1))

            # total_loss = loss1 + loss3 - 0.01 * loss2 - 0.01 * loss4
            total_loss = loss1  + loss3 
            print(total_loss)
            total_loss.backward()
            optimizer.step()

        return noise.detach().cpu()


    def get_cam(self, images, p_labels):
        batch_targets = []
        p_labels = p_labels.numpy().tolist()
        for i in range(images.shape[0]):
            batch_targets.append(ClassifierOutputTarget(p_labels[i]))

        with GradCAM(model=self.model, target_layers=[self.model.layer4]) as cam:
            cam.batch_size = images.size()[0]
            grayscale_cam = cam(input_tensor=images, targets=batch_targets)
        
        grayscale_cam = torch.from_numpy(grayscale_cam).to(self.device)
        grayscale_cam = grayscale_cam.unsqueeze(1).expand(-1, 3, -1, -1).cpu() 
        return grayscale_cam


    def get_ood_score(self, outputs):
        return 1 * torch.logsumexp(outputs / 1, dim=1).data.cpu()


    def eval(self, data_loader, logits):
        self.model.eval()
        result = []

        for (images, _) in tqdm(data_loader):
            images = images.to(self.device)
            outputs = self.model(images)
            p_labels = outputs.argmax(1).data.cpu()

            noise = self.get_noise(images, p_labels, logits)
            pooled_tensor = F.max_pool2d(noise, kernel_size=7)
            np.save('./result/noise/id_imagenet.npy', pooled_tensor.numpy())

            cam = self.get_cam(images + noise.to(self.device), p_labels)
            o_score = self.get_ood_score(outputs)
            break
            noise = (1 + cam) * noise

            score = []
            for idx in range(images.shape[0]):
                score.append(-(torch.norm(noise[idx] / o_score[idx], p=2)).cpu())

            print(score)
            result.append(score)

        return np.concatenate(result)


def kl_divergence(p, q):
    epsilon = 1e-8
    p = p + epsilon
    q = q + epsilon

    kl_div_batch = torch.sum(p * torch.log(p / q), dim=1)
    kl_loss = torch.sum(kl_div_batch, dim=0)

    return kl_div_batch, kl_loss