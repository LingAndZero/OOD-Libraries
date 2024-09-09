import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from pytorch_grad_cam import EigenCAM, GradCAM
import cv2
from pytorch_grad_cam.utils.image import (
    show_cam_on_image, deprocess_image, preprocess_image
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


class CAM:

    def __init__(self, model, device):
        self.model = model
        self.device = device

        '''
        Special Parameters:
            T--Temperature
        '''
        self.T = 1

    def eval(self, data_loader):
        self.model.eval()
        
        result = []

        for (images, _) in tqdm(data_loader):
            images = images.to(self.device)
            output = self.model(images)
            p_labels = output.argmax(1)

            batch_targets = []
            p_labels = p_labels.data.cpu().numpy().tolist()
            for i in range(images.shape[0]):
                batch_targets.append(ClassifierOutputTarget(p_labels[i]))

            # get CAM
            with GradCAM(model=self.model, target_layers=[self.model.layer4]) as cam:
                cam.batch_size = images.size()[0]
                grayscale_cam = cam(input_tensor=images, targets=batch_targets, aug_smooth=False, eigen_smooth=False)
            
                grayscale_cam = torch.from_numpy(grayscale_cam).to(self.device)
                # grayscale_cam = grayscale_cam[0, :]
                
                # denormalizer = Denormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                # rgb_img = denormalizer(images)[0]
                # rgb_img = rgb_img.permute(1, 2, 0).cpu().numpy()
                # cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
                # cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
                # cv2.imwrite('a.png', cam_image)

            mask = (torch.tensor((grayscale_cam)) * -0.5 + 1).unsqueeze(1).to(self.device)
            # output = self.model(images)
            output = self.model(images * mask)

            output = self.T * torch.logsumexp(output / self.T, dim=1).data.cpu().numpy()

            result.append(output)
            # break

        print(result)
        return np.concatenate(result)




class Denormalize:
    def __init__(self, expected_values, variance):
        self.n_channels = 3
        self.expected_values = expected_values
        self.variance = variance
        assert self.n_channels == len(self.expected_values)

    def __call__(self, x):
        x_clone = x.clone()
        for channel in range(self.n_channels):
            x_clone[:, channel] = x[:, channel] * self.variance[channel] + self.expected_values[channel]
        return x_clone
