import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm


class ODIN:

    def __init__(self, model, device):
        self.model = model
        self.device = device

        '''
        Special Parameters:
            T--Temperature
            epsilon--Perturbation Magnitude
        '''
        self.T = 1000
        self.epsilon = 0.005

    def inputPreprocessing(self, images):
        outputs = self.model(images)
        criterion = nn.CrossEntropyLoss()

        maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)

        # Using temperature scaling
        outputs = outputs / self.T

        labels = Variable(torch.LongTensor(maxIndexTemp).cuda())
        loss = criterion(outputs, labels)
        loss.backward()

        # Normalizing the gradient to binary in {0, 1}
        gradient = torch.ge(images.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2

        # Adding small perturbations to images
        tempInputs = torch.add(images.data, gradient, alpha=-self.epsilon)
        with torch.no_grad():
            outputs = self.model(Variable(tempInputs))
        outputs = outputs / self.T

        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
        nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)

        return nnOutputs


    def eval(self, data_loader):
        self.model.eval()
        result = []

        for (images, _) in tqdm(data_loader):
            images = Variable(images.to(self.device), requires_grad=True)
            score = self.inputPreprocessing(images)
            score = np.max(score, 1)

            result.append(score)

        return np.concatenate(result)
