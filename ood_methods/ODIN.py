import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm


def odin_eval(model, data_loader, device):
    model.eval()
    output = None

    for (images, _) in tqdm(data_loader):
        images = Variable(images.to(device), requires_grad=True)
        a = model(images)
        odin_score = ODIN(images, a, model, 1000.0, 0.005)
        result = np.max(odin_score, 1)

        if output is None:
            output = result
        else:
            output = np.append(output, result)

    return output


def ODIN(inputs, outputs, model, temper, noiseMagnitude1):
    criterion = nn.CrossEntropyLoss()

    maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)

    # Using temperature scaling
    outputs = outputs / temper

    labels = Variable(torch.LongTensor(maxIndexTemp).cuda())
    loss = criterion(outputs, labels)
    loss.backward()

    # Normalizing the gradient to binary in {0, 1}
    gradient = torch.ge(inputs.grad.data, 0)
    gradient = (gradient.float() - 0.5) * 2

    # Adding small perturbations to images
    tempInputs = torch.add(inputs.data, gradient, alpha=-noiseMagnitude1)
    with torch.no_grad():
        outputs = model(Variable(tempInputs))
    outputs = outputs / temper

    # Calculating the confidence after adding perturbations
    nnOutputs = outputs.data.cpu()
    nnOutputs = nnOutputs.numpy()
    nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
    nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)

    return nnOutputs
