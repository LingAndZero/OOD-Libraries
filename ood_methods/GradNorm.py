from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torch
from tqdm import tqdm


def gradnorm_eval(model, data_loader, args, device):
    model.eval()
    result = None
    logsoftmax = torch.nn.LogSoftmax(dim=-1).cuda()

    for (images, _) in tqdm(data_loader):
        images = Variable(images.to(device), requires_grad=True)

        model.zero_grad()
        outputs = model(images)
        targets = torch.ones((images.shape[0], args.num_classes)).to(device)
        outputs = outputs / 1

        loss = torch.mean(torch.sum(-targets * logsoftmax(outputs), dim=-1))
        loss.backward()

        fc_layer_grad = model.linear.weight.grad.data
        score = torch.sum(torch.abs(fc_layer_grad)).cpu().numpy()

        if result is None:
            result = score
        else:
            result = np.append(result, score)

    return result
