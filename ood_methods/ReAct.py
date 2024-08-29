import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm


def react_eval(model, data_loader, threshold, device):
    model.eval()
    result = None

    for (images, _) in tqdm(data_loader):
        images = images.to(device)
        output = model.forward_threshold(images, threshold)

        output = 1 * torch.logsumexp(output / 1, dim=1).data.cpu().numpy()

        if result is None:
            result = output
        else:
            result = np.append(result, output)

    return result


def get_threshold(model, train_loader, p):
    model.eval()
    result = None

    with torch.no_grad():
        for (images, _) in tqdm(train_loader):
            images = images.cuda()
            output = model.compute_threshold(images).data.cpu().numpy()

            if result is None:
                result = output
            else:
                result = np.append(result, output)

    threshold = np.percentile(result, p)

    return threshold
