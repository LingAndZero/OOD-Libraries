import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm


def ash_eval(model, data_loader, device):
    model.eval()
    result = None

    for (images, _) in tqdm(data_loader):
        images = images.to(device)
        output = model.ash_forward(images)

        output = 1 * torch.logsumexp(output / 1, dim=1).data.cpu().numpy()

        if result is None:
            result = output
        else:
            result = np.append(result, output)

    return result
