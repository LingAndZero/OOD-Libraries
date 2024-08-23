from torchvision import transforms
import torch.nn.functional as F
import torch
import numpy as np
from tqdm import tqdm


def mutation_eval(model, data_loader):
    model.eval()
    result = None

    for (images, _) in tqdm(data_loader):
        images = images.cuda()
        out_base = model(images)
        a = 1 * torch.logsumexp(out_base / 1, dim=1).data.cpu().numpy()

        t1 = transforms.RandomHorizontalFlip(p=1)
        t2 = transforms.RandomVerticalFlip(p=1)
    
        out_t1 = model(t1(images))
        out_t2 = model(t2(images))

        b = 1 * torch.logsumexp(out_t1 / 1, dim=1).data.cpu().numpy()
        c = 1 * torch.logsumexp(out_t2 / 1, dim=1).data.cpu().numpy()

        score = b + c - 2 * a
        # score += np.abs(a - d)
        # score += np.abs(a - e)
        # score += F.kl_div(out_base.log(), out_t3, reduction='none').sum(dim=-1)
        # score += F.kl_div(out_base.log(), out_t4, reduction='none').sum(dim=-1)

        if result is None:
            result = -score
        else:
            result = np.append(result, -score)

    return result
