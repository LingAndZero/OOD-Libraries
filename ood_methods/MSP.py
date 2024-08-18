import numpy as np
import torch.nn.functional as F
from tqdm import tqdm


def msp_eval(model, data_loader):
    model.eval()
    result = None

    for (images, _) in tqdm(data_loader):
        images = images.cuda()
        output = model(images)

        smax = (F.softmax(output, dim=1)).data.cpu().numpy()
        output = np.max(smax, axis=1)

        if result is None:
            result = output
        else:
            result = np.append(result, output)

    return result
