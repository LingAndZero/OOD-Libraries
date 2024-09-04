from torch import nn
from torch.autograd import Variable
import numpy as np
import torch
from torchvision.transforms import transforms
from tqdm import tqdm
import torch.nn.functional as F
from copy import deepcopy
# import wandb
# from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad

# wandb.init(
#     project="ood-loss",

#     config={
#     "T": 100,
#     "lr": 0.01,
#     "epoch": 10,
#     "architecture": "ResNet",
#     "dataset": "cifar10",
#     }
# )

def distil_eval(model, data_loader, logits, args, device):

    result = None

    for (images, labels) in tqdm(data_loader):
        images = images.to(device)
        distil_score = DISTIL(images, model, logits, args, device)

        # break
        if result is None:
            result = distil_score
        else:
            result = np.append(result, distil_score)

    return result


def DISTIL(origin_data, model, logits, args, device):
    score = []

    outputs = model(origin_data)
    p_labels = outputs.argmax(1)
    p_labels = p_labels.data.cpu()

    # # get CAM
    # with EigenCAM(model=model, target_layers=[model.layer4]) as cam:
    #     cam.batch_size = origin_data.size()[0]
    #     grayscale_cam = cam(input_tensor=origin_data, targets=p_labels, aug_smooth=False, eigen_smooth=False)
    
    # grayscale_cam = torch.from_numpy(grayscale_cam).to(device)
    # grayscale_cam = grayscale_cam.unsqueeze(1).expand(-1, 3, -1, -1).cpu()

    # get base logits
    base_logits = None
    for i in range(origin_data.shape[0]):

        if base_logits is None:
            base_logits = logits[p_labels[i]].unsqueeze(0)
        else:
            base_logits = torch.cat((base_logits, logits[p_labels[i]].unsqueeze(0)), dim=0)

    base_logits = base_logits.float().to(device)

    noise = torch.zeros(origin_data.size()[0], 3, 224, 224).to(device)
    noise = nn.Parameter(noise, requires_grad=True)
    optimizer = torch.optim.Adam([noise], lr=0.01)
    
    for iters in range(100):
        optimizer.zero_grad()

        tmp_pred = model.forward_noise(origin_data, noise)
        loss_batch, total_loss = kl_divergence(F.softmax(base_logits / args.T, dim=1), F.softmax(tmp_pred / args.T, dim=1))
        

        flip = transforms.RandomHorizontalFlip(p=1)
        tmp_pred = model.forward_noise(flip(origin_data), flip(noise))
        loss_b, loss = kl_divergence(F.softmax(base_logits / args.T, dim=1), F.softmax(tmp_pred / args.T, dim=1))
        
        total_loss += loss
        loss_batch += loss_b

        total_loss.backward()
        print(total_loss)

        optimizer.step()

    noise = noise.detach().cpu()
    loss_batch = loss_batch.detach().cpu()
    # noise = 0.5 * noise + 0.5 * grayscale_cam * noise

    for idx in range(origin_data.shape[0]):
        print((loss_batch[idx] * torch.norm(noise[idx], p=2) * 1e4).cpu())
        score.append(-(loss_batch[idx] * torch.norm(noise[idx], p=2)).cpu())

    return np.array(score)


def get_logits(model, data_loader, args, device, mode="train"):
    model.eval()
    result = [[] for i in range(args.num_classes)]

    with torch.no_grad():
        for (images, labels) in tqdm(data_loader):
            images, labels = images.to(device), labels.to(device)
            output = model(images)

            if mode == "train":
                output = output.data.cpu().numpy()
                for i in range(labels.size(0)):
                    result[labels[i]].append(output[i])
            else:
                p_labels = output.argmax(1)
                p_labels = p_labels.data.cpu().numpy()
                output = output.data.cpu().numpy()
                for i in range(labels.size(0)):
                    result[p_labels[i]].append(output[i])

    logits = []
    for i in range(args.num_classes):
        tmp = np.mean(result[i], axis=0)
        logits.append(tmp)
    
    return np.array(logits)


def kl_divergence(p, q):
    epsilon = 1e-8
    p = p + epsilon
    q = q + epsilon

    kl_div_batch = torch.sum(p * torch.log(p / q), dim=1)
    # mask = kl_div_batch < 1e-5
    kl_loss = torch.sum(kl_div_batch, dim=0)

    return kl_div_batch, kl_loss