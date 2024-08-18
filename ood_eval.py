import time
import torch
import argparse
import numpy as np

from ood_methods.MSP import msp_eval
from ood_methods.Energy import energy_eval

from utils.utils import fix_random_seed
from utils.dataset import get_dataset
from utils.models import get_model
from utils.metrics import cal_metric


def get_eval_options():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ind_dataset", type=str, default="ImageNet")
    parser.add_argument("--ood_dataset", type=str, default="iNat")
    parser.add_argument("--model", type=str, default="ResNet")
    parser.add_argument("--gpu", type=int, default=0)

    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--bs", type=int, default=128)
    parser.add_argument("--OOD_method", type=str, default="Energy")

    parser.add_argument('--num_classes', type=int, default=1000)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    start_time = time.time()
    args = get_eval_options()
    fix_random_seed(args.random_seed)
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    _, ind_dataset = get_dataset(args.ind_dataset)
    _, ood_dataset = get_dataset(args.ood_dataset)

    ind_loader = torch.utils.data.DataLoader(dataset=ind_dataset, batch_size=args.bs, pin_memory=True, num_workers=2, shuffle=False)
    ood_loader = torch.utils.data.DataLoader(dataset=ood_dataset, batch_size=args.bs, pin_memory=True, num_workers=2, shuffle=False)

    model = get_model(args).to(device)
    model.eval()
    ind_scores, ood_scores = None, None

    if args.OOD_method == "MSP":
        ind_scores = msp_eval(model, ind_loader)
        ood_scores = msp_eval(model, ood_loader)
    elif args.OOD_method == "Energy":
        ind_scores = energy_eval(model, ind_loader)
        ood_scores = energy_eval(model, ood_loader)

    ind_labels = np.ones(ind_scores.shape[0])
    ood_labels = np.zeros(ood_scores.shape[0])

    labels = np.concatenate([ind_labels, ood_labels])
    scores = np.concatenate([ind_scores, ood_scores])

    auroc, aupr, fpr = cal_metric(labels, scores)

    print("{:10} {}".format("AUROC:", auroc))
    print("{:10} {}".format("FPR:", fpr))

    finish_time = time.time()
    print(finish_time - start_time)
