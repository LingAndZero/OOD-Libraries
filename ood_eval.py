import time
import torch
import torch.nn as nn
import argparse
import os
import numpy as np

from ood_methods.MSP import msp_eval
from ood_methods.Energy import energy_eval
from ood_methods.ODIN import odin_eval
from ood_methods.Mahalanobis import mahalanobis_eval
from ood_methods.ReAct import react_eval, get_threshold
from ood_methods.GradNorm import gradnorm_eval

from ood_methods.Mutation import mutation_eval
from ood_methods.Distil import distil_eval, get_logits

from utils.utils import fix_random_seed
from utils.dataset import get_dataset
from utils.models import get_model
from utils.metrics import cal_metric


def get_eval_options():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ind_dataset", type=str, default="cifar10")
    parser.add_argument("--ood_dataset", type=str, default="LSUN_resize")
    parser.add_argument("--model", type=str, default="ResNet")
    parser.add_argument("--gpu", type=int, default=0)

    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--bs", type=int, default=256)
    parser.add_argument("--OOD_method", type=str, default="MSP")

    parser.add_argument('--num_classes', type=int, default=10)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    start_time = time.time()
    args = get_eval_options()
    fix_random_seed(args.random_seed)
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    _, ind_dataset = get_dataset(args.ind_dataset)
    _, ood_dataset = get_dataset(args.ood_dataset)

    if args.OOD_method == 'GradNorm':
        args.bs = 1

    ind_loader = torch.utils.data.DataLoader(dataset=ind_dataset, batch_size=args.bs, pin_memory=True, num_workers=2, shuffle=False)
    ood_loader = torch.utils.data.DataLoader(dataset=ood_dataset, batch_size=args.bs, pin_memory=True, num_workers=2, shuffle=False)

    model = get_model(args, pretrain=True).cuda()
    if torch.cuda.device_count() > 1 and args.OOD_method != 'GradNorm':
        model = nn.DataParallel(model, device_ids=[0, 1])
    model.eval()
    ind_scores, ood_scores = None, None

    if args.OOD_method == "MSP":
        ind_scores = msp_eval(model, ind_loader)
        ood_scores = msp_eval(model, ood_loader)
    elif args.OOD_method == "Energy":
        ind_scores = energy_eval(model, ind_loader)
        ood_scores = energy_eval(model, ood_loader)
    elif args.OOD_method == "ODIN":
        ind_scores = odin_eval(model, ind_loader)
        ood_scores = odin_eval(model, ood_loader)
    elif args.OOD_method == "Mahalanobis":
        ind_scores = mahalanobis_eval(model, ind_loader)
        ood_scores = mahalanobis_eval(model, ood_loader)
    elif args.OOD_method == "ReAct":
        train_data, _ = get_dataset(args.ind_dataset)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=1024, shuffle=False)
        threshold = get_threshold(model, train_loader, 90)
        # with open('result/threshold.txt', 'w') as f:
        #     f.write(str(threshold))
        ind_scores = react_eval(model, ind_loader, threshold)
        ood_scores = react_eval(model, ood_loader, threshold)
    elif args.OOD_method == "GradNorm":
        ind_scores = gradnorm_eval(model, ind_loader, args)
        ood_scores = gradnorm_eval(model, ood_loader, args)

    elif args.OOD_method == "Mutation":
        ind_scores = mutation_eval(model, ind_loader)
        ood_scores = mutation_eval(model, ood_loader)
    elif args.OOD_method == "Distil":
        file_logits = "result/logits/{}_{}_in.csv".format(args.ind_dataset, args.model)
        os.makedirs(os.path.dirname(file_logits), exist_ok=True)

        train_data, _ = get_dataset(args.ind_dataset)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=1024, pin_memory=True, shuffle=True, num_workers=2)
        train_logits = get_logits(model, train_loader, args)
        train_logits = torch.from_numpy(train_logits)

        np.savetxt(file_logits, train_logits) 

        ind_scores = distil_eval(model, ind_loader, train_logits, args)
        ood_scores = distil_eval(model, ood_loader, train_logits, args)

    ind_labels = np.ones(ind_scores.shape[0])
    ood_labels = np.zeros(ood_scores.shape[0])

    labels = np.concatenate([ind_labels, ood_labels])
    scores = np.concatenate([ind_scores, ood_scores])

    auroc, aupr, fpr = cal_metric(labels, scores)

    print("{:10} {}".format("AUROC:", auroc))
    print("{:10} {}".format("FPR:", fpr))

    finish_time = time.time()
    print(finish_time - start_time)
