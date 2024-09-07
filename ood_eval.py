import time
import torch
import torch.nn as nn
import argparse
import os
import numpy as np
from ood_methods.Mahalanobis import mahalanobis_eval
from ood_methods.ReAct import react_eval, get_threshold
from ood_methods.GradNorm import gradnorm_eval

from ood_methods.Mutation import mutation_eval
from ood_methods.Distil_del import distil_eval

from utils.utils import fix_random_seed
from utils.dataset import get_dataset
from utils.models import get_model
from utils.metrics import cal_metric


def get_eval_options():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ind_dataset", type=str, default="cifar10")
    parser.add_argument("--ood_dataset", type=str, default="iSUN")
    parser.add_argument("--model", type=str, default="ResNet")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument('--num_classes', type=int, default=10)

    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--bs", type=int, default=1024)
    parser.add_argument("--OOD_method", type=str, default="MSP")

    parser.add_argument('--T', type=int, default=100)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    start_time = time.time()
    args = get_eval_options()
    fix_random_seed(args.random_seed)
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    _, ind_dataset = get_dataset(args.ind_dataset)
    _, ood_dataset = get_dataset(args.ood_dataset)

    if args.OOD_method in ['GradNorm']:
        args.bs = 1

    ind_loader = torch.utils.data.DataLoader(dataset=ind_dataset, batch_size=args.bs, pin_memory=True, num_workers=8, shuffle=False)
    ood_loader = torch.utils.data.DataLoader(dataset=ood_dataset, batch_size=args.bs, pin_memory=True, num_workers=8, shuffle=False)

    model = get_model(args, pretrain=True).to(device)
    model.eval()
    
    ind_scores, ood_scores = None, None

    if args.OOD_method == "MSP":
        from ood_methods.MSP import MSP
        msp = MSP(model, device)

        # step 1: get msp score
        ind_scores = msp.eval(ind_loader)
        ood_scores = msp.eval(ood_loader)

    elif args.OOD_method == "Energy":
        from ood_methods.Energy import Energy
        energy = Energy(model, device)

        # step 1: get energy score
        ind_scores = energy.eval(ind_loader)
        ood_scores = energy.eval(ood_loader)

    elif args.OOD_method == "ODIN":
        from ood_methods.ODIN import odin_eval

        ind_scores = odin_eval(model, ind_loader, device)
        ood_scores = odin_eval(model, ood_loader, device)

    elif args.OOD_method == "Mahalanobis":
        ind_scores = mahalanobis_eval(model, ind_loader)
        ood_scores = mahalanobis_eval(model, ood_loader)

    elif args.OOD_method == "ReAct":

        # load ID threshold
        file_threshold = "result/threshold/{}_{}_in.csv".format(args.ind_dataset, args.model)
        if os.path.exists(file_threshold):
            threshold = torch.from_numpy(np.genfromtxt(file_threshold)).to(device)
            print("load ID threshold")
        else:    
            train_data, _ = get_dataset(args.ind_dataset)
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=1024, pin_memory=True, shuffle=False, num_workers=8)
            threshold = get_threshold(model, train_loader, 90)
            np.savetxt(file_threshold, np.array([threshold]))

        ind_scores = react_eval(model, ind_loader, threshold, device)
        ood_scores = react_eval(model, ood_loader, threshold, device)

    elif args.OOD_method == "GradNorm":
        ind_scores = gradnorm_eval(model, ind_loader, args, device)
        ood_scores = gradnorm_eval(model, ood_loader, args, device)

    elif args.OOD_method == "ASH":
        from ood_methods.ASH import ASH
        ash = ASH(model, device)

        # step 1: get ash score
        ind_scores = ash.eval(ind_loader)
        ood_scores = ash.eval(ood_loader)


    elif args.OOD_method == "Mutation":
        ind_scores = mutation_eval(model, ind_loader)
        ood_scores = mutation_eval(model, ood_loader)

    elif args.OOD_method == "Distil":
        
        # file_logits = "result/logits/{}_{}_in.csv".format(args.ind_dataset, args.model)

        # # load training logits
        # if os.path.exists(file_logits):
        #     train_logits = torch.from_numpy(np.genfromtxt(file_logits))
        #     print("load train_logits")
        # else:
        #     train_data, _ = get_dataset(args.ind_dataset)
        #     train_loader = torch.utils.data.DataLoader(train_data, batch_size=1024, pin_memory=True, shuffle=True, num_workers=8)
        #     train_logits = get_logits(model, train_loader, args, device)
        #     train_logits = torch.from_numpy(train_logits)
        #     np.savetxt(file_logits, train_logits)

        # file_path_in = "result/{}_{}_{}_in.csv".format(args.OOD_method, args.ind_dataset, args.model)
        # if os.path.exists(file_path_in):
        #    ind_scores = np.genfromtxt(file_path_in, delimiter=' ')
        #    print("load ind-scores")
        # else:
        #    ind_scores = distil_eval(model, ind_loader, train_logits, args)
        #    np.savetxt(file_path_in, ind_scores, delimiter=' ')

        # file_path_out = "result/{}_{}_{}_{}_out.csv".format(args.OOD_method, args.ind_dataset, args.ood_dataset, args.model)
        # if os.path.exists(file_path_out):
        #     ood_scores = np.genfromtxt(file_path_out, delimiter=' ')
        #     print("load ood-scores")
        # else:
        #     ood_scores = distil_eval(model, ood_loader, train_logits, args)
        #     np.savetxt(file_path_out, ood_scores, delimiter=' ')
        ind_scores = distil_eval(model, ind_loader, args, device)
        ood_scores = distil_eval(model, ood_loader, args, device)

    ind_labels = np.ones(ind_scores.shape[0])
    ood_labels = np.zeros(ood_scores.shape[0])

    labels = np.concatenate([ind_labels, ood_labels])
    scores = np.concatenate([ind_scores, ood_scores])

    auroc, aupr, fpr = cal_metric(labels, scores)

    print("{:10} {}".format("AUROC:", auroc))
    print("{:10} {}".format("FPR:", fpr))

    finish_time = time.time()
    print(finish_time - start_time)
