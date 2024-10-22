import time
import torch
import torch.nn as nn
import argparse
import os
import numpy as np

from utils.utils import fix_random_seed
from utils.dataset import get_dataset
from utils.models import get_model
from tqdm import tqdm
import pickle




def get_logits(model, data_loader, args,device, file_name_prefix):
    model.eval()
    features = [[] for i in range(args.num_classes)]
    logits = [[] for i in range(args.num_classes)]
    with torch.no_grad():
        print(data_loader)
        for (images, labels) in tqdm(data_loader):
            images, labels = images.to(device), labels.to(device)
            # output,feature = model.featur_norm(images)
            output, feature = model.feature(images)
            
            p_labels = output.argmax(1)
            for i in range(labels.size(0)):
                logits[p_labels[i]].append(output[i].cpu())
                features[p_labels[i]].append(feature[i].cpu())

    save_features = []
    save_logits = []
    for i in range(args.num_classes):
        if len(logits[i])==0:
            save_features.append(torch.Tensor([]))
            save_logits.append(torch.Tensor([]))
            continue
        tmp = torch.stack(features[i], dim=0)
        save_features.append(tmp)
        tmp = torch.stack(logits[i], dim=0)
        save_logits.append(tmp)
    #保存logits

    with open(file_name_prefix + '_features.pkl', 'wb') as f:
        pickle.dump(save_features, f)
    with open(file_name_prefix + '_logits.pkl', 'wb') as f:
        pickle.dump(save_logits, f)


parser = argparse.ArgumentParser()

parser.add_argument("--ind_dataset", type=str, default="ImageNet")
parser.add_argument("--ood_dataset", type=str, default="iNat")
parser.add_argument("--model", type=str, default="ResNet")
parser.add_argument("--gpu", type=int, default=1)

parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument("--bs", type=int, default=64)
args = parser.parse_args()
device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
fix_random_seed(0)
_, ind_dataset = get_dataset(args.ind_dataset)
_, ood_dataset = get_dataset(args.ood_dataset)
train_data, _ = get_dataset(args.ind_dataset)

ind_loader = torch.utils.data.DataLoader(dataset=ind_dataset, batch_size=args.bs, pin_memory=True, num_workers=24, shuffle=False)
ood_loader = torch.utils.data.DataLoader(dataset=ood_dataset, batch_size=args.bs, pin_memory=True, num_workers=24, shuffle=False)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.bs, pin_memory=True, shuffle=False, num_workers=24)

if args.ind_dataset == "ImageNet":
    args.num_classes = 1000

    model = get_model(args).to(device)
else:
    args.num_classes = 10
    model = get_model(args).to(device)

model.eval()

prefix="stat/"+args.model+"/"
if not os.path.exists(prefix):
    os.makedirs(prefix)

if not os.path.exists(prefix+args.ind_dataset+"_"+"Intrain"+"_features.pkl"):
    get_logits(model,train_loader,args,device,prefix+args.ind_dataset+"_"+"Intrain")



if not os.path.exists(prefix+args.ood_dataset+"_"+"Oodtest"+"_features.pkl"):

    get_logits(model,ood_loader,args,device,prefix+args.ood_dataset+"_"+"Oodtest")

if not os.path.exists(prefix+args.ind_dataset+"_"+"Intest"+"_features.pkl"):
    get_logits(model,ind_loader,args,device,prefix+args.ind_dataset+"_"+"Intest")


