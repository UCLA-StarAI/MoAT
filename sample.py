# Code for sampling experiments
# Compares gibbs sampling, importance sampling, and collapsed sampling.

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

import os
import math
import argparse
import random as rand

from models import *

device = 'cuda'

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def init():
    global device
    global CUDA_CORE

    torch.set_default_dtype(torch.float64)

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--num_vars', type=int)
    arg_parser.add_argument('--device', default='cuda', type=str)
    arg_parser.add_argument('--cuda_core', default='0', type=str)
    arg_parser.add_argument('--model_path', type=str)
    arg_parser.add_argument('--data_path', default=None, type=str)
    arg_parser.add_argument('--num_samples', type=int)
    arg_parser.add_argument('--num_seeds', type=int)
    arg_parser.add_argument('--evidence_count', type=int)

    args = arg_parser.parse_args()

    device = args.device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_core

    return args

def get_samples(s,args,model,evidence,t):
    if t=="imp":
        imp_klds,imp_wts=model.get_importance_samples(evidence,args.num_samples)
        return (s,t,imp_klds,imp_wts)
    if t=="col":
        col_klds,col_wts=model.get_collapsed_importance_samples(evidence,args.num_samples)
        return (s,t,col_klds,col_wts)
    if t=="gibbs":
        gibbs_klds=model.get_gibbs_samples(evidence,burn_in=100,num_samples=args.num_samples)
        return (s,t,gibbs_klds)


def main():
    args = init()
    model=torch.load(args.model_path)
    n=args.num_vars


    evidence=[-1 for i in range(n)]

    if args.data_path is None:
        while(evidence.count(-1)!=n-args.evidence_count):
            idx=rand.randint(0,n-1)
            evidence[idx]=rand.randint(0,1)

    else:
        f=open(args.data_path)
        lines=f.readlines()
        idx=rand.randint(0,len(lines)-1)
        evidence=[int(x) for x in lines[idx].split(',')]
        while(evidence.count(-1)!=n-args.evidence_count):
            idx=rand.randint(0,n-1)
            evidence[idx]=-1

    output=[]
    inputs=[(s,args,model,evidence,t) for t in ["imp", "col", "gibbs"] for s in range(args.num_seeds)]

    for a,b,c,d,e in inputs:
        output.append(get_samples(a,b,c,d,e))

    print("Evidence:")
    print(evidence)
    print("Sampling KLDs:")
    print(output)

if __name__ == '__main__':
    main()