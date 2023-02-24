import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader
import numpy as np
import numpy.linalg as la
from sklearn.cluster import SpectralClustering
from tqdm import tqdm

import os
import math
import argparse

from models import *

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class DatasetFromFile(Dataset):
    def __init__(self, filename):
        examples = []
        with open(filename, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line == '':
                    continue
                line = [int(x) for x in line.split(',')]
                examples.append(line)
        x = torch.LongTensor(examples)
        self.x = x

    def __getitem__(self, index):
        return self.x[index]

    def __len__(self):
        return len(self.x)

def init():
    global device
    global CUDA_CORE

    torch.set_default_dtype(torch.float64)

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--dataset_path', default='', type=str)
    arg_parser.add_argument('--dataset', default='', type=str)
    arg_parser.add_argument('--device', default='cuda', type=str)
    arg_parser.add_argument('--cuda_core', default='0', type=str)
    arg_parser.add_argument('--model', default='DPP', type=str)
    arg_parser.add_argument('--max_epoch', default=20, type=int)
    arg_parser.add_argument('--batch_size', default=8, type=int)
    arg_parser.add_argument('--lr', default=0.001, type=float)
    arg_parser.add_argument('--weight_decay', default=0.0, type=float)
    arg_parser.add_argument('--component_num', default=10, type=int)
    arg_parser.add_argument('--max_cluster_size', default=10, type=int)
    arg_parser.add_argument('--log_file', default='log.txt', type=str)
    arg_parser.add_argument('--output_model_file', default='model.pt', type=str)
    arg_parser.add_argument('--evidence_idx', default=0, type=int)

    args = arg_parser.parse_args()

    device = args.device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_core

    return args

def load_data(dataset_path, dataset,
            load_train=True, load_valid=True, load_test=True):
    dataset_path += '{}/'.format(dataset)
    train_path = dataset_path + '{}.train.data'.format(dataset)
    valid_path = dataset_path + '{}.valid.data'.format(dataset)
    test_path = dataset_path + '{}.test.data'.format(dataset)

    train, valid, test = None, None, None

    if load_train:
        train = DatasetFromFile(train_path)
    if load_valid:
        valid = DatasetFromFile(valid_path)
    if load_test:
        test = DatasetFromFile(test_path)

    return train, valid, test

def partition_variables(trainx, max_cluster_size):
    n = len(trainx)
    m = len(trainx[0])
    k = max_cluster_size

    freq = {}
    for i in range(0, m):
        freq[i] = 0
        for j in range(i + 1, m):
            freq[(i, j)] = 0
    for t in tqdm(range(0, n)):
        for i in range(0, m):
            if trainx[t][i] == 1:
                freq[i] += 1
                for j in range(i + 1, m):
                    if trainx[t][j] == 1:
                        freq[(i, j)] += 1
    for i in freq:
        freq[i] /= n

    E = []
    for i in range(0, m):
        if abs(freq[i]) < 1e-15:
            continue
        for j in range(i + 1, m):
            if abs(freq[j]) < 1e-15:
                continue
            p = freq[(i, j)] / (freq[i] * freq[j])
            if p < 1.0:
                continue
            w = freq[(i, j)] * math.log(p)
            E.append(((i, j), w))
    E = sorted(E, key=lambda x: x[1], reverse=True)

    fa = [i for i in range(0, m)]
    def find(x):
        if fa[x] == x:
            return x
        fa[x] = find(fa[x])
        return fa[x]

    def count(x):
        cnt = 0
        for i in range(0, m):
            if find(i) == x:
                cnt += 1
        return cnt

    set_cnt = m
    for e, w in E:
        if w < 0:
            break
        u, v = e
        fu, fv = find(u), find(v)
        size_u, size_v = count(fu), count(fv)
        if size_u + size_v > k:
            continue
        fa[u] = fv
        fa[fu] = fv
        if fu != fv:
            set_cnt -= 1

    for i in range(0, m):
        fa[i] = find(i)

    res = {}
    for u in range(0, m):
        fu = fa[u]
        if fu not in res:
            res[fu] = []
        res[fu].append(u)

    partition = []
    for k, v in res.items():
        partition.append(v)

    return partition

def nll(y):
    ll = -torch.sum(y)
    return ll

def avg_ll(model, dataset_loader):
    lls = []
    dataset_len = 0
    for x_batch in dataset_loader:
        x_batch = x_batch.to(device)
        y_batch = model(x_batch)
        ll = torch.sum(y_batch)
        lls.append(ll.item())
        dataset_len += x_batch.shape[0]

    avg_ll = torch.sum(torch.Tensor(lls)).item() / dataset_len
    return avg_ll

def train_model(model, train, valid, test,
                lr, weight_decay, batch_size, max_epoch,
                log_file, output_model_file, dataset_name):
    valid_loader, test_loader = None, None
    train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
    if valid is not None:
        valid_loader = DataLoader(dataset=valid, batch_size=batch_size, shuffle=True)
    if test is not None:
        test_loader = DataLoader(dataset=test, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # training loop
    max_valid_ll = -1.0e7
    model = model.to(device)
    model.train()

    for epoch in range(0, max_epoch):
        print('Epoch: {}'.format(epoch))

        # step in train
        for x_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = model(x_batch)
            loss = nll(y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.save(model, output_model_file)

        # compute likelihood on train, valid and test
        train_ll = avg_ll(model, train_loader)
        valid_ll = avg_ll(model, valid_loader)
        test_ll = avg_ll(model, test_loader)

        print('Dataset {}; Epoch {}; train ll: {}; valid ll: {}; test ll: {}'.format(dataset_name, epoch, train_ll, valid_ll, test_ll))

        with open(log_file, 'a+') as f:
            f.write('{} {} {} {}\n'.format(epoch, train_ll, valid_ll, test_ll))

        if output_model_file != '' and valid_ll > max_valid_ll:
            torch.save(model, output_model_file)
            max_valid_ll = valid_ll


def main():
    args = init()

    train, valid, test = load_data(args.dataset_path, args.dataset)

    print('train: {}'.format(train.x.shape))
    if valid is not None:
        print('valid: {}'.format(valid.x.shape))
    if test is not None:
        print('test: {}'.format(test.x.shape))

    m = train.x.shape[1]

    model = None

    if args.model == 'MoAT':
        t_data=train.x.clone()
        t_data.to(device)
        model = MoAT(m, t_data)
        model.to(device)
        train_loader = DataLoader(dataset=train, batch_size=args.batch_size, shuffle=True)
        print('average ll: {}'.format(avg_ll(model, train_loader)))

    if model is None:
        print("invalid model")
        exit(1)

    train_model(model, train=train, valid=valid, test=test,
        lr=args.lr, weight_decay=args.weight_decay,
        batch_size=args.batch_size, max_epoch=args.max_epoch,
        log_file=args.log_file, output_model_file=args.output_model_file,
        dataset_name=args.dataset)


if __name__ == '__main__':
    main()