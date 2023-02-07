import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset
import torchvision
from torchvision import transforms
import datetime
import os
import random

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split

import models
import collections
import numpy as np
import byzantine

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_pc", help="the number of data the server holds", type=int, default=100)
    parser.add_argument("--dataset", help="dataset", type=str, default="FashionMNIST")
    parser.add_argument("--bias", help="degree of non-iid", type=float, default=0.5)
    parser.add_argument("--net", help="net", type=str, default="cnn")
    parser.add_argument("--batch_size", help="batch size", type=int, default=32)
    # parser.add_argument("--lr", help="learning rate", type=float, default=0.6)
    parser.add_argument("--local_lr", help="local learning rate", type=float, default=0.6)
    parser.add_argument("--global_lr", help="global learning rate", type=float, default=0.6)
    parser.add_argument("--nworkers", help="# workers", type=int, default=100)
    parser.add_argument("--niter", help="# iterations", type=int, default=2500)
    parser.add_argument("--nrepeats", help="seed", type=int, default=0)
    parser.add_argument("--nbyz", help="# byzantines", type=int, default=20)
    parser.add_argument("--byz_type", help="type of attack", type=str, default="no")
    parser.add_argument("--aggregation", help="aggregation", type=str, default="fedavg")
    parser.add_argument("--MULTIGPU", help="MULTIGPU", type=bool, default=False)
    parser.add_argument("--p", help="bias probability of 1 in server sample", type=float, default=0.1)
    parser.add_argument("--local_iter", help="the number of local iterations", type=int, default=5)
    parser.add_argument("--pre_iter", help="the number of pretrain iterations", type=int, default=60)
    parser.add_argument("--pool_size", help="process pool size", type=int, default=5)
    parser.add_argument("--bd_type", help="backdoor type", type=str, default="pattern")
    parser.add_argument("--thres", help="mask thresholding", type=float, default=0.5)
    return parser.parse_args()



def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for i, group in enumerate(optimizer.param_groups):
        for j, param in enumerate(group["params"]):
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def HAR_dataloader():
    df_train=pd.read_csv('./data/HAR/train.csv')
    X=pd.DataFrame(df_train.drop(['Activity','subject'],axis=1))
    y=df_train.Activity.values.astype(object)
    group=df_train.subject.values.astype(object)

    encoder=preprocessing.LabelEncoder()
    encoder.fit(y)
    y_train_csv=encoder.transform(y)
    scaler=StandardScaler()
    X_train_csv=scaler.fit_transform(X)
    
    temp_data = [[] for _ in range(30)]
    temp_label = [[] for _ in range(30)]
    for i in range(len(group)):
        g = group[i] - 1
        temp_data[g].append(X_train_csv[i])
        temp_label[g].append(y_train_csv[i])


    df_test=pd.read_csv('./data/HAR/test.csv')
    Xx=pd.DataFrame(df_test.drop(['Activity','subject'],axis=1))
    yy=df_test.Activity.values.astype(object)
    group_1=df_test.subject.values.astype(object)

    encoder_1=preprocessing.LabelEncoder()
    encoder_1.fit(yy)
    y_test_csv=encoder_1.transform(yy)
    scaler=StandardScaler()
    X_test_csv=scaler.fit_transform(Xx)

    for i in range(len(group_1)):
        g = group_1[i] - 1
        temp_data[g].append(X_test_csv[i])
        temp_label[g].append(y_test_csv[i])

    each_worker_data = []
    each_worker_label = []
    server_data = []
    server_label = []
    X_test = []
    y_test = []

    for i in range(30):
        X_t, Xx_test, y_t, yy_test = train_test_split(temp_data[i],temp_label[i],test_size=0.2,random_state=100)
        X_test = X_test + Xx_test
        y_test = y_test + yy_test
        X_worker, X_server, y_worker, y_server = train_test_split(X_t,y_t,test_size=0.03,random_state=100)
        X_worker, y_worker = torch.tensor(X_worker, dtype=torch.float), torch.tensor(y_worker, dtype=torch.long)
        X_worker = X_worker.cpu()
        y_worker = y_worker.cpu()
        each_worker_data.append(X_worker)
        each_worker_label.append(y_worker)
        server_data = server_data + X_server
        server_label = server_label + y_server

    server_data, server_label = torch.tensor(server_data, dtype=torch.float), torch.tensor(server_label, dtype=torch.long)
    server_data = server_data.cpu()
    server_label = server_label.cpu()
    X_test, y_test = torch.tensor(X_test, dtype=torch.float), torch.tensor(y_test, dtype=torch.long)
    return each_worker_data, each_worker_label, server_data, server_label, DataLoader(TensorDataset(X_test, y_test), len(X_test), drop_last = True, shuffle=False)

def load_data(dataset):
    # load the dataset
    if dataset == 'FashionMNIST':
        train_data = DataLoader(torchvision.datasets.FashionMNIST(root = './data/', train=True, download=True, transform=transforms.ToTensor()), 60000, drop_last = True, shuffle=True)
        test_data = DataLoader(torchvision.datasets.FashionMNIST(root = './data/', train=False, download=True, transform=transforms.ToTensor()), 250, drop_last = True, shuffle=False)
    elif dataset == 'CIFAR-10':
        train_data = DataLoader(torchvision.datasets.CIFAR10(root = './data/', train=True, download=True, transform=transforms.ToTensor()), 50000, drop_last = True, shuffle=True)
        test_data = DataLoader(torchvision.datasets.CIFAR10(root = './data/', train=False, download=True, transform=transforms.ToTensor()), 250, drop_last = True, shuffle=False)
    elif dataset == 'HAR':
        X_train, X_test, y_train, y_test = HAR_dataloader()
        X_train, X_test, y_train, y_test = torch.tensor(X_train, dtype=torch.float), torch.tensor(X_test, dtype=torch.float), torch.tensor(y_train, dtype=torch.long), torch.tensor(y_test, dtype=torch.long)
        train_data = DataLoader(TensorDataset(X_train, y_train),6984, drop_last = True, shuffle=True)
        test_data = DataLoader(TensorDataset(X_test, y_test), 368, drop_last = True, shuffle=False)
    else:
        raise NotImplementedError
    return train_data, test_data

def get_net(net_type):
    # define the model architecture
    if net_type == 'cnn':
        net = models.CNN().cpu()
    elif net_type == 'resnet20':
        net = models.ResNet(models.ResidualBlock).cpu()
    elif net_type == 'LR':
        net = models.LR().cpu()
    else:
        raise NotImplementedError
    return net

def get_byz(byz_type):
    # get the attack type
    if byz_type == 'trim_attack':
        return byzantine.trim_attack
    elif byz_type == 'krum_attack':
        return byzantine.krum_attack
    elif byz_type == 'minmax_agnostic':
        return byzantine.minmax_agnostic
    elif byz_type == 'minsum_agnostic':
        return byzantine.minsum_agnostic
    elif byz_type == 'lie_backdoor':
        return byzantine.lie_drift
    else:
        return byzantine.no_byz

def get_n_gpu(MULTIGPU):
    if MULTIGPU == True:
        n_gpu = torch.cuda.device_count()
    else:
        n_gpu = 1
    return n_gpu

def count_num(each_worker_label):    
    for worker_label in each_worker_label:
        worker_label = worker_label.cpu().numpy()
        c = collections.Counter(worker_label)
        print(c)

def assign_data(train_data, bias, ctx, num_labels=10, num_workers=100, server_pc=100, p=0.1, dataset="FashionMNIST", seed=1):
    # assign data to the clients
    other_group_size = (1 - bias) / (num_labels - 1)
    worker_per_group = num_workers / num_labels
    #assign training data to each worker
    each_worker_data = [[] for _ in range(num_workers)]
    each_worker_label = [[] for _ in range(num_workers)]   
    server_data = []
    server_label = [] 
    
    # compute the labels needed for each class
    real_dis = [1. / num_labels for _ in range(num_labels)]
    samp_dis = [0 for _ in range(num_labels)]
    num1 = int(server_pc * p)
    samp_dis[1] = num1
    average_num = (server_pc - num1) / (num_labels - 1)
    resid = average_num - np.floor(average_num)
    sum_res = 0.
    for other_num in range(num_labels - 1):
        if other_num == 1:
            continue
        samp_dis[other_num] = int(average_num)
        sum_res += resid
        if sum_res >= 1.0:
            samp_dis[other_num] += 1
            sum_res -= 1
    samp_dis[num_labels - 1] = server_pc - np.sum(samp_dis[:num_labels - 1])

    # randomly assign the data points based on the labels
    server_counter = [0 for _ in range(num_labels)]
    
    for _, item in enumerate(train_data):
        data, label = item
        for (x, y) in zip(data, label):
            if dataset == "FashionMNIST":
                x = x.to(ctx).reshape(1,1,28,28)
            elif dataset == "CIFAR-10":
                x = x.to(ctx).reshape(1,3,32,32)
            elif dataset == "HAR":
                x = x.to(ctx).reshape(1,561)
            else:
                raise NotImplementedError
            
            y = y.to(ctx).reshape(1)
            
            upper_bound = (y.cpu().numpy()) * (1. - bias) / (num_labels - 1) + bias
            lower_bound = (y.cpu().numpy()) * (1. - bias) / (num_labels - 1)
            rd = np.random.random_sample()
            
            if rd > upper_bound:
                worker_group = int(np.floor((rd - upper_bound) / other_group_size) + y.cpu().numpy() + 1)
            elif rd < lower_bound:
                worker_group = int(np.floor(rd / other_group_size))
            else:
                worker_group = y.cpu().numpy()
            
            if server_counter[int(y.cpu().numpy())] < samp_dis[int(y.cpu().numpy())]:
                server_data.append(x)
                server_label.append(y)
                server_counter[int(y.cpu().numpy())] += 1
            else:
                rd = np.random.random_sample()
                selected_worker = int(worker_group * worker_per_group + int(np.floor(rd * worker_per_group)))
                each_worker_data[selected_worker].append(x)
                each_worker_label[selected_worker].append(y)
                
    server_data = torch.cat(server_data, dim=0).cpu()
    server_label = torch.cat(server_label, dim=0).cpu()
    
    each_worker_data = [torch.cat(each_worker, dim=0).cpu() for each_worker in each_worker_data] 
    each_worker_label = [torch.cat(each_worker, dim=0).cpu() for each_worker in each_worker_label]
    
    # randomly permute the workers
    random_order = np.random.RandomState(seed=seed).permutation(num_workers)
    each_worker_data = [each_worker_data[i] for i in random_order]
    each_worker_label = [each_worker_label[i] for i in random_order]

    return server_data, server_label, each_worker_data, each_worker_label


def get_log(args):
    if args.dataset == "HAR":
        id_flag = '/'
    else:
        if args.bias == 0.1:
            id_flag = "/iid/"
        else:
            id_flag = "/noniid/"

    path = "./new_log/"+ args.dataset + id_flag + args.byz_type + "/" + args.aggregation + "/"
    if not os.path.exists(path):
        os.makedirs(path)

    log_info = str(args.nworkers) + '_' + str(args.nbyz) + '_' + str(args.bias) + '_' + str(args.p) + '_'
    timestr = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")

    if os.path.exists(path+"log_"+log_info+timestr+".txt"):
        return path+"log_"+log_info+timestr+".txt"
    else:
        os.mknod(path+"log_"+log_info+timestr+".txt")

    return path+"log_"+log_info+timestr+".txt"


def get_maskdir(args):
    if args.dataset == "HAR":
        id_flag = '/'
    else:
        if args.bias == 0.1:
            id_flag = "/iid/"
        else:
            id_flag = "/noniid/"

    path = "./new_mask/"+ args.dataset + id_flag + args.byz_type + "/" + args.aggregation + "/"
    if not os.path.exists(path):
        os.makedirs(path)

    return path


def write_log(log_file, content):
    fo = open(log_file, 'a')
    fo.write(content)
    fo.close