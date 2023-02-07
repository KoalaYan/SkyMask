import torch.optim as optim
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import random
from copy import deepcopy
import yaml
import os
import psutil
from torch.utils.data import DataLoader, TensorDataset

import attack
import aggregation
import utils
import time
import models


def get_params(net):
    model_dict = deepcopy(net.state_dict())
    param_1 = [model_dict[key] for key in model_dict.keys() if "num_batches_tracked" not in key]

    return param_1

def create_masknet(param_list, net_type, ctx):
    nworker = len(param_list)
    if net_type == "cnn":
        masknet = models.CNNMaskNet(param_list, nworker, ctx)
    elif net_type == "resnet20":
        masknet = models.ResMaskNet(param_list, nworker, ctx)
    elif net_type == "LR":
        masknet = models.LRMaskNet(param_list, nworker, ctx)
    else:
        masknet = None

    return masknet

def train(net, local_net, data, label, ctx, args):
    train_data = DataLoader(TensorDataset(data,label),args.batch_size,shuffle=True,drop_last=False)
    model_dict = net.state_dict()
    local_net.load_state_dict(model_dict)
    optimizer = optim.SGD(local_net.parameters(), lr=args.local_lr)

    param_1 = get_params(local_net)
    
    for i in range(args.local_iter):
        for j, item in enumerate(train_data):
            x, y = item
            optimizer.zero_grad()
            output = local_net(x)
            loss = F.nll_loss(output, y)
            loss.backward(retain_graph=True)
            utils.clip_gradient(optimizer=optimizer, grad_clip=1e-2)

            optimizer.step()

    param_2 = get_params(local_net)

    res = [(param_1[i]-param_2[i])  for i in range(len(param_2))]
    return res


def evaluate_accuracy(data_iterator, net, ctx, args):
    correct = 0
    with torch.no_grad():
        net.eval()
        for data, target in data_iterator:
            data, target = data.to(ctx), target.to(ctx)
            output = net(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
    correct = 1. * correct / len(data_iterator.dataset)
    net.train()
    return correct    

def main(args):
    s = psutil.Process(os.getpid())
    info = ''
    for i in s.cmdline():
        info += i + ' '

    # Data Parallelism
    if args.MULTIGPU is False:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if device == torch.device('cpu'):
            raise EnvironmentError('No GPUs, cannot initialize multigpu training.')
    print(device)
    # device to use
    ctx = device

    # model architecture
    net = utils.get_net(args.net).to(ctx)
    local_net = utils.get_net(args.net).to(ctx)
    masknet = None

    batch_size = args.batch_size
    num_classes = 0
    if args.dataset == "FashionMNIST" or args.dataset == "CIFAR-10":
        num_classes = 10
    elif args.dataset == "HAR":
        num_classes = 6
    byz = utils.get_byz(args.byz_type)
    num_workers = args.nworkers
    nbyz = args.nbyz
    lr = args.global_lr
    niter = args.niter
    
    log_file = utils.get_log(args)
    
    fo = open(log_file, 'a')
    fo.write(info)
    fo.close

    grad_list = []
    test_acc_list = []
       

    # load the data
    # fix the seeds for loading data                                                                                                                            
    seed = args.nrepeats
    if seed > 0:
        torch.random.seed(seed)
        random.seed(seed)
        np.random.seed(seed)
    
    # assign data to the server and clients
    if args.dataset == 'HAR':
        each_worker_data, each_worker_label, server_data, server_label, test_data = utils.HAR_dataloader()
    else:
        train_data, test_data = utils.load_data(args.dataset)
        server_data, server_label, each_worker_data, each_worker_label = utils.assign_data(
                                                                    train_data, args.bias, ctx, num_labels=num_classes, num_workers=num_workers,
                                                                    server_pc=args.server_pc, p=args.p, dataset=args.dataset, seed=seed)

    # count_num(each_worker_label)
    mal_clients = [c for c in range(nbyz)]

    if args.byz_type == "label_flipped":
        labels = [label for label in range(10)]
        if args.dataset == "HAR":
            labels = [label for label in range(6)]

        utils.count_num(each_worker_label[:nbyz])
        each_worker_label = attack.LF_attack(each_worker_label, attack.set_lfa_labels(labels, type=1),mal_clients, ctx)
        print("---------------")
        utils.count_num(each_worker_label[:nbyz])

    elif args.byz_type == 'scaling':
        for id in mal_clients:
            length = len(each_worker_data[id])
            change_list = random.sample(range(length), int(length * 0.5))
            for idx in change_list:
                each_worker_data[id][idx][:, :5, :5] = 1
                each_worker_label[id][idx] = 0
        bd_test_dataloader = deepcopy(test_data)
    
    elif args.byz_type == "lie_backdoor":
        bd_train_data = []
        bd_train_label = []
        bd_test_data = []
        bd_test_label = []
        for id in mal_clients:
            length = len(each_worker_data[id])
            bd_data = []
            bd_label = []  
            for idx in range(length):
                img = deepcopy(each_worker_data[id][idx])
                label = deepcopy(each_worker_label[id][idx])
                if args.dataset == "FashionMNIST":
                    img[:, :5, :5] = 2.8
                    img = img.reshape(1,1,28,28)
                    label *= 0
                elif args.dataset == "CIFAR-10":
                    img[:, :5, :5] = 1
                    img = img.reshape(1,3,32,32)
                    label *= 0
                label = label.reshape(1)
                bd_data.append(img)
                bd_label.append(label)

            bd_data = torch.cat(bd_data, dim=0)
            bd_label = torch.cat(bd_label, dim=0)
            bd_train_data.append(bd_data)
            bd_train_label.append(bd_label)
            
            bd_test_data = bd_data if bd_test_data == [] else torch.cat([bd_test_data, bd_data], dim=0)
            bd_test_label = bd_label if bd_test_label == [] else torch.cat([bd_test_label, bd_label], dim=0) 
        bd_test_dataloader = DataLoader(TensorDataset(bd_test_data.to(ctx), bd_test_label.to(ctx)), 100,shuffle=False,drop_last=False)
        

    # training
    for e in range(niter):
        grad_list = []

        data_list = get_params(net)
        
        for client_id in range(num_workers):
            grad_list.append(train(net, local_net, each_worker_data[client_id].to(ctx), each_worker_label[client_id].to(ctx), ctx, args))
                
        if args.byz_type == 'scaling': # how to backdoor (Scaling)
            grad_list = [[item * (num_workers / nbyz) for item in grad] for grad in grad_list[:nbyz]]+ grad_list[nbyz:]
        elif args.byz_type == 'lie_backdoor': # A little is enough
            mal_grad_list = []
            for mal_id in mal_clients:
                mal_grad = attack.lie_backdoor(grad_list[nbyz:], net, local_net, bd_train_data[mal_id].to(ctx), bd_train_label[mal_id].to(ctx), ctx, args, args.bd_type, 0.8)
                grad_list[mal_id] = mal_grad

        if args.aggregation == "fltrust":
            server_data = server_data
            server_label = server_label

            grad_list.append(train(net, local_net, server_data.to(ctx), server_label.to(ctx), ctx, args))

            grad_list = [torch.cat([xx.reshape((-1, 1)) for xx in x], dim=0) for x in grad_list]
            grad_list = byz(grad_list, net, lr, nbyz, ctx, args)
            t1 = time.time()
            model_dict = aggregation.fltrust(grad_list, net, ctx, args)
            t2 = time.time()
            
        elif args.aggregation == "fedavg":
            grad_list = [torch.cat([xx.reshape((-1, 1)) for xx in x], dim=0) for x in grad_list]
            grad_list = byz(grad_list, net, lr, nbyz, ctx, args)
            t1 = time.time()
            model_dict = aggregation.fedavg(grad_list, net, args)
            t2 = time.time()
        elif args.aggregation == "tolpegin":
            grad_list = [torch.cat([xx.reshape((-1, 1)) for xx in x], dim=0) for x in grad_list]
            grad_list = byz(grad_list, net, lr, nbyz, ctx, args)
            t1 = time.time()
            model_dict = aggregation.Tolpegin(grad_list, e, net, args, log_file)
            t2 = time.time()
        elif args.aggregation == "trim":
            grad_list = [torch.cat([xx.reshape((-1, 1)) for xx in x], dim=0) for x in grad_list]
            grad_list = byz(grad_list, net, lr, nbyz, ctx, args)
            t1 = time.time()
            model_dict = aggregation.trim(grad_list, net, args)
            t2 = time.time()
        elif args.aggregation == "krum":
            grad_list = [torch.cat([xx.reshape((-1, 1)) for xx in x], dim=0) for x in grad_list]
            grad_list = byz(grad_list, net, lr, nbyz, ctx, args)
            t1 = time.time()
            model_dict = aggregation.krum(grad_list, net, args)
            t2 = time.time()
        elif args.aggregation == "mkrum":
            grad_list = [torch.cat([xx.reshape((-1, 1)) for xx in x], dim=0) for x in grad_list]
            grad_list = byz(grad_list, net, lr, nbyz, ctx, args)
            t1 = time.time()
            model_dict = aggregation.multikrum(grad_list, net, args)
            t2 = time.time()
        elif args.aggregation == "bulyan":
            grad_list = [torch.cat([xx.reshape((-1, 1)) for xx in x], dim=0) for x in grad_list]
            grad_list = byz(grad_list, net, lr, nbyz, ctx, args)
            t1 = time.time()
            model_dict = aggregation.bulyan(grad_list, net, args)
            t2 = time.time()
        elif args.aggregation == "skymask":
            server_data = server_data
            server_label = server_label

            grad_list.append(train(net, local_net, server_data.to(ctx), server_label.to(ctx), ctx, args))
            grad_list = [torch.cat([xx.reshape((-1, 1)) for xx in x], dim=0) for x in grad_list]
            grad_list = byz(grad_list, net, lr, nbyz, ctx, args)

            param_list = []
            for grad in grad_list:
                data_item = []
                idx = 0
                for data in data_list:
                    size = 1
                    for item in data.shape:
                        size *= item
                    temp = data - args.local_lr * grad[idx:(idx+size)].reshape(data.shape)
                    data_item.append(temp)
                param_list.append(data_item)

            masknet = create_masknet(param_list, args.net, ctx)

            t1 = time.time()
            model_dict = aggregation.skymask(grad_list, data_list, masknet, ctx, e, server_data.to(ctx), server_label.to(ctx), net, args, log_file)
            t2 = time.time()

        net.load_state_dict(model_dict)

        grad_list = []
        
        # evaluate the model accuracy
        if (e) % 5 == 0:
            test_accuracy = evaluate_accuracy(test_data, net, ctx, args)
            test_acc_list.append(test_accuracy)
            
            fo = open(log_file, 'a')
            fo.write("Iteration %02d. Test_acc %0.4f" % (e, test_accuracy) + '\n')
            fo.write("Iteration %02d. Time %0.4f" % (e, t2-t1) + '\n')
            if args.byz_type == 'lie_backdoor':
                bd_acc = evaluate_accuracy(bd_test_dataloader, net, ctx, args)
                fo.write("Iteration %02d. Poisoned_test_acc %0.4f" % (e, bd_acc) + '\n')
            elif args.byz_type == 'scaling':
                bd_acc = evaluate_accuracy(bd_test_dataloader, net, ctx, args)
                fo.write("Iteration %02d. Poisoned_test_acc %0.4f" % (e, bd_acc) + '\n')
            fo.close
            print("Iteration %02d. Test_acc %0.4f" % (e, test_accuracy))
            print("Iteration %02d. Time %0.4f" % (e, t2-t1) + '\n')
            if args.byz_type == 'lie_backdoor':
                print("Iteration %02d. Poisoned_test_acc %0.4f" % (e, bd_acc))
            elif args.byz_type == 'scaling':
                print("Iteration %02d. Poisoned_test_acc %0.4f" % (e, bd_acc))

    del test_acc_list
    test_acc_list = []

if __name__ == "__main__":
    args = utils.parse_args()
    main(args)