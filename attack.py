import torch
import numpy as np
import enum
import copy
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from copy import deepcopy
import random


def LF_attack(each_worker_label, flip_labels, mal_clients, ctx, type = 1):
    if type == 0:
        for client in mal_clients:
            each_worker_label[client] = torch.Tensor(label_flip(each_worker_label[client].cpu().numpy(), flip_labels)).long().to(ctx)
    else:
        for client in mal_clients:
            each_worker_label[client] = torch.Tensor(label_flip_next(each_worker_label[client].cpu().numpy(), flip_labels)).long().to(ctx)

    return each_worker_label

def set_lfa_labels(label_list, flip_labels = None, type = 0):
    if flip_labels is None:
        if type == 0:
            flip_labels_return = {2: 7}
        else:
            flip_labels_return = {label: label_list[len(label_list) - index - 1] for index, label in enumerate(label_list)}
    else:
        flip_labels_return = flip_labels

    return flip_labels_return

def label_flip(data, flip_labels, poison_percent = -1):
    data = list(data)
    for source_label, target_label in flip_labels.items():
        total_occurences = len([1 for label in data if label == source_label])
        poison_count = poison_percent * total_occurences

        # Poison all and keep only poisoned samples
        if poison_percent == -1:
            data=[target_label for label in data if label == source_label]

        else:
            label_poisoned = 0
            for index, _ in enumerate(data):
                if data[index] == source_label:
                    data[index] = target_label
                    label_poisoned += 1
                if label_poisoned >= poison_count:
                    break

    return data

def label_flip_next(data, flip_labels, poison_percent = 1.0):
    data = list(data)
    poison_count = poison_percent * len(data)
    if poison_percent == -1:
        poison_count = len(data) 

    label_poisoned = 0
    for index, _ in enumerate(data):
        if data[index] in flip_labels.keys():
            data[index] = flip_labels[data[index]]
            label_poisoned += 1
        if label_poisoned >= poison_count:
            break

    return data

def layer_replacement_attack(model_to_attack, model_to_reference, layers):
    params1 = model_to_attack.state_dict().copy()
    params2 = model_to_reference.state_dict().copy()
    
    for layer in layers:
        params1[layer] = params2[layer]
    
    model = copy.deepcopy(model_to_attack)
    model.load_state_dict(params1, strict=False)
    return model

def get_malicious_updates_fang_trmean(all_updates, deviation, n_attackers, epoch_num):
    b = 2
    max_vector = torch.max(all_updates, 0)[0] # maximum_dim
    min_vector = torch.min(all_updates, 0)[0] # minimum_dim

    max_ = (max_vector > 0).type(torch.FloatTensor).cuda()
    min_ = (min_vector < 0).type(torch.FloatTensor).cuda()

    max_[max_ == 1] = b
    max_[max_ == 0] = 1 / b
    min_[min_ == 1] = b
    min_[min_ == 0] = 1 / b

    max_range = torch.cat((max_vector[:, None], (max_vector * max_)[:, None]), dim=1)
    min_range = torch.cat(((min_vector * min_)[:, None], min_vector[:, None]), dim=1)

    rand = torch.from_numpy(np.random.uniform(0, 1, [len(deviation), n_attackers])).type(torch.FloatTensor).cuda()

    max_rand = torch.stack([max_range[:, 0]] * rand.shape[1]).T + rand * torch.stack([max_range[:, 1] - max_range[:, 0]] * rand.shape[1]).T
    min_rand = torch.stack([min_range[:, 0]] * rand.shape[1]).T + rand * torch.stack([min_range[:, 1] - min_range[:, 0]] * rand.shape[1]).T

    mal_vec = (torch.stack([(deviation > 0).type(torch.FloatTensor)] * max_rand.shape[1]).T.cuda() * max_rand + torch.stack(
        [(deviation > 0).type(torch.FloatTensor)] * min_rand.shape[1]).T.cuda() * min_rand).T

    mal_updates = torch.cat((mal_vec, all_updates), 0)

    return mal_updates

def fang_attack_trmean_partial(all_updates, n_attackers):

    model_re = torch.mean(all_updates, 0)
    model_std = torch.std(all_updates, 0)
    deviation = torch.sign(model_re)
    
    max_vector_low = model_re + 3 * model_std 
    max_vector_hig = model_re + 4 * model_std
    min_vector_low = model_re - 4 * model_std
    min_vector_hig = model_re - 3 * model_std

    max_range = torch.cat((max_vector_low[:,None], max_vector_hig[:,None]), dim=1)
    min_range = torch.cat((min_vector_low[:,None], min_vector_hig[:,None]), dim=1)

    rand = torch.from_numpy(np.random.uniform(0, 1, [len(deviation), n_attackers])).type(torch.FloatTensor).cuda()

    max_rand = torch.stack([max_range[:, 0]] * rand.shape[1]).T + rand * torch.stack([max_range[:, 1] - max_range[:, 0]] * rand.shape[1]).T
    min_rand = torch.stack([min_range[:, 0]] * rand.shape[1]).T + rand * torch.stack([min_range[:, 1] - min_range[:, 0]] * rand.shape[1]).T

    mal_vec = (torch.stack([(deviation > 0).type(torch.FloatTensor)] * max_rand.shape[1]).T.cuda() * max_rand + torch.stack(
        [(deviation > 0).type(torch.FloatTensor)] * min_rand.shape[1]).T.cuda() * min_rand).T

    return mal_vec

def multi_krum(all_updates, n_attackers, multi_k=False, verbose=False):
    nusers = all_updates.shape[0]
    candidates = []
    candidate_indices = []
    remaining_updates = all_updates
    all_indices = np.arange(len(all_updates))

    while len(remaining_updates) > 2 * n_attackers + 2:
        distances = []
        for update in remaining_updates:
            distance = torch.norm((remaining_updates - update), dim=1) ** 2
            distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)

        distances = torch.sort(distances, dim=1)[0]
        scores = torch.sum(distances[:, :len(remaining_updates) - 2 - n_attackers], dim=1)
        indices = torch.argsort(scores) # [:len(remaining_updates) - 2 - n_attackers]

        #if verbose: print(indices)
        candidate_indices.append(all_indices[indices[0].cpu().numpy()])
        all_indices = np.delete(all_indices, indices[0].cpu().numpy())
        candidates = remaining_updates[indices[0]][None, :] if not len(candidates) else torch.cat((candidates, remaining_updates[indices[0]][None, :]), 0)
        remaining_updates = torch.cat((remaining_updates[:indices[0]], remaining_updates[indices[0] + 1:]), 0)
        if not multi_k:
            break
    
    aggregate = torch.mean(candidates, dim=0)
    return aggregate, np.array(candidate_indices)

def compute_lambda(all_updates, model_re, n_attackers):

    distances = []
    n_benign, d = all_updates.shape
    for update in all_updates:
        distance = torch.norm((all_updates - update), dim=1)
        distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)

    distances[distances == 0] = 10000
    distances = torch.sort(distances, dim=1)[0]
    scores = torch.sum(distances[:, :n_benign - 2 - n_attackers], dim=1)
    min_score = torch.min(scores)
    term_1 = min_score / ((n_benign - n_attackers - 1) * torch.sqrt(torch.Tensor([d]))[0])
    max_wre_dist = torch.max(torch.norm((all_updates - model_re), dim=1)) / (torch.sqrt(torch.Tensor([d]))[0])

    return (term_1 + max_wre_dist)


def get_malicious_updates_fang_krum(all_updates, model_re, deviation, n_attackers):

    lamda = compute_lambda(all_updates, model_re, n_attackers)

    threshold = 1e-5
    mal_update = []

    while lamda > threshold:
        mal_update = (-lamda * deviation)
        mal_updates = torch.stack([mal_update] * n_attackers)
        mal_updates = torch.cat((mal_updates, all_updates), 0)

        agg_grads, krum_candidate = multi_krum(mal_updates, n_attackers, multi_k=False)
        if krum_candidate < n_attackers:
            return mal_update
        else:
            mal_update = []

        lamda *= 0.5

    if not len(mal_update):
        mal_update = (model_re - lamda * deviation)
        
    return mal_update


def lie_backdoor(origin_params, net, local_net, data, label, ctx, args, backdoor_type, alpha):
    model_dict = deepcopy(net.state_dict())
    param_1 = [model_dict[key] for key in model_dict.keys() if "num_batches_tracked" not in key]

    local_net.load_state_dict(model_dict)
    optimizer = optim.SGD(local_net.parameters(), lr=args.global_lr)        

    local_net.train()
        
    optimizer = optim.SGD(local_net.parameters(), lr=0.5)
    classification_loss = nn.NLLLoss()
    dist_loss_func = nn.MSELoss()

    train_data = DataLoader(TensorDataset(data,label),args.batch_size,shuffle=True,drop_last=False)
    
    for i in range(10):
        for j, item in enumerate(train_data):
            x, y = item
            optimizer.zero_grad()
            output = local_net(x)
            loss = classification_loss(output, y) * alpha
            
            if alpha > 0:
                dist_loss = 0
                name_list = [key for key in model_dict.keys() if "num_batches_tracked" not in key] # [model_dict[key] for key in model_dict.keys() if "num_batches_tracked" not in key]
                for name, p in local_net.named_parameters():
                    idx = name_list.index(name)
                    dist_loss += dist_loss_func((param_1[idx] - p), torch.mean(torch.stack([param[idx] for param in origin_params]), dim=0))
                
                if torch.isnan(dist_loss):
                    raise Exception("Got nan dist loss")

                loss += dist_loss * (1-alpha)

            if torch.isnan(loss):
                raise Exception("Got nan loss")
            loss.backward(retain_graph=True)
            optimizer.step()

    model_dict_2 = deepcopy(local_net.state_dict())
    param_2 = [model_dict_2[key] for key in model_dict_2.keys() if "num_batches_tracked" not in key]

    res = [(param_1[i]-param_2[i])  for i in range(len(param_2))]

    return res