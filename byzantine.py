import numpy as np
import torch

def no_byz(v, net, lr, f, ctx, args):
    return v

def trim_attack(v, net, lr, f, ctx, args):
    vi_shape = v[0].shape
    v_tran = torch.cat(v, dim=1)
    maximum_dim = torch.max(v_tran, dim=1)[0].reshape(vi_shape)
    minimum_dim = torch.min(v_tran, dim=1)[0].reshape(vi_shape)
    direction = torch.sign(torch.sum(v_tran, dim=1, keepdims=True))
    directed_dim = (direction > 0) * minimum_dim + (direction < 0) * maximum_dim
    for i in range(f):
        random_12 = 1. + torch.rand(vi_shape).to(ctx)
        v[i] = directed_dim * ((direction * directed_dim > 0) / random_12 + (direction * directed_dim < 0) * random_12)
    return v

def trim_attack_partial(v, net, lr, f, ctx, args):
    vi_shape = v[0].shape
    v_tran = torch.cat(v, dim=1)
    model_re = torch.mean(v_tran, dim=1).reshape(vi_shape)
    model_std = torch.std(v_tran, dim=1).reshape(vi_shape)

    direction = torch.sign(torch.sum(v_tran, dim=1, keepdims=True))
    directed_dim = (direction > 0) * (- 3 * model_std) + (direction < 0) * (3 * model_std)

    for i in range(f):
        random_34 = 1. + torch.rand(vi_shape).to(ctx) * (1.0/3)
        v[i] = directed_dim * random_34 + model_re
    return v

def compute_lambda(v, n_attackers, ctx):    
    distances = []
    vi_shape = v[0].shape
    v_tran = torch.cat(v, dim=1)
    d, n_benign = v_tran.shape
    model_re = torch.mean(v_tran, dim=1).reshape(vi_shape)
    for update in v:
        distance = torch.norm((v_tran - update), dim=0)
        distances = distance[:, None] if not len(distances) else torch.cat((distances, distance[:, None]), dim=1)

    distances[distances == 0] = 10000
    distances = torch.sort(distances, dim=0)[0]
    scores = torch.sum(distances[:n_benign - 2 - n_attackers, :], dim=0)
    min_score = torch.min(scores, dim=0)[0]
    term_1 = min_score / ((n_benign - n_attackers - 1) * torch.sqrt(torch.Tensor([d]).to(ctx))[0])
    max_wre_dist = torch.max(torch.norm((v_tran - model_re), dim=0), dim=0)[0] / (torch.sqrt(torch.Tensor([d]).to(ctx)))

    return (term_1 + max_wre_dist)
def multi_krum(v_tran, n_attackers, multi_k=False, verbose=False):
    nusers = v_tran.shape[1]
    candidates = []
    candidate_indices = []
    remaining_updates = v_tran
    all_indices = np.arange(nusers)
    while remaining_updates.shape[1] > 2*n_attackers+2:
        distances = []
        for idx in range(remaining_updates.shape[1]):
            update = remaining_updates[:, idx].reshape((remaining_updates[:, idx].shape[0], 1))
            distance = torch.norm((remaining_updates - update), dim=0) ** 2
            
            distances = distance[:, None] if not len(distances) else torch.cat((distances, distance[:, None]), dim=1)
        distances = torch.sort(distances, dim=0)[0]
        scores = torch.sum(distances[:remaining_updates.shape[1]-2-n_attackers,:], dim=0)
        indices = torch.argsort(scores)

        candidate_indices.append(all_indices[indices[0].cpu().numpy()])
        all_indices = np.delete(all_indices, indices[0].cpu().numpy())
        candidates = remaining_updates[:,indices[0]].reshape((remaining_updates[:,indices[0]].shape[0], 1)) if not len(candidates) else torch.cat((candidates, remaining_updates[:,indices[0]]), dim=1)
        remaining_updates = torch.cat((remaining_updates[:,:indices[0]], remaining_updates[:,indices[0]+1:]), 1)

        if not multi_k:
            break

    aggregate = torch.mean(candidates, dim=1)
    return aggregate, np.array(candidate_indices)

def krum_attack(v, net, lr, n_attackers, ctx, args):

    lamda = compute_lambda(v, n_attackers, ctx)

    vi_shape = v[0].shape
    v_tran = torch.cat(v, dim=1)
    model_re = torch.mean(v_tran, dim=1).reshape(vi_shape)
    direction = torch.sign(torch.sum(v_tran, dim=1, keepdims=True))

    threshold = 1e-5
    mal_update = []

    while lamda > threshold:
        mal_update = (-lamda * direction)
        mal_updates = mal_update
        for i in range(n_attackers-1):
            mal_updates = torch.cat((mal_updates, mal_update), dim=1)
        mal_updates = torch.cat((mal_updates, v_tran), dim=1)

        agg_grads, krum_candidate = multi_krum(mal_updates, n_attackers, multi_k=False)
        if krum_candidate < n_attackers:
            del mal_update
            mal_update = []
            for idx in range(mal_updates.shape[1]):
                mal_update.append(mal_updates[:,idx].reshape(mal_updates[:,idx].shape[0], 1))

            return mal_update
        else:
            del mal_update
            mal_update = []

        lamda *= 0.5

    if not len(mal_update):
        res = []
        mal_update = (model_re - lamda * direction)
        for i in range(n_attackers):
            res.append(mal_update)
        
    return res + v


def h(param_list, ctx, part_0, direction):

    n = len(param_list) - 1

    baseline = torch.squeeze(param_list[-1])
    cos_sim = []
    new_param_list = []
    
    for each_param_list in param_list:
        each_param_array = torch.squeeze(each_param_list)
        cos_sim_temp = torch.dot(baseline, each_param_array) / (torch.norm(baseline) + 1e-9) / (torch.norm(each_param_array) + 1e-9)
        cos_sim.append(cos_sim_temp)
    cos_sim = torch.stack(cos_sim)[:-1]

    zero = torch.tensor(np.zeros(cos_sim.shape)).to(ctx)
    cos_sim = torch.maximum(cos_sim, zero)
    
    normalized_weights = cos_sim / (torch.sum(cos_sim) + 1e-9)

    for i in range(n):
        new_param_list.append(param_list[i] * normalized_weights[i] / (torch.norm(param_list[i]) + 1e-9) * torch.norm(baseline))

    part_1 = torch.sum(torch.cat(new_param_list, dim=1), dim=-1)

    return torch.dot(direction, (part_0-part_1))


def adaptive_attack(param_list, n_attackers, ctx, Q, V, args):
    v_tran = torch.cat(param_list, dim=1)
    direction = torch.sign(torch.sum(v_tran, dim=1, keepdims=True))

    n = len(param_list) - 1

    baseline = torch.squeeze(param_list[-1])
    cos_sim = []
    new_param_list = []
    
    for each_param_list in param_list:
        each_param_array = torch.squeeze(each_param_list)
        cos_sim_temp = torch.dot(baseline, each_param_array) / (torch.norm(baseline) + 1e-9) / (torch.norm(each_param_array) + 1e-9)
        cos_sim.append(cos_sim_temp)
    cos_sim = torch.stack(cos_sim)[:-1]

    zero = torch.tensor(np.zeros(cos_sim.shape)).to(ctx)
    cos_sim = torch.maximum(cos_sim, zero)
    
    normalized_weights = cos_sim / (torch.sum(cos_sim) + 1e-9)

    for i in range(n):
        new_param_list.append(param_list[i] * normalized_weights[i] / (torch.norm(param_list[i]) + 1e-9) * torch.norm(baseline))

    part_0 = torch.sum(torch.cat(new_param_list, dim=1), dim=-1)

    e_list = []
    for item in param_list:
        e_list.append(item / torch.norm(item))

    gamma = 0.1
    eta = 0.1
    for v in range(V):
        for i in range(n_attackers):
            for t in range(Q):
                temp = []
                for item in e_list:
                    temp.append(item)
                u = torch.randn(baseline.shape)
                temp[i] += gamma * u

                delta = (h(temp, ctx, part_0, direction) - h(e_list, ctx, part_0, direction)) / gamma * u
                e_list[i] += eta * delta


    for i in range(n_attackers):
        v[i] = torch.norm(param_list[i]) * e_list[i]

    return v


def ah(param_list, ctx, part_0, direction):

    n = len(param_list) - 1

    new_param_list = []
    
    weights = 1. / n

    for i in range(n):
        new_param_list.append(param_list[i] * weights)

    part_1 = torch.sum(torch.cat(new_param_list, dim=1), dim=-1)

    return torch.dot(direction, (part_0-part_1))

def adaptive_ours_attack(param_list, n_attackers, ctx, Q, V, args):
    v_tran = torch.cat(param_list, dim=1)
    direction = torch.sign(torch.sum(v_tran, dim=1, keepdims=True))

    n = len(param_list) - 1

    baseline = torch.squeeze(param_list[-1])
    new_param_list = []    
    
    weight = 1. / n

    for i in range(n):
        new_param_list.append(param_list[i] * weight)

    part_0 = torch.sum(torch.cat(new_param_list, dim=1), dim=-1)

    e_list = []
    for item in param_list:
        e_list.append(item / torch.norm(item))

    gamma = 0.1
    eta = 0.1
    for v in range(V):
        for i in range(n_attackers):
            for t in range(Q):
                temp = []
                for item in e_list:
                    temp.append(item)
                u = torch.randn(baseline.shape)
                temp[i] += gamma * u

                delta = (ah(temp, ctx, part_0, direction) - ah(e_list, ctx, part_0, direction)) / gamma * u
                e_list[i] += eta * delta


    for i in range(n_attackers):
        v[i] = torch.norm(param_list[i]) * e_list[i]

    return v


def minmax_agnostic(all_updates, net, lr, f, ctx, args):
    vi_shape = all_updates[0].shape
    v_tran = torch.cat(all_updates, dim=1)
    model_re = torch.mean(v_tran, dim=1).reshape(vi_shape)
    dev_type='unit_vec'

    if dev_type == 'unit_vec':
        deviation = model_re / torch.norm(model_re)  # unit vector, dir opp to good dir
    elif dev_type == 'sign':
        deviation = torch.sign(model_re)
    elif dev_type == 'std':
        deviation = torch.std(v_tran, 1)

    lamda = torch.Tensor([50.0]).float().cuda()
    threshold_diff = 1e-5
    lamda_fail = lamda
    lamda_succ = 0
    
    distances = []
    for update in all_updates:
        distance = torch.norm((v_tran - update), dim=0) ** 2
        distances = distance[:, None] if not len(distances) else torch.cat((distances, distance[:, None]), dim=1)

    max_distance = torch.max(distances)
    del distances

    while torch.abs(lamda_succ - lamda) > threshold_diff:
        mal_update = (model_re - lamda * deviation)
        distance = torch.norm((v_tran - mal_update), dim=0) ** 2
        max_d = torch.max(distance)
        
        if max_d <= max_distance:
            lamda_succ = lamda
            lamda = lamda + lamda_fail / 2
        else:
            lamda = lamda - lamda_fail / 2

        lamda_fail = lamda_fail / 2

    mal_update = (model_re - lamda_succ * deviation)
    res = []
    for i in range(f):
        res.append(mal_update)
        
    return res + all_updates


def minsum_agnostic(all_updates, net, lr, f, ctx, args): 
    vi_shape = all_updates[0].shape
    v_tran = torch.cat(all_updates, dim=1)
    model_re = torch.mean(v_tran, dim=1).reshape(vi_shape)
    dev_type='unit_vec'

    if dev_type == 'unit_vec':
        deviation = model_re / torch.norm(model_re)
    elif dev_type == 'sign':
        deviation = torch.sign(model_re)
    elif dev_type == 'std':
        deviation = torch.std(all_updates, 0)
    
    lamda = torch.Tensor([50.0]).float().cuda()
    threshold_diff = 1e-5
    lamda_fail = lamda
    lamda_succ = 0
    
    distances = []
    for update in all_updates:
        distance = torch.norm((v_tran - update), dim=0) ** 2
        distances = distance[:, None] if not len(distances) else torch.cat((distances, distance[:, None]), dim=1)

    scores = torch.sum(distances, dim=1)
    min_score = torch.min(scores)
    del distances

    while torch.abs(lamda_succ - lamda) > threshold_diff:
        mal_update = (model_re - lamda * deviation)
        distance = torch.norm((v_tran - mal_update), dim=0) ** 2
        score = torch.sum(distance)
        
        if score <= min_score:
            lamda_succ = lamda
            lamda = lamda + lamda_fail / 2
        else:
            lamda = lamda - lamda_fail / 2

        lamda_fail = lamda_fail / 2

    mal_update = (model_re - lamda_succ * deviation)
    res = []
    for i in range(f):
        res.append(mal_update)
        
    return res + all_updates


def lie_drift(v, net, lr, f, ctx, args):
    vi_shape = v[0].shape
    v_tran = torch.cat(v[f:], dim=1)
    grad_mean = torch.mean(v_tran, dim=1).reshape(vi_shape)
    grad_stdev = torch.std(v_tran, dim=1).reshape(vi_shape)
    upper_bound = grad_mean + 1.43 * grad_stdev
    lower_bound = grad_mean - 1.43 * grad_stdev
    for i in range(f):
        v[i] = torch.maximum(torch.minimum(v[i], upper_bound), lower_bound)
    return v

