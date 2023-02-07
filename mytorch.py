import torch
import torch.nn as nn
import torch.nn.functional as F

class myconv2d(nn.Module):
    def __init__(self, num_workers, device, weight_list, bias_list=[], stride=1, padding=0, dilation=1, groups=1):
        super(myconv2d,self).__init__()

        self.num_workers = num_workers
        self.weight_list = weight_list
        self.device = device
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.weight_mask = nn.ParameterList([nn.Parameter(torch.ones(size=self.weight_list[0].shape, device=self.device), requires_grad= True) for i in range(self.num_workers)])
        self.bias_list = bias_list
        if bias_list:
            self.bias_mask = nn.ParameterList([nn.Parameter(torch.ones(size=self.bias_list[0].shape, device=self.device), requires_grad= True) for i in range(self.num_workers)])
    
    def update(self, weight_list, bias_list=[]):
        self.weight_list = weight_list
        self.bias_list = bias_list

    def forward(self, x):
        weight_shape = [self.num_workers]
        for size in self.weight_list[0].shape:
            weight_shape.append(size)

        mask1 = [torch.sigmoid(self.weight_mask[idx]) for idx in range(self.num_workers)]
        weight_t = torch.div(torch.sum(torch.cat([torch.mul(mask1[idx], self.weight_list[idx]) for idx in range(self.num_workers)], dim=0).reshape(weight_shape), dim=0) \
            , torch.sum(torch.cat(mask1, dim=0).reshape(weight_shape), dim=0))
        
        if self.bias_list:
            bias_shape = [self.num_workers]
            
            for size in self.bias_list[0].shape:
                bias_shape.append(size)
            
            mask2 = [torch.sigmoid(self.bias_mask[idx]) for idx in range(self.num_workers)]
            bias_t = torch.div(torch.sum(torch.cat([torch.mul(mask2[idx], self.bias_list[idx]) for idx in range(self.num_workers)], dim=0).reshape(bias_shape), dim=0) \
                , torch.sum(torch.cat(mask2, dim=0).reshape(bias_shape), dim=0))

            out = F.conv2d(x, weight_t, bias_t, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        else:
            out = F.conv2d(x, weight_t, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)

        return out


class mylinear(nn.Module):
    def __init__(self, num_workers, device, weight_list, bias_list=[]):
        super(mylinear,self).__init__()

        self.num_workers = num_workers
        self.weight_list = weight_list
        self.device = device

        self.weight_mask = nn.ParameterList([nn.Parameter(torch.ones(size=self.weight_list[0].shape, device=self.device), requires_grad= True) for i in range(self.num_workers)])
        self.bias_list = bias_list
        if bias_list:
            self.bias_mask = nn.ParameterList([nn.Parameter(torch.ones(size=self.bias_list[0].shape, device=self.device), requires_grad= True) for i in range(self.num_workers)])

    def update(self, weight_list, bias_list=[]):
        self.weight_list = weight_list
        self.bias_list = bias_list

    def forward(self, x):
        weight_shape = [self.num_workers]
        for size in self.weight_list[0].shape:
            weight_shape.append(size)

        mask1 = [torch.sigmoid(self.weight_mask[idx]) for idx in range(self.num_workers)]
        weight_t = torch.div(torch.sum(torch.cat([torch.mul(mask1[idx], self.weight_list[idx]) for idx in range(self.num_workers)], dim=0).reshape(weight_shape), dim=0) \
            , torch.sum(torch.cat(mask1, dim=0).reshape(weight_shape), dim=0))
        
        if self.bias_list:
            bias_shape = [self.num_workers]
            
            for size in self.bias_list[0].shape:
                bias_shape.append(size)
            
            mask2 = [torch.sigmoid(self.bias_mask[idx]) for idx in range(self.num_workers)]
            bias_t = torch.div(torch.sum(torch.cat([torch.mul(mask2[idx], self.bias_list[idx]) for idx in range(self.num_workers)], dim=0).reshape(bias_shape), dim=0) \
                , torch.sum(torch.cat(mask2, dim=0).reshape(bias_shape), dim=0))

            out = F.linear(x, weight_t, bias_t)
        else:
            out = F.linear(x, weight_t)

        return out



class mybatch_norm(nn.Module):
    def __init__(self, num_workers, device, weight_list, bias_list=[]):
        super(mybatch_norm,self).__init__()

        self.num_workers = num_workers
        self.weight_list = weight_list
        self.device = device

        self.weight_mask = nn.ParameterList([nn.Parameter(torch.ones(size=self.weight_list[0].shape, device=self.device), requires_grad= True) for i in range(self.num_workers)])
        self.bias_list = bias_list
        if bias_list:
            self.bias_mask = nn.ParameterList([nn.Parameter(torch.ones(size=self.bias_list[0].shape, device=self.device), requires_grad= True) for i in range(self.num_workers)])

    def update(self, weight_list, bias_list=[]):
        self.weight_list = weight_list
        self.bias_list = bias_list
        
    def forward(self, x):
        # print(x.shape)
        weight_shape = [self.num_workers]
        for size in self.weight_list[0].shape:
            weight_shape.append(size)

        mask1 = [torch.sigmoid(self.weight_mask[idx]) for idx in range(self.num_workers)]
        weight_t = torch.div(torch.sum(torch.cat([torch.mul(mask1[idx], self.weight_list[idx]) for idx in range(self.num_workers)], dim=0).reshape(weight_shape), dim=0) \
            , torch.sum(torch.cat(mask1, dim=0).reshape(weight_shape), dim=0))
        
        m = torch.zeros_like(self.weight_list[0])
        v = torch.ones_like(self.weight_list[0])
        
        if self.bias_list:
            bias_shape = [self.num_workers]
            
            for size in self.bias_list[0].shape:
                bias_shape.append(size)
            
            mask2 = [torch.sigmoid(self.bias_mask[idx]) for idx in range(self.num_workers)]
            bias_t = torch.div(torch.sum(torch.cat([torch.mul(mask2[idx], self.bias_list[idx]) for idx in range(self.num_workers)], dim=0).reshape(bias_shape), dim=0) \
                , torch.sum(torch.cat(mask2, dim=0).reshape(bias_shape), dim=0))

            out = F.batch_norm(x, m, v, weight_t, bias_t)
        else:
            out = F.batch_norm(x, m, v, weight_t)

        return out