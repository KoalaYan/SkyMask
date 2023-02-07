import torch
import torch.nn as nn
import torch.nn.functional as F
import mytorch as my
from collections import OrderedDict

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        # image_size = 28
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,30,kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(30,50,kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )

        self.fc1 = nn.Linear(1250, 100)
        self.fc2 = nn.Linear(100, 10)
        
    def forward(self, x):
        # x尺寸：(batch_size, image_channels, image_width, image_height)

        x = self.conv1(x)        
        x = self.conv2(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)

        x = F.relu(self.fc1(x))        
        x = self.fc2(x)

        x = F.log_softmax(x, dim = 0)

        return x



class CNNMaskNet(nn.Module):
    def __init__(self, param_list, num_workers, device):
        super(CNNMaskNet,self).__init__()

        self.num_workers = num_workers
        self.param_list = param_list
        self.device = device

        self.conv1 = nn.Sequential(OrderedDict([
            ('conv1', my.myconv2d(num_workers, device, [x[0] for x in param_list], [x[1] for x in param_list])),
            ('relu1', nn.ReLU()),
            ('maxpool1', nn.MaxPool2d(2,2))])
        )

        self.conv2 = nn.Sequential(OrderedDict([
            ('conv1', my.myconv2d(num_workers, device, [x[2] for x in param_list], [x[3] for x in param_list])),
            ('relu1', nn.ReLU()),
            ('maxpool1', nn.MaxPool2d(2,2))])
        )

        self.fc1 = my.mylinear(num_workers, device, [x[4] for x in param_list], [x[5] for x in param_list])
        self.fc2 = my.mylinear(num_workers, device, [x[6] for x in param_list], [x[7] for x in param_list])

    def update(self, param_list):
        self.param_list = param_list
        self.conv1.conv1.update([x[0] for x in param_list], [x[1] for x in param_list])
        self.conv2.conv1.update([x[2] for x in param_list], [x[3] for x in param_list])
        self.fc1.update([x[4] for x in param_list], [x[5] for x in param_list])
        self.fc2.update([x[6] for x in param_list], [x[7] for x in param_list])

    def forward(self, x):        
        x = self.conv1(x)        
        x = self.conv2(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)

        x = F.relu(self.fc1(x))        
        x = self.fc2(x)

        x = F.log_softmax(x, dim = 0)

        return x


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 16
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 16, 3, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 32, 3, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 64, 3, stride=2)
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        
        out = F.log_softmax(out, dim = 0)
        return out


def ResNet20():
    return ResNet(ResidualBlock)


class ResidualMaskBlock(nn.Module):
    def __init__(self, param_list, num_workers, device, stride=1):
        super(ResidualMaskBlock,self).__init__()

        self.num_workers = num_workers
        self.param_list = param_list
        self.device = device

        self.left = nn.Sequential(
            my.myconv2d(num_workers, device, [x[0] for x in param_list], stride=stride, padding=1),
            my.mybatch_norm(num_workers, device, [x[1] for x in param_list], [x[2] for x in param_list]),
            nn.ReLU(),
            my.myconv2d(num_workers, device, [x[5] for x in param_list], padding=1),
            my.mybatch_norm(num_workers, device, [x[6] for x in param_list], [x[7] for x in param_list])
        )
        self.shortcut = nn.Sequential()
        if len(param_list[0]) > 10:
            self.shortcut = nn.Sequential(
                my.myconv2d(num_workers, device, [x[10] for x in param_list], stride=stride),
                my.mybatch_norm(num_workers, device, [x[11] for x in param_list], [x[12] for x in param_list])
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out



class ResMaskNet(nn.Module):
    def __init__(self, param_list, num_workers, device):
        super(ResMaskNet,self).__init__()

        self.num_workers = num_workers
        self.param_list = param_list
        self.device = device
        self.conv1 = nn.Sequential(
            my.myconv2d(num_workers, device, [x[0] for x in param_list], stride=1, padding=1),
            my.mybatch_norm(num_workers, device, [x[1] for x in param_list], [x[2] for x in param_list]),
            nn.ReLU()
        )        
        self.layer1 = self.make_layer(ResidualMaskBlock, [x[5:35] for x in param_list], num_workers, device, 3, stride=1) # 10+10+10
        self.layer2 = self.make_layer(ResidualMaskBlock, [x[35:70] for x in param_list], num_workers, device, 3, stride=2) # 15+10+10
        self.layer3 = self.make_layer(ResidualMaskBlock, [x[70:105] for x in param_list], num_workers, device, 3, stride=2) # 15+10+10
        self.fc = my.mylinear(num_workers, device, [x[105] for x in param_list], [x[106] for x in param_list])

    def make_layer(self, block, param_list, num_workers, device, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        head = 0
        for stride in strides:
            if stride == 1:
                layers.append(block([x[head:head+10] for x in param_list], num_workers, device, stride))
                head += 10
            else:
                layers.append(block([x[head:head+15] for x in param_list], num_workers, device, stride))
                head += 15

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        out = F.log_softmax(out, dim = 0)
        return out


class LR(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(561, 6)

    def forward(self, x):
        y_pred= self.linear(x)
        out = F.log_softmax(y_pred, dim = 0)
        return out


class LRMaskNet(nn.Module):
    def __init__(self, param_list, num_workers, device):
        super(LRMaskNet,self).__init__()

        self.num_workers = num_workers
        self.param_list = param_list
        self.device = device
     
        self.fc = my.mylinear(num_workers, device, [x[0] for x in param_list], [x[1] for x in param_list])


    def forward(self, x):
        y_pred= self.fc(x)
        out = F.log_softmax(y_pred, dim = 0)
        return out