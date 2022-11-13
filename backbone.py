# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate
# The ResNet code is modified from https://github.com/plai-group/simple-cnaps

import torch
from torch.autograd import Variable
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.nn.utils.weight_norm import WeightNorm
import pdb
from torchvision import models
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Basic ResNet model
pretrained_path = "./checkpoint_models/Pretrained_ResNet_FETI.pt.tar"

def pretrain_load(pretrained_path):
    pretrained_dict = torch.load(pretrained_path)
    pretrained_dict['state_dict'] = {key.replace(
        "module.resnet.", ""): value for key, value in pretrained_dict['state_dict'].items()}
    return pretrained_dict


def init_layer(L):
    # Initialization using fan-in
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0]*L.kernel_size[1]*L.out_channels
        L.weight.data.normal_(0,math.sqrt(2.0/float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)
        
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
class CosineDistLinear(nn.Module):
    def __init__(self, indim, outdim):
        super(CosineDistLinear, self).__init__()
        self.L = nn.Linear( indim, outdim, bias = False)
        self.class_wise_learnable_norm = True  #See the issue#4&8 in the github 
        if self.class_wise_learnable_norm:      
            WeightNorm.apply(self.L, 'weight', dim=0) #split the weight update component to direction and norm      

        if outdim <=200:
            self.scale_factor = 2; #a fixed scale factor to scale the output of cos value into a reasonably large input for softmax, for to reproduce the result of CUB with ResNet10, use 4. see the issue#31 in the github 
        else:
            self.scale_factor = 10; #in Omniglot, a larger scale factor is required to handle >1000 output classes.

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim =1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm+ 0.00001)
        if not self.class_wise_learnable_norm:
            L_norm = torch.norm(self.L.weight.data, p=2, dim =1).unsqueeze(1).expand_as(self.L.weight.data)
            self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
        cos_dist = self.L(x_normalized) #matrix product by forward function, but when using WeightNorm, this also multiply the cosine distance by a class-wise learnable norm, see the issue#4&8 in the github
        scores = self.scale_factor* (cos_dist) 

        return scores

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        
    def forward(self, x):        
        return x.view(x.size(0), -1)

# Simple Conv Block


class ConvBlock(nn.Module):
    
    def __init__(self, indim, outdim, pool=True, padding=1):
        super(ConvBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.C = nn.Conv2d(indim, outdim, 3, padding=padding)
        self.BN = nn.BatchNorm2d(outdim)
        self.relu = nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C, self.BN, self.relu]
        if pool:
            self.pool = nn.MaxPool2d(2)
            self.parametrized_layers.append(self.pool)

        for layer in self.parametrized_layers:
            init_layer(layer)

        self.trunk = nn.Sequential(*self.parametrized_layers)

    def forward(self, x):
        out = self.trunk(x)
        return out

class ConvNet(nn.Module):
    def __init__(self, depth, dataset, flatten=True):
        super(ConvNet, self).__init__()
        trunk = []
        for i in range(depth):
            indim = 3 if i == 0 else 64
            outdim = 64
            # only pooling for fist 4 layers
            B = ConvBlock(indim, outdim, pool=i < 4)
            trunk.append(B)

        if flatten:
            trunk.append(Flatten())

        self.trunk = nn.Sequential(*trunk)
        dim = 4 if dataset =='CIFAR' else 5
        self.final_feat_dim = 64 * dim * dim if flatten else [64, dim, dim]

    def forward(self, x):
        out = self.trunk(x)
        return out

class ConvNetNopool(nn.Module): #Relation net use a 4 layer conv with pooling in only first two layers, else no pooling
    def __init__(self, depth, flatten=True):
        super(ConvNetNopool,self).__init__()
        trunk = []
        for i in range(depth):
            indim = 3 if i == 0 else 64
            outdim = 64
            B = ConvBlock(indim, outdim, pool = ( i in [0,1] ), padding = 0 if i in[0,1] else 1  ) #only first two layer has pooling and no padding
            trunk.append(B)
        if flatten:
            trunk.append(Flatten())
        
        self.trunk = nn.Sequential(*trunk)
        if flatten:
            self.final_feat_dim = 64 * 19 * 19
        else:
            self.final_feat_dim = [64, 19, 19]

    def forward(self,x):
        out = self.trunk(x)
        return out

class ConvNetS(nn.Module): #For Omniglot, only 1 input channel, output dim is 64
    def __init__(self, depth, flatten = True):
        super(ConvNetS,self).__init__()
        trunk = []
        for i in range(depth):
            indim = 1 if i == 0 else 64
            outdim = 64
            B = ConvBlock(indim, outdim, pool = ( i <4 ) ) #only pooling for fist 4 layers
            trunk.append(B)

        if flatten:
            trunk.append(Flatten())

        self.trunk = nn.Sequential(*trunk)
        self.final_feat_dim = 64

    def forward(self,x):
        out = x[:,0:1,:,:] #only use the first dimension
        out = self.trunk(out)
        return out

class ConvNetSNopool(nn.Module): #Relation net use a 4 layer conv with pooling in only first two layers, else no pooling. For Omniglot, only 1 input channel, output dim is [64,5,5]
    def __init__(self, depth, flatten=False):
        super(ConvNetSNopool,self).__init__()
        trunk = []
        for i in range(depth):
            indim = 1 if i == 0 else 64
            outdim = 64
            B = ConvBlock(indim, outdim, pool = ( i in [0,1] ), padding = 0 if i in[0,1] else 1  ) #only first two layer has pooling and no padding
            trunk.append(B)
        if (flatten):
            trunk.append(Flatten())

        self.trunk = nn.Sequential(*trunk)
        if (flatten):
            self.final_feat_dim = 64 * 19 * 19
        else:
            self.final_feat_dim = [64, 19, 19]

    def forward(self,x):
        out = x[:,0:1,:,:] #only use the first dimension
        out = self.trunk(out)
        return out

class ResNetModel():
    def __init__(self, dataset, variant = 34, flatten = False):
        super(ResNetModel, self).__init__()
        trunk = []
        dim = 4 if dataset == 'CIFAR' else 7
        self.final_feat_dim = 512 * dim * dim if flatten else [512, dim, dim]
        if variant ==18:
            resnet = models.resnet18(pretrained = True).to(device)  #pretrained on full ImageNet
        elif variant == 34:
            resnet = models.resnet34(pretrained = True).to(device)
        self.model = nn.Sequential(*[*resnet.children()][:-2])

    def forward(self,x):
        out = self.model(x)
        return out
class ResNet(nn.Module):
    def __init__(self, block, layers, flatten = False):
        super(ResNet, self).__init__()
        dim = 7
        self.final_feat_dim = 512 * dim * dim if flatten else [512, dim, dim]
        self.initial_pool = False
        inplanes = self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=5, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, inplanes, layers[0])
        self.layer2 = self._make_layer(
            block, inplanes * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(
            block, inplanes * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(
            block, inplanes * 8, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(7)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, param_dict=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.initial_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        return x


def Conv4(dataset, flatten=True):
    return ConvNet(4, dataset, flatten)


def Conv6(dataset, flatten=True):
    return ConvNet(6, dataset, flatten)


def Conv4NP(dataset, flatten=True):
    return ConvNetNopool(4, flatten)


def Conv6NP(dataset, flatten=True):
    return ConvNetNopool(6, flatten)


def Conv4S(dataset, flatten=True):
    return ConvNetS(4, flatten)


def Conv6S(dataset, flatten=True):
    return ConvNetS(6, flatten)


def Conv4SNP(dataset, flatten=True):
    return ConvNetSNopool(4, flatten)


def Conv6SNP(dataset, flatten=True):
    return ConvNetSNopool(6, flatten)


def ResNet12(FETI, dataset, flatten=True):
    if FETI:
        model = ResNet(BasicBlock, [2, 1, 1, 1], flatten)
        pretrained_dict = pretrain_load(pretrained_path)
        model.load_state_dict(pretrained_dict['state_dict'], strict=False)
    else:
        print("Torchvision.model does not support ResNet12. Change to ResNet18 instead.")
        model = ResNetModel(dataset, 18, flatten)
    return model


def ResNet18(FETI, dataset, flatten=True):
    if FETI:
        model = ResNet(BasicBlock, [2, 2, 2, 2], flatten)
        pretrained_dict = pretrain_load(pretrained_path)
        model.load_state_dict(pretrained_dict['state_dict'], strict=False)
    else:
        model = ResNetModel(dataset, 18, flatten)
    return model


def ResNet34(FETI, dataset, flatten=True):
    if FETI:
        model = ResNet(BasicBlock, [3, 4, 6, 3], flatten)
        pretrained_dict = pretrain_load(pretrained_path)
        model.load_state_dict(pretrained_dict['state_dict'], strict=False)
    else:
        model = ResNetModel(dataset, 34, flatten)
    return model
