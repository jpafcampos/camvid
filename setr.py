from __future__ import print_function

import os.path as osp
import copy
import torch
from torch import einsum
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from torchvision.models.vgg import VGG
from torchvision.models.utils import load_state_dict_from_url
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class Setr(nn.Module):

    def __init__(self, vit_backbone, num_class, bilinear = False):
        super(Setr, self).__init__()

        self.vit_backbone = vit_backbone
        #self.channel_reduction = nn.Conv2d(in_channels=dim, out_channels=1024, kernel_size=3, padding=1)
        self.n_class = num_class
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(768, 256, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bn1     = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bn3     = nn.BatchNorm2d(256)
        self.deconv4 = nn.ConvTranspose2d(256, 32, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bn4     = nn.BatchNorm2d(32)
        
        self.up1 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.up_final = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)

        #if bilinear:
        #    self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        #    self.up_final = nn.Upsample(scale_factor=2, mode='bilinear')   	
        #else:
        #    self.up = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        #    self.up_final = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        
        self.classifier = nn.Conv2d(32, num_class, kernel_size=1)

    def forward(self, x):   
        bs = x.size(0)     
        score = self.vit_backbone(x)
        print(score.size())
        score = torch.reshape(score, (bs, 24, 24, 768))
        print(score.size())
        score = score.view(
            score.size(0),
            24,
            24,
            768
        )
        score = score.permute(0, 3, 1, 2).contiguous()
        #score = torch.transpose(score, 1, 3)
        print(score.size())

        score = self.up1(self.bn1(self.relu(self.deconv1(score))))    
        score = self.up2(self.bn2(self.relu(self.deconv2(score)))) 
        score = self.up3(self.bn3(self.relu(self.deconv3(score))))
        score = self.up_final(self.bn4(self.relu(self.deconv4(score))))  
        score = self.classifier(score)                   

        return score  # size=(N, n_class, x.H/1, x.W/1)  