from __future__ import print_function
import os
import os.path as osp
import copy
import torch
from torch import einsum
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import random

class ResViT(nn.Module):

    def __init__(self, pretrained_net, num_class, dim, depth, heads, batch_size, trans_img_size):
        super(ResViT, self).__init__()
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.batch_size = batch_size
        self.trans_img_size = trans_img_size
        #Transformer unit (encoder)
        self.transformer = ViT(
            image_size = trans_img_size,
            patch_size = 1,
            num_classes = 64, #not used
            dim = dim,
            depth = depth,    #number of encoders
            heads = heads,    #number of heads in self attention
            mlp_dim = 3072,   #hidden dimension in feedforward layer
            channels = 512,
            dropout = 0.1,
            emb_dropout = 0.1
        )
        self.channel_reduction = nn.Conv2d(in_channels=dim, out_channels=512, kernel_size=3, padding=1)
        self.n_class = num_class
        self.pretrained_net = pretrained_net
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(64)
        self.classifier = nn.Conv2d(64, num_class, kernel_size=1)

    def forward(self, x):
        #print(x.size())
        bs = x.size(0)
        out_resnet = self.pretrained_net(x)
        #x1 = out_resnet['feat1']    #H/4   256 ch
        #x2 = out_resnet['feat2']    #H/8   512 ch
        #x3 = out_resnet['feat3']    #H/16  1024 ch
        #x4 = out_resnet['feat4']    #H/32  2048 channels
        
        x3 = out_resnet['feat2']
        #print(x3.size())
        img_size = x3.shape[-2:]   
        
        x3 = self.transformer(x3)
        x3 = torch.reshape(x3, (bs, img_size[1], img_size[0], self.dim))
        x3 = torch.transpose(x3, 1, 3)
        x3 = self.channel_reduction(x3)

        score = self.bn1(self.relu(self.deconv1(x3)))    
        score = self.bn2(self.relu(self.deconv2(score))) 
        score = self.bn3(self.relu(self.deconv3(score)))
        score = self.classifier(score)                    

        return score  # size=(N, n_class, x.H/1, x.W/1)                   
