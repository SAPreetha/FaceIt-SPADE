import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils import spectral_norm
import torchvision
from torchvision.utils import make_grid

import cv2
import numpy as np
from imageio import imread
import matplotlib.pyplot as plt




class SPADE(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.conv = spectral_norm(nn.Conv2d(1, 128, kernel_size=(3, 3), padding=1))
        self.conv_gamma = spectral_norm(nn.Conv2d(128, k, kernel_size=(3, 3), padding=1))
        self.conv_beta = spectral_norm(nn.Conv2d(128, k, kernel_size=(3, 3), padding=1))

    def forward(self, x, seg):
        # Store the sizes for easy reference
        N, C, H, W = x.size()

        sum_channel = torch.sum(x.reshape(N, C, H * W), dim=-1)
        mean = sum_channel / (N * H * W)
        std = torch.sqrt((sum_channel ** 2 - mean ** 2) / (N * H * W))

        mean = torch.unsqueeze(torch.unsqueeze(mean, -1), -1)
        std = torch.unsqueeze(torch.unsqueeze(std, -1), -1)
        x = (x - mean) / std

        seg = F.interpolate(seg, size=(H, W), mode='nearest')
        seg = F.relu(self.conv(seg))
        seg_gamma = self.conv_gamma(seg)
        seg_beta = self.conv_beta(seg)
        #print("seg_beta", seg_beta.size())
        #print("seg_gamma", seg_gamma.size())
        #print("x", x.size())

        x = torch.matmul(seg_gamma, x) + seg_beta
        #print("mul", torch.matmul(seg_gamma, x).size)

        return x
