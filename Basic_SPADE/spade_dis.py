import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils import spectral_norm
import torchvision
from torchvision.utils import make_grid
from torchvision import transforms

import cv2
import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
#%matplotlib inline
#plt.rcParams['figure.figsize'] = (10, 5)

from pathlib import Path
from tqdm import tqdm
import os

from celeb_dataload import celeb
from spade_gen import SPADEGenerator



def custom_model1(in_chan, out_chan):
    return nn.Sequential(
        spectral_norm(nn.Conv2d(in_chan, out_chan, kernel_size=(4, 4), stride=2, padding=1, bias=False)),
        nn.LeakyReLU(inplace=True)
    )


def custom_model2(in_chan, out_chan, stride=2):
    return nn.Sequential(
        spectral_norm(nn.Conv2d(in_chan, out_chan, kernel_size=(4, 4), stride=stride, padding=1, bias=False)),
        nn.InstanceNorm2d(out_chan),
        nn.LeakyReLU(inplace=True)
    )


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = custom_model1(4, 64)
        self.layer2 = custom_model2(64, 128)
        self.layer3 = custom_model2(128, 256)
        self.layer4 = custom_model2(256, 512, stride=1)
        self.inst_norm = nn.InstanceNorm2d(512)
        self.conv = spectral_norm(nn.Conv2d(512, 1, kernel_size=(4, 4), padding=1))

    def forward(self, img, seg):
        x = torch.cat((img, seg), dim=1)
        print(x.size())
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.leaky_relu(self.inst_norm(x))
        x = self.conv(x)
        return x.squeeze()

path = '/Users/preethasaha/PycharmProjects/Insight_pr/pytorch/datasets/'
dataset = {
    x: celeb(path, split=x, is_transform=True) for x in ['train', 'val']
}

data = {
    x: DataLoader(dataset[x],
                  batch_size=1,
                  shuffle=True,
                  num_workers=0) for x in ['train', 'val']
}

iterator = iter(data['train'])
img, seg = next(iterator)

"""
noise = torch.rand(1, 256)
spade = SPADEGenerator()
out = spade(noise,seg)
print(out.size())
out = F.interpolate(out, size=256, mode='nearest')
print(out.size())
print(img.size(), seg.size())
"""

dis = Discriminator()
out2 = dis(img, seg)

print(out2.size())

