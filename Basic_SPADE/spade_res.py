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


from spade import SPADE

class SPADEResBlk(nn.Module):
    def __init__(self, k, skip=False):
        super().__init__()
        kernel_size = 3
        self.skip = skip

        if self.skip:
            self.spade1 = SPADE(2 * k)
            self.conv1 = nn.Conv2d(2 * k, k, kernel_size=(kernel_size, kernel_size), padding=1, bias=False)
            self.spade_skip = SPADE(2 * k)
            self.conv_skip = nn.Conv2d(2 * k, k, kernel_size=(kernel_size, kernel_size), padding=1, bias=False)
        else:
            self.spade1 = SPADE(k)
            self.conv1 = nn.Conv2d(k, k, kernel_size=(kernel_size, kernel_size), padding=1, bias=False)

        self.spade2 = SPADE(k)
        self.conv2 = nn.Conv2d(k, k, kernel_size=(kernel_size, kernel_size), padding=1, bias=False)

    def forward(self, x, seg):
        x_skip = x
        x = F.relu(self.spade1(x, seg))
        x = self.conv1(x)
        x = F.relu(self.spade2(x, seg))
        x = self.conv2(x)

        if self.skip:
            x_skip = F.relu(self.spade_skip(x_skip, seg))
            x_skip = self.conv_skip(x_skip)

        return x_skip + x