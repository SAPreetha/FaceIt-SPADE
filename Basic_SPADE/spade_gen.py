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


from spade_res import SPADEResBlk
from celeb_dataload import celeb

class SPADEGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(256, 16384)
        self.spade_resblk1 = SPADEResBlk(1024)
        self.spade_resblk2 = SPADEResBlk(1024)
        self.spade_resblk3 = SPADEResBlk(1024)
        self.spade_resblk4 = SPADEResBlk(512, skip=True)
        self.spade_resblk5 = SPADEResBlk(256, skip=True)
        self.spade_resblk6 = SPADEResBlk(128, skip=True)
        self.spade_resblk7 = SPADEResBlk(64, skip=True)
        self.conv = spectral_norm(nn.Conv2d(64, 3, kernel_size=(3, 3), padding=1))

    def forward(self, x, seg):
        b, c, h, w = seg.size()
        x = self.linear(x)
        x = x.view(b, 1024, 4, 4)

        x = F.interpolate(self.spade_resblk1(x, seg), size=(2 * h, 2 * w), mode='nearest')
        x = F.interpolate(self.spade_resblk2(x, seg), size=(4 * h, 4 * w), mode='nearest')
        x = F.interpolate(self.spade_resblk3(x, seg), size=(8 * h, 8 * w), mode='nearest')
        x = F.interpolate(self.spade_resblk4(x, seg), size=(16 * h, 16 * w), mode='nearest')
        x = F.interpolate(self.spade_resblk5(x, seg), size=(32 * h, 32 * w), mode='nearest')
        x = F.interpolate(self.spade_resblk6(x, seg), size=(64 * h, 64 * w), mode='nearest')
        x = F.interpolate(self.spade_resblk7(x, seg), size=(128 * h, 128 * w), mode='nearest')

        x = F.tanh(self.conv(x))

        return x

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
"""
iterator = iter(data['train'])
img, seg = next(iterator)
noise = torch.rand(1, 256)


spade = SPADEGenerator()
out = spade(noise,seg)

def imshow(img):
    img = img.detach().numpy().transpose((1, 2, 0))
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    plt.pause(0.001)

grid_img = make_grid(out, nrow=1)
imshow(grid_img)
plt.show()

"""