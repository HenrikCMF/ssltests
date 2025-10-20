import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms, models

class ResNet18v1(nn.Module):
    def __init__(self, emb_dim=128):
        super().__init__()
        base = models.resnet18()#(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        base.fc = nn.Identity()
        self.backbone = base
        self.proj = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(inplace=True),
            nn.Linear(512, emb_dim)
        )
    def forward(self, x, normalize=True):
        h = self.backbone(x)
        z = self.proj(h)
        return F.normalize(z, dim=1) if normalize else z
    

class ResNet18Projv2(nn.Module):
    def __init__(self, emb_dim=128):
        super().__init__()
        base = models.resnet18(weights=None)     # from scratch if you want purity
        base.fc = nn.Identity()
        self.backbone = base
        self.proj = nn.Sequential(
            nn.Linear(512, 2048, bias=False),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, emb_dim, bias=False),
            nn.BatchNorm1d(emb_dim, affine=False)  # no affine; improves SSL stability
        )
    def forward(self, x, normalize=True):
        h = self.backbone(x)
        z = self.proj(h)
        return F.normalize(z, dim=1) if normalize else z
    
    # add to ResNet18Proj
    def encode_backbone(self, x, normalize=True):
        h = self.backbone(x)               # [B,512]
        return F.normalize(h, dim=1) if normalize else h

